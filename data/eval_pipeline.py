"""图像级流水线评测：对 val/test 每张图像跑「YOLO+SVM」两级流水线，
按图像级判定（is_defective）与真值（是否有 GT 缺陷框）计算指标。

与样本级 F1（GA 适应度）互补：样本级刻画框级定位质量，本脚本刻画
最终交付口径——"这张轴承图是否被判为缺陷"。

用法（项目根目录，venv 已激活）：
    python backend/scripts/eval_pipeline.py                 # 用当前固化模型评 val+test
    python backend/scripts/eval_pipeline.py --splits test
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io_utils  # noqa: E402
from app.services.detector import DetectionPipeline  # noqa: E402

YOLO_DIR = PROJECT_ROOT / "datasets" / "kolektor_sdd" / "yolo_format"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="图像级两级流水线评测")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="要评测的划分")
    parser.add_argument("--conf-th", type=float, default=None,
                        help="覆盖 YOLO 阈值（默认用模型内置 GA 最优阈值）")
    parser.add_argument("--prob-th", type=float, default=None,
                        help="覆盖 SVM 阈值（默认用模型内置 GA 最优阈值）")
    parser.add_argument("--top-k", type=int, default=None,
                        help="覆盖每图送入 SVM 的候选上限（默认 20，与训练一致）")
    return parser.parse_args()


def evaluate_split(pipeline: DetectionPipeline, split: str) -> dict:
    """对一个划分逐图评测，返回图像级指标与逐图明细。

    参数:
        pipeline: 已加载的检测流水线
        split: val / test

    返回:
        {"total", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
         "accuracy", "details": [...]}
    """
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split

    tp = fp = fn = tn = 0
    details = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        has_gt = (lbl_dir / f"{img_path.stem}.txt").exists()
        result = pipeline.detect(img_path.read_bytes(), img_path.name)
        pred = result["is_defective"]
        if has_gt and pred:
            tp += 1
        elif not has_gt and pred:
            fp += 1
        elif has_gt and not pred:
            fn += 1
        else:
            tn += 1
        details.append({"image": img_path.name, "gt": has_gt,
                        "pred": pred, "num_boxes": result["num_defects"],
                        "avg_conf": result["avg_confidence"]})

    total = tp + fp + fn + tn
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return {
        "total": total, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall + 1e-9), 4),
        "accuracy": round((tp + tn) / (total + 1e-9), 4),
        "details": details,
    }


def main() -> None:
    """图像级评测入口。"""
    io_utils.force_utf8_stdout()
    args = parse_args()
    pipeline = DetectionPipeline()
    pipeline.load()
    if args.conf_th is not None:
        pipeline.conf_th = args.conf_th
    if args.prob_th is not None:
        pipeline.prob_th = args.prob_th
    if args.top_k is not None:
        pipeline.top_k = args.top_k

    report = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "conf_th": pipeline.conf_th,
        "prob_th": pipeline.prob_th,
        "splits": {},
    }
    for split in args.splits:
        metrics = evaluate_split(pipeline, split)
        report["splits"][split] = metrics
        print(f"[{split}] 图像 {metrics['total']} 张 | TP={metrics['tp']} FP={metrics['fp']} "
              f"FN={metrics['fn']} TN={metrics['tn']} | P={metrics['precision']} "
              f"R={metrics['recall']} F1={metrics['f1']} ACC={metrics['accuracy']}")

    out = ARTIFACTS_DIR / "pipeline_image_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存: {out}")


if __name__ == "__main__":
    main()
