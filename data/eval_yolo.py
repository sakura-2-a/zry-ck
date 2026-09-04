"""
M2 YOLOv8 评估脚本：在官方 Test 集（80 张，完全未参与训练）上做 hold-out 评估。

输出（backend/artifacts/）：
  - yolo_{split}_metrics.json     评估指标
  - yolo_{split}_PR_curve.png     精确率-召回率曲线
  - yolo_{split}_confusion.png    混淆矩阵

用法（项目根目录，venv 已激活）：
    python backend/scripts/eval_yolo.py                          # 默认评估 Test 集
    python backend/scripts/eval_yolo.py --weights backend/weights/yolo_best.pt
    python backend/scripts/eval_yolo.py --split val              # 评估验证集
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

import io_utils  # 同目录工具：UTF-8 控制台输出（Windows 默认 GBK 会乱码/报错）

# 项目根目录（backend/scripts/ 向上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "datasets" / "kolektor_sdd" / "data.yaml"
WEIGHTS_DIR = PROJECT_ROOT / "backend" / "weights"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
RUNS_DIR = PROJECT_ROOT / "backend" / "runs"


def detect_device(requested: str | None) -> str:
    """确定评估设备（与 train_yolo.py 保持一致）。"""
    if requested:
        return requested
    if torch.cuda.is_available():
        print(f"[设备] 检测到 GPU: {torch.cuda.get_device_name(0)}")
        return "0"
    print("[设备] 未检测到 GPU，使用 CPU 评估")
    return "cpu"


def evaluate(weights: Path, split: str, device: str) -> dict:
    """在指定划分上评估 YOLO 模型，保存指标 JSON 与曲线图。

    参数:
        weights: 模型权重路径
        split: 评估划分（test / val）
        device: torch 设备字符串

    返回:
        评估指标字典
    """
    if not weights.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights}（请先运行 train_yolo.py）")

    model = YOLO(str(weights))
    results = model.val(
        data=str(DATA_YAML),
        split=split,
        device=device,
        project=str(RUNS_DIR),
        name=f"kolektor_eval_{split}",
        exist_ok=True,
    )

    # 提取指标（ultralytics 结果字典，键带 (B) 后缀）
    rd = results.results_dict
    metrics = {
        "weights": str(weights),
        "split": split,
        "device": device,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "precision": round(float(rd.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(rd.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
    }

    # 保存指标 JSON
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS_DIR / f"yolo_{split}_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 把 ultralytics 自动绘制的曲线图复制到 artifacts（供报告与前端使用）
    save_dir = Path(results.save_dir)
    for src_name, dst_name in (("PR_curve.png", f"yolo_{split}_PR_curve.png"),
                               ("confusion_matrix.png", f"yolo_{split}_confusion.png")):
        src = save_dir / src_name
        if src.exists():
            shutil.copy2(src, ARTIFACTS_DIR / dst_name)

    print("=" * 60)
    print(f"[评估] {weights.name} @ {split} 集")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  mAP50     : {metrics['mAP50']}")
    print(f"  mAP50-95  : {metrics['mAP50_95']}")
    print(f"  指标文件  : {out_json}")
    print("=" * 60)
    return metrics


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YOLOv8 缺陷检测模型评估")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_DIR / "yolo_best.pt",
                        help="模型权重路径")
    parser.add_argument("--split", type=str, default="test", choices=("test", "val"),
                        help="评估划分")
    parser.add_argument("--device", type=str, default=None,
                        help="设备：0=GPU / cpu=CPU，默认自动检测")
    return parser.parse_args()


def main() -> None:
    """评估入口。"""
    io_utils.force_utf8_stdout()
    if not DATA_YAML.exists():
        raise SystemExit(
            f"[错误] 未找到数据集配置 {DATA_YAML}\n"
            "       请先运行: python backend/scripts/prepare_dataset.py"
        )
    args = parse_args()
    device = detect_device(args.device)
    evaluate(args.weights, args.split, device)


if __name__ == "__main__":
    main()
