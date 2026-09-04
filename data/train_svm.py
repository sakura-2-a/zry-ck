"""
M3 SVM 训练脚本：在缓存特征上训练二级判别 SVM，并评估验证集表现。

输出（backend/artifacts/）：
  svm_classifier.joblib    {scaler, svm, feature_config, conf_th, prob_th}
                           后端推理与 GA 最终固化共用同一格式
  svm_train_metrics.json   验证集分类指标（acc/P/R/F1/AUC）+ 流水线检测指标

说明：本脚本用人工给定的超参数训练（作为 GA 优化的对比基线），
      GA（M4）运行后会用其最优参数覆盖 svm_classifier.joblib。

用法（项目根目录，venv 已激活）：
    python backend/scripts/train_svm.py                                # 默认超参数
    python backend/scripts/train_svm.py --C 10 --gamma 0.01 --kernel rbf
    python backend/scripts/train_svm.py --conf-th 0.25 --prob-th 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib

# 同目录共享模块（特征计算 / 数据加载 / SVM 训练评估 + UTF-8 控制台）
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import io_utils
import svm_core

# 项目根目录（backend/scripts/ 向上两级）
PROJECT_ROOT = SCRIPT_DIR.parents[1]
BACKEND_DIR = SCRIPT_DIR.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"


def parse_gamma(value: str) -> float | str:
    """解析 gamma 参数：数字字符串转 float，其余（scale/auto）原样返回。

    参数:
        value: 命令行传入的 gamma 字符串

    返回:
        float 或 "scale"/"auto"
    """
    try:
        return float(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SVM 二级判别器训练")
    parser.add_argument("--C", type=float, default=1.0, help="SVM 惩罚系数")
    parser.add_argument("--gamma", type=str, default="scale",
                        help="RBF 核宽度（数字或 scale/auto）")
    parser.add_argument("--kernel", type=str, default="rbf",
                        choices=("rbf", "linear", "poly"), help="核函数")
    parser.add_argument("--conf-th", type=float, default=0.001,
                        help="YOLO 置信度阈值（训练后模型 conf 普遍 <0.02，GA 会进一步优化）")
    parser.add_argument("--prob-th", type=float, default=0.5, help="SVM 概率阈值")
    return parser.parse_args()


def main() -> None:
    """SVM 训练入口。"""
    io_utils.force_utf8_stdout()
    args = parse_args()
    train = svm_core.load_feature_set(ARTIFACTS_DIR / "svm_features_train.npz")
    val = svm_core.load_feature_set(ARTIFACTS_DIR / "svm_features_val.npz")

    # 按 YOLO 置信度阈值过滤训练候选框（与流水线推理逻辑一致）
    keep = train["conf"] >= args.conf_th
    print(f"[训练] 过滤后样本 {int(keep.sum())} 个"
          f"（正 {int(train['y'][keep].sum())} / 负 {int((train['y'][keep] == 0).sum())}）")

    scaler, svm = svm_core.train_svm(
        train["X"][keep], train["y"][keep],
        C=args.C, gamma=parse_gamma(args.gamma), kernel=args.kernel,
    )

    # 验证集评估：分类器本身 + 两级流水线
    cls_metrics = svm_core.evaluate_samples(svm, scaler, val["X"], val["y"])
    pipe_metrics = svm_core.evaluate_pipeline(
        svm, scaler, val["X"], val["y"], val["conf"], args.conf_th, args.prob_th,
    )

    # 固化模型（后端推理加载此文件）
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "scaler": scaler,
        "svm": svm,
        "feature_config": svm_core.FEATURE_CONFIG,
        "conf_th": args.conf_th,
        "prob_th": args.prob_th,
    }
    joblib.dump(bundle, ARTIFACTS_DIR / "svm_classifier.joblib")

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"C": args.C, "gamma": args.gamma, "kernel": args.kernel,
                   "conf_th": args.conf_th, "prob_th": args.prob_th},
        "classifier_metrics": cls_metrics,
        "pipeline_metrics": pipe_metrics,
    }
    out_json = ARTIFACTS_DIR / "svm_train_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("[SVM 分类器指标（验证集）]")
    for k, v in cls_metrics.items():
        print(f"  {k:>10}: {v}")
    print("[两级流水线指标（验证集）]")
    for k, v in pipe_metrics.items():
        print(f"  {k:>10}: {v}")
    print(f"模型已固化: {ARTIFACTS_DIR / 'svm_classifier.joblib'}")
    print(f"指标已保存: {out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
