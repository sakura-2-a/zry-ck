"""
M2 YOLOv8 训练脚本：在 KolektorSDD 转换数据集上训练缺陷区域定位模型（一级检测）。

流程：
  1. 自动检测 GPU；无 GPU 时降级为轻量 CPU 配置（保证流程可跑通）
  2. 调用 ultralytics 训练 YOLOv8（默认数据增强 + 早停）
  3. 把最优权重 best.pt 复制到 backend/weights/yolo_best.pt
  4. 把验证集指标写入 backend/artifacts/yolo_train_metrics.json

用法（项目根目录，venv 已激活）：
    python backend/scripts/train_yolo.py                      # 默认配置（自动选择设备）
    python backend/scripts/train_yolo.py --model yolov8s.pt --epochs 150
    python backend/scripts/train_yolo.py --device cpu         # 强制 CPU（仅演示）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
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

# 默认训练参数（GPU 配置）
DEFAULT_CONFIG = {
    "model": "yolov8n.pt",   # 小数据集首选轻量模型；显存充足可换 yolov8s.pt
    "imgsz": 640,
    "epochs": 100,
    "batch": 8,
    "patience": 20,          # 验证指标连续 20 轮不提升则提前停止
}

# CPU 降级配置：小分辨率 + 小批次 + 少轮次，仅保证流程可跑通（速度很慢）
CPU_CONFIG = {
    "model": "yolov8n.pt",
    "imgsz": 416,
    "epochs": 20,
    "batch": 4,
    "patience": 5,
}


def detect_device(requested: str | None) -> str:
    """确定训练设备：显式指定 > 自动检测 GPU > CPU 降级。

    参数:
        requested: 用户通过命令行指定的设备（--device），None 表示自动检测

    返回:
        torch 设备字符串（"0" 表示 GPU 0，"cpu" 表示 CPU）
    """
    if requested:
        return requested
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[设备] 检测到 GPU: {gpu_name}（{vram_gb:.1f} GB），使用 CUDA 训练")
        return "0"
    print("[设备] 未检测到 GPU，降级为 CPU 轻量配置（训练较慢，仅保证流程可跑通）")
    return "cpu"


def train(config: dict, device: str) -> tuple[Path, dict]:
    """训练 YOLOv8 模型，固化最优权重并返回验证集指标。

    参数:
        config: 训练参数字典（model/imgsz/epochs/batch/patience）
        device: torch 设备字符串

    返回:
        (固化后的最优权重路径, 验证集指标字典)
    """
    model = YOLO(config["model"])
    print(f"[训练] 模型 {config['model']} | imgsz={config['imgsz']} | "
          f"epochs={config['epochs']} | batch={config['batch']} | 设备 {device}")

    results = model.train(
        data=str(DATA_YAML),
        imgsz=config["imgsz"],
        epochs=config["epochs"],
        batch=config["batch"],
        patience=config["patience"],
        device=device,
        project=str(RUNS_DIR),          # 训练输出目录：backend/runs/
        name="kolektor_yolo",           # 子目录名（exist_ok 时覆盖旧实验）
        exist_ok=True,
        rect=True,                      # 矩形训练：按长宽比组批，适合 500×1258 细长图
        seed=42,                        # 固定随机种子，保证实验可复现
        workers=0,                      # Windows 下多进程数据加载易出问题，单进程最稳
        verbose=True,
    )

    # ultralytics 会把最优权重保存在 <save_dir>/weights/best.pt
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise RuntimeError(f"未找到训练输出权重: {best_path}")

    # 固化权重到固定位置（后端推理与评估脚本统一从这里加载）
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weight_out = WEIGHTS_DIR / "yolo_best.pt"
    shutil.copy2(best_path, weight_out)
    print(f"[训练] 最优权重已固化: {weight_out}")

    # 提取验证集指标（ultralytics 结果字典，键带 (B) 后缀）
    rd = results.results_dict
    metrics = {
        "model": config["model"],
        "imgsz": config["imgsz"],
        "device": device,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "precision": round(float(rd.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(rd.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
        "best_weight": str(weight_out),
        "save_dir": str(results.save_dir),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS_DIR / "yolo_train_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[训练] 指标已保存: {out_json}")

    print("=" * 60)
    print(f"验证集指标: P={metrics['precision']}  R={metrics['recall']}  "
          f"mAP50={metrics['mAP50']}  mAP50-95={metrics['mAP50_95']}")
    print("=" * 60)
    return weight_out, metrics


def parse_args() -> argparse.Namespace:
    """解析命令行参数；未指定的参数使用默认/降级配置。"""
    parser = argparse.ArgumentParser(description="YOLOv8 缺陷检测模型训练")
    parser.add_argument("--model", type=str, default=None,
                        help="模型权重，如 yolov8n.pt / yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=None, help="训练输入分辨率")
    parser.add_argument("--epochs", type=int, default=None, help="最大训练轮数")
    parser.add_argument("--batch", type=int, default=None, help="批次大小")
    parser.add_argument("--device", type=str, default=None,
                        help="设备：0=GPU / cpu=CPU，默认自动检测")
    return parser.parse_args()


def main() -> None:
    """训练入口。"""
    io_utils.force_utf8_stdout()
    if not DATA_YAML.exists():
        raise SystemExit(
            f"[错误] 未找到数据集配置 {DATA_YAML}\n"
            "       请先运行: python backend/scripts/prepare_dataset.py"
        )

    args = parse_args()
    device = detect_device(args.device)
    config = DEFAULT_CONFIG.copy() if device != "cpu" else CPU_CONFIG.copy()

    # 命令行参数覆盖默认配置
    for key in ("model", "imgsz", "epochs", "batch"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    train(config, device)


if __name__ == "__main__":
    main()
