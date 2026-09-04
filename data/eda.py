"""
M1 数据探索性分析（EDA）脚本：统计并可视化 KolektorSDD 转换后的 YOLO 数据集。

输出目录 backend/artifacts/eda/：
  - stats_summary.csv    各划分（train/val/test）样本统计表
  - distributions.png    样本构成 + 缺陷框数量/面积/长宽比分布（4 子图）
  - samples.png          样例可视化（原图 / 缺陷掩码真值 / 转换后的 bbox 对比）

用途：验证「掩码→bbox」转换的正确性，并为报告提供数据集分析图表。

用法（项目根目录）：
    python backend/scripts/eda.py
    python backend/scripts/eda.py --samples 8 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，直接保存图片文件
import matplotlib.pyplot as plt
import numpy as np
import cv2

import io_utils  # 同目录工具：中文路径安全读写 + UTF-8 控制台输出

# 项目根目录（backend/scripts/ 向上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_DIR = PROJECT_ROOT / "datasets" / "kolektor_sdd" / "yolo_format"
RAW_TRAIN_DIR = PROJECT_ROOT / "datasets" / "kolektor_sdd" / "raw" / "train"
OUT_DIR = PROJECT_ROOT / "backend" / "artifacts" / "eda"

# Windows 中文字体，避免图表中文乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def collect_stats(yolo_dir: Path) -> dict[str, dict[str, int]]:
    """统计每个划分（train/val/test）的图像数、缺陷图像数、缺陷框总数。

    参数:
        yolo_dir: YOLO 格式数据集根目录

    返回:
        {split: {"images": int, "defect_images": int, "normal_images": int, "boxes": int}}
    """
    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        images = sorted(img_dir.glob("*.jpg"))
        n_defect, n_boxes = 0, 0
        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():  # 有标签文件 = 有缺陷
                n_defect += 1
                n_boxes += len(lbl.read_text().splitlines())
        stats[split] = {
            "images": len(images),
            "defect_images": n_defect,
            "normal_images": len(images) - n_defect,
            "boxes": n_boxes,
        }
    return stats


def read_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    """读取 YOLO 标签文件，返回归一化 bbox 列表 [(cx, cy, w, h), ...]。

    参数:
        label_path: 标签 txt 文件路径

    返回:
        归一化 bbox 元组列表
    """
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:  # class cx cy w h
            boxes.append(tuple(map(float, parts[1:5])))
    return boxes


def collect_box_shapes(labels_dir: Path) -> tuple[list[float], list[float], list[int]]:
    """统计缺陷框的归一化面积、长宽比与每图框数。

    参数:
        labels_dir: 某个划分的 labels 目录

    返回:
        (归一化面积列表, 长宽比列表, 每张缺陷图的框数列表)
    """
    areas: list[float] = []
    ratios: list[float] = []
    counts: list[int] = []
    for lbl in sorted(labels_dir.glob("*.txt")):
        boxes = read_boxes(lbl)
        counts.append(len(boxes))
        for _, _, bw, bh in boxes:
            areas.append(bw * bh)
            ratios.append(max(bw, bh) / (min(bw, bh) + 1e-9))
    return areas, ratios, counts


def plot_distributions(
    stats: dict[str, dict[str, int]], labels_dir: Path, out_path: Path
) -> None:
    """绘制 4 子图分布图并保存为 PNG。

    子图内容：各划分样本构成、每图缺陷框数量直方图、
             缺陷框归一化面积分布（对数刻度）、缺陷框长宽比分布。

    参数:
        stats: collect_stats 的返回结果
        labels_dir: 训练集 labels 目录（用于框统计）
        out_path: 输出图片路径
    """
    splits = list(stats.keys())
    defect = [stats[s]["defect_images"] for s in splits]
    normal = [stats[s]["normal_images"] for s in splits]
    areas, ratios, counts = collect_box_shapes(labels_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (1) 各划分样本构成（分组柱状图）
    x = np.arange(len(splits))
    axes[0, 0].bar(x - 0.2, defect, 0.4, label="缺陷图像", color="#e74c3c")
    axes[0, 0].bar(x + 0.2, normal, 0.4, label="正常图像", color="#2ecc71")
    axes[0, 0].set_xticks(x, splits)
    axes[0, 0].set_title("各划分样本构成")
    axes[0, 0].legend()

    # (2) 每张缺陷图的缺陷框数量直方图
    axes[0, 1].hist(counts, bins=range(1, max(counts) + 2), align="left",
                    color="#3498db", edgecolor="white")
    axes[0, 1].set_title("每张缺陷图的缺陷框数量分布")
    axes[0, 1].set_xlabel("框数")

    # (3) 缺陷框归一化面积分布（对数刻度，便于观察小缺陷长尾）
    axes[1, 0].hist(areas, bins=40, color="#9b59b6", edgecolor="white")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("缺陷框归一化面积分布（对数刻度）")
    axes[1, 0].set_xlabel("面积（占全图比例）")

    # (4) 缺陷框长宽比分布
    axes[1, 1].hist(ratios, bins=30, color="#f39c12", edgecolor="white")
    axes[1, 1].set_title("缺陷框长宽比分布")
    axes[1, 1].set_xlabel("长 / 宽")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[EDA] 分布图已保存: {out_path}")


def plot_samples(
    yolo_dir: Path, raw_dir: Path, out_path: Path, n: int = 6, seed: int = 42
) -> None:
    """随机抽取 n 张训练集缺陷图，每张三列对比：原图 / 掩码真值 / 转换后 bbox。

    用于人工抽查「掩码→bbox」转换是否正确。

    参数:
        yolo_dir: YOLO 格式数据集根目录
        raw_dir: 原始 train 目录（用于读取掩码）
        out_path: 输出图片路径
        n: 抽样数量
        seed: 随机种子
    """
    rng = random.Random(seed)
    label_files = sorted((yolo_dir / "labels" / "train").glob("*.txt"))
    samples = rng.sample(label_files, min(n, len(label_files)))

    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 2.6 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]  # 单样本时保持二维结构

    for row, lbl in enumerate(samples):
        name = lbl.stem                       # 形如 kos01_Part0
        img_path = yolo_dir / "images" / "train" / f"{name}.jpg"
        img = io_utils.imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)

        # 掩码与图像位于同名原始物理件目录下
        piece, part = name.split("_", 1)
        mask = io_utils.imread_unicode(raw_dir / piece / f"{part}_label.bmp",
                                       cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros_like(img)  # 防御：无掩码时用全黑代替

        boxes = read_boxes(lbl)
        h, w = img.shape[:2]

        # 左列：原图
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"{name} 原图")

        # 中列：掩码真值（红色叠加）
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay[mask > 127] = (0, 0, 255)
        axes[row, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[row, 1].set_title("缺陷掩码（真值）")

        # 右列：转换后的 bbox（绿色框）
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cx, cy, bw, bh in boxes:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axes[row, 2].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title("转换后 bbox（YOLO 标签）")

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[EDA] 样例对比图已保存: {out_path}（共 {len(samples)} 张）")


def write_stats_csv(stats: dict[str, dict[str, int]], out_path: Path) -> None:
    """把统计结果写为 CSV（utf-8-sig 编码，Excel 可直接打开）。

    参数:
        stats: collect_stats 的返回结果
        out_path: 输出 CSV 路径
    """
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["划分", "图像总数", "缺陷图像", "正常图像", "缺陷框总数"])
        for split, v in stats.items():
            writer.writerow([split, v["images"], v["defect_images"],
                             v["normal_images"], v["boxes"]])
    print(f"[EDA] 统计表已保存: {out_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="KolektorSDD 数据集 EDA 可视化")
    parser.add_argument("--samples", type=int, default=6, help="样例对比图抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """EDA 入口：统计并生成全部可视化产物。"""
    io_utils.force_utf8_stdout()
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 统计各划分样本
    stats = collect_stats(YOLO_DIR)
    print("=" * 60)
    for split, v in stats.items():
        print(f"[{split}] 图像 {v['images']:>4} 张 | 缺陷 {v['defect_images']:>4} 张 | "
              f"正常 {v['normal_images']:>4} 张 | bbox {v['boxes']:>5} 个")
    print("=" * 60)

    # 2. 输出统计表与可视化
    write_stats_csv(stats, OUT_DIR / "stats_summary.csv")
    plot_distributions(stats, YOLO_DIR / "labels" / "train", OUT_DIR / "distributions.png")
    plot_samples(YOLO_DIR, RAW_TRAIN_DIR, OUT_DIR / "samples.png",
                 n=args.samples, seed=args.seed)
    print("EDA 完成，输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()
