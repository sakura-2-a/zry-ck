"""
M1 数据准备脚本：整理 KolektorSDD 原始数据并转换为 YOLOv8 检测格式。

处理流程：
  1. 把原始数据（Train/Test）拷贝到 datasets/kolektor_sdd/raw/
  2. 像素掩码 → 二值化 → 连通域分析 → YOLO 归一化 bbox（类别 0 = defect）
  3. 按「物理件 kosXX」分层划分 train / val / test（防止同一物理件的图像跨划分造成数据泄漏）
  4. 生成 datasets/kolektor_sdd/data.yaml 与 split_info.json

说明：
  - KolektorSDD 原始标注只有像素掩码（PartX_label.bmp），没有检测框，
    本脚本用连通域分析自动生成 YOLOv8 训练所需的 bbox 标签；
  - 无缺陷（正常）图像不生成标签文件，YOLO 将其作为背景样本参与训练。

用法（项目根目录）：
    python backend/scripts/prepare_dataset.py
    python backend/scripts/prepare_dataset.py --val-ratio 0.2 --min-area 25 --seed 42
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

import io_utils  # 同目录工具：中文路径安全读写 + UTF-8 控制台输出

# 项目根目录（本文件位于 backend/scripts/，向上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认原始数据位置与本项目内输出位置
DEFAULT_SOURCE = Path("C:/Users/Administrator/Desktop/KolektorSDD/KolektorSDD")
DEFAULT_DST = PROJECT_ROOT / "datasets" / "kolektor_sdd"

CLASS_NAMES = ["defect"]  # 类别索引 0 = 缺陷


# ---------------------------------------------------------------------------
# 第一步：拷贝原始数据
# ---------------------------------------------------------------------------
def copy_raw_dataset(src: Path, dst: Path) -> None:
    """把原始数据集的 Train/Test 目录拷贝到项目 datasets/kolektor_sdd/raw/ 下。

    参数:
        src: 原始数据集根目录（内含 Train/ 与 Test/ 两个子目录）
        dst: 项目内数据集根目录
    """
    raw_dir = dst / "raw"
    if raw_dir.exists():
        print(f"[1/4] 跳过拷贝：{raw_dir} 已存在")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in ("Train", "Test"):
        shutil.copytree(src / split, raw_dir / split.lower())
        print(f"[1/4] 拷贝 {split} -> {raw_dir / split.lower()}")


# ---------------------------------------------------------------------------
# 第二步：掩码 → bbox 转换（核心算法）
# ---------------------------------------------------------------------------
def read_mask(img_path: Path) -> np.ndarray | None:
    """读取图像对应的缺陷掩码（PartX_label.bmp）。

    参数:
        img_path: 图像文件路径

    返回:
        灰度掩码数组；若掩码文件不存在（无标注的正常样本）返回 None
    """
    mask_path = img_path.with_name(f"{img_path.stem}_label.bmp")
    if not mask_path.exists():
        return None
    return io_utils.imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)


def mask_to_bboxes(mask: np.ndarray | None, min_area: int = 25) -> list[list[float]]:
    """把像素掩码转换为 YOLO 归一化 bbox 列表（中心点 + 宽高格式）。

    算法：二值化 → 连通域分析（8 邻域）→ 面积不小于 min_area 的连通域
          取外接矩形 → 归一化到 [0, 1]。

    参数:
        mask: 灰度掩码（缺陷像素为白色）；None 表示无缺陷
        min_area: 连通域最小面积（像素），小于该值视为噪声丢弃

    返回:
        形如 [[cx, cy, w, h], ...] 的归一化 bbox 列表；无缺陷返回空列表
    """
    if mask is None:
        return []
    h, w = mask.shape[:2]
    # 掩码可能带抗锯齿中间灰度，>127 视为缺陷像素
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return []  # 全黑掩码 = 无缺陷

    # 连通域分析：stats[i] = [x, y, w, h, area]，其中 i=0 为背景
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes: list[list[float]] = []
    for x, y, bw, bh, area in stats[1:]:
        if area < min_area:
            continue  # 过滤噪声小区域
        # 外接矩形 → YOLO 归一化 (中心x, 中心y, 宽, 高)
        boxes.append([
            round((x + bw / 2) / w, 6),
            round((y + bh / 2) / h, 6),
            round(bw / w, 6),
            round(bh / h, 6),
        ])
    return boxes


# ---------------------------------------------------------------------------
# 第三步：按物理件分层划分 train / val
# ---------------------------------------------------------------------------
def piece_has_defect(piece_dir: Path) -> bool:
    """判断一个物理件（kosXX 目录）是否含缺陷图像。

    参数:
        piece_dir: 物理件目录（如 raw/train/kos01）

    返回:
        该物理件下存在任意一张缺陷图像（掩码非空）返回 True
    """
    for img_path in sorted(piece_dir.glob("*.jpg")):
        mask = read_mask(img_path)
        if mask is not None and (mask > 127).sum() > 0:
            return True
    return False


def stratified_split_by_piece(
    pieces: list[Path], val_ratio: float, seed: int
) -> dict[str, list[str]]:
    """按物理件分层（stratified）划分 train/val。

    设计要点：KolektorSDD 中每个 kosXX 目录是同一个物理产品的多张子图像，
    内容高度相似。若把同一物理件的图像分到 train 和 val 两边，模型会在验证集
    上"见过"几乎一样的图，造成数据泄漏、指标虚高。因此本函数以物理件为
    最小划分单元，并按「有缺陷/无缺陷」分层抽样保证验证集类别平衡。

    参数:
        pieces: 官方 Train 集内的物理件目录列表
        val_ratio: 验证集占比（按物理件数量计）
        seed: 随机种子（保证划分可复现）

    返回:
        {"train": [kosXX, ...], "val": [kosXX, ...]}
    """
    rng = np.random.default_rng(seed)
    pos = sorted(p.name for p in pieces if piece_has_defect(p))
    neg = sorted(p.name for p in pieces if not piece_has_defect(p))
    print(f"[2/4] 官方 Train 共 {len(pieces)} 个物理件：含缺陷 {len(pos)} / 无缺陷 {len(neg)}")

    def _split(names: list[str]) -> tuple[list[str], list[str]]:
        """单个分层内随机划分；样本不足 2 个时全部归 train。"""
        if len(names) < 2:
            return names, []
        n_val = max(1, int(round(len(names) * val_ratio)))
        idx = rng.permutation(len(names))
        val = sorted(names[i] for i in idx[:n_val])
        train = sorted(names[i] for i in idx[n_val:])
        return train, val

    train_pos, val_pos = _split(pos)
    train_neg, val_neg = _split(neg)
    return {"train": train_pos + train_neg, "val": val_pos + val_neg}


# ---------------------------------------------------------------------------
# 第四步：生成 YOLO 格式数据集与配置文件
# ---------------------------------------------------------------------------
def build_yolo_dataset(
    dst: Path, split_map: dict[str, list[str]], min_area: int
) -> dict[str, dict[str, int]]:
    """遍历所有图像，生成 YOLO 格式 images/ 与 labels/ 目录。

    参数:
        dst: 数据集根目录（datasets/kolektor_sdd）
        split_map: {"train": [kosXX, ...], "val": [...], "test": [...]}
        min_area: 连通域最小面积

    返回:
        各划分统计 {"train": {...}, "val": {...}, "test": {...}}
    """
    yolo_dir = dst / "yolo_format"
    raw_root = dst / "raw"
    stats: dict[str, dict[str, int]] = {}

    for split, pieces in split_map.items():
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        # val 来自官方 Train 部分；test 来自官方 Test 部分
        raw_split = "train" if split in ("train", "val") else "test"

        n_img = n_def = n_box = 0
        for piece in pieces:
            for img_path in sorted((raw_root / raw_split / piece).glob("*.jpg")):
                new_stem = f"{piece}_{img_path.stem}"   # 如 kos01_Part0
                shutil.copy2(img_path, yolo_dir / "images" / split / f"{new_stem}.jpg")
                n_img += 1
                boxes = mask_to_bboxes(read_mask(img_path), min_area)
                if boxes:
                    n_def += 1
                    n_box += len(boxes)
                    # YOLO 标签：每行 "class cx cy w h"
                    with open(yolo_dir / "labels" / split / f"{new_stem}.txt", "w", encoding="utf-8") as f:
                        for b in boxes:
                            f.write("0 " + " ".join(f"{v:.6f}" for v in b) + "\n")
                # 无缺陷图像不写标签文件（YOLO 视为背景样本）

        stats[split] = {
            "images": n_img,
            "defect_images": n_def,
            "normal_images": n_img - n_def,
            "boxes": n_box,
        }
        print(f"[3/4] {split}: 图像 {n_img} 张（缺陷 {n_def} / 正常 {n_img - n_def}），bbox 共 {n_box} 个")
    return stats


def write_data_yaml(dst: Path) -> Path:
    """生成 YOLO 训练使用的 data.yaml 配置文件。

    参数:
        dst: 数据集根目录（datasets/kolektor_sdd）

    返回:
        生成的 data.yaml 路径
    """
    content = {
        "path": str((dst / "yolo_format").resolve()),  # 绝对路径，训练脚本直接可用
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",  # test 仅用于最终评估，不参与训练
        "nc": 1,
        "names": CLASS_NAMES,
    }
    yaml_path = dst / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)
    print(f"[4/4] 生成数据集配置: {yaml_path}")
    return yaml_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="KolektorSDD 数据准备：掩码→bbox、分层划分、生成 YOLO 数据集"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="原始数据集根目录")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="项目内数据集输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集占比（按物理件数）")
    parser.add_argument("--min-area", type=int, default=25, help="连通域最小面积（像素）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """数据准备流程入口。"""
    io_utils.force_utf8_stdout()
    args = parse_args()
    print("=" * 60)
    print("KolektorSDD 数据准备（掩码→bbox → 分层划分 → YOLO 格式）")
    print("=" * 60)

    # 1. 拷贝原始数据
    copy_raw_dataset(args.source, args.dst)

    # 2. 按物理件分层划分（test 沿用官方划分）
    raw_train = args.dst / "raw" / "train"
    raw_test = args.dst / "raw" / "test"
    split_map = stratified_split_by_piece(sorted(raw_train.iterdir()), args.val_ratio, args.seed)
    split_map["test"] = sorted(p.name for p in raw_test.iterdir())

    # 3. 生成 YOLO 格式数据集
    stats = build_yolo_dataset(args.dst, split_map, args.min_area)

    # 4. 生成配置文件与划分信息
    write_data_yaml(args.dst)
    split_info = {
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "min_area": args.min_area,
        "pieces": split_map,
        "stats": stats,
    }
    info_path = args.dst / "split_info.json"
    info_path.write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"划分信息已保存: {info_path}")

    # 5. 校验：验证集类别平衡提示
    val = stats.get("val", {})
    if val.get("normal_images", 0) == 0 or val.get("defect_images", 0) == 0:
        print("⚠ 警告：验证集缺少某个类别（缺陷/正常），后续 SVM 评估时请注意")
    print("完成！")


if __name__ == "__main__":
    main()
