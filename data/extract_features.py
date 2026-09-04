"""
M3 特征提取脚本：用训练好的 YOLO 模型在数据集上推理，收集缺陷候选框，
为每个候选框计算「HOG + LBP + 统计」特征并缓存，供 SVM 训练（M3）与
GA 优化（M4）复用（GA 每代评估只需训练 SVM，无需重复提取特征）。

样本标签定义（关键决策）：
  - 候选框与任一真值框（掩码连通域生成的 bbox）IoU ≥ 0.5 → 正样本（真缺陷）
  - 其余候选框（YOLO 误检）→ 负样本
  - YOLO 漏检的真值框（无候选框与之 IoU ≥ 0.5）→ 追加为正样本
  - 若负样本不足，随机采样背景块（与真值框 IoU < 0.1）补齐，保持类别平衡

缓存文件（backend/artifacts/）：
  svm_features_{split}.npz   X=特征矩阵 / y=标签 / conf=YOLO置信度 /
                             boxes=候选框xyxy / names=图像文件名

用法（项目根目录，venv 已激活）：
    python backend/scripts/extract_features.py
    python backend/scripts/extract_features.py --conf 0.05 --device 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import io_utils  # 同目录工具：中文路径安全读写 + UTF-8 控制台输出

# 同目录共享模块（特征计算 / 数据加载 / SVM 训练评估）
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import svm_core

# 项目根目录（backend/scripts/ 向上两级）；权重与产物在 backend/ 下
PROJECT_ROOT = SCRIPT_DIR.parents[1]
BACKEND_DIR = SCRIPT_DIR.parent
WEIGHTS_DIR = BACKEND_DIR / "weights"
YOLO_DIR = PROJECT_ROOT / "datasets" / "kolektor_sdd" / "yolo_format"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

IOU_POS_TH = 0.5    # 候选框与真值框 IoU ≥ 该值视为正样本
IOU_NEG_TH = 0.1    # 背景块与真值框 IoU < 该值才可作为负样本


def detect_device(requested: str | None) -> str:
    """确定推理设备（与训练脚本逻辑一致）。"""
    if requested:
        return requested
    if torch.cuda.is_available():
        print(f"[设备] 检测到 GPU: {torch.cuda.get_device_name(0)}")
        return "0"
    print("[设备] 未检测到 GPU，使用 CPU 推理")
    return "cpu"


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算两个 xyxy 框的 IoU（交并比）。

    参数:
        box_a, box_b: 形如 [x1, y1, x2, y2] 的边界框

    返回:
        IoU ∈ [0, 1]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """从 YOLO 标签文件读取真值框（归一化 xywh → 绝对 xyxy）。

    参数:
        label_path: 标签 txt 路径（不存在则返回空数组）
        img_w, img_h: 图像宽高

    返回:
        (m, 4) 的 xyxy 真值框数组；无标签返回 shape (0, 4)
    """
    if not label_path.exists():
        return np.empty((0, 4))
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=float).reshape(-1, 4) if boxes else np.empty((0, 4))


def crop_box(img: np.ndarray, box_xyxy: np.ndarray, pad_ratio: float = 0.1) -> np.ndarray | None:
    """按 bbox 裁剪图像区域，外扩 pad_ratio（10%）并裁剪到图像边界。

    参数:
        img: 原图（BGR ndarray）
        box_xyxy: [x1, y1, x2, y2] 框
        pad_ratio: 外扩比例（为特征保留少量上下文）

    返回:
        裁剪图；裁剪结果过小（<4px）返回 None
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad_ratio))
    y1 = max(0, int(y1 - bh * pad_ratio))
    x2 = min(w, int(x2 + bw * pad_ratio))
    y2 = min(h, int(y2 + bh * pad_ratio))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img[y1:y2, x1:x2]


def random_negative_patch(
    img: np.ndarray, gt_boxes: np.ndarray, rng: np.random.Generator,
    min_size: int = 64, max_size: int = 256, max_tries: int = 20,
) -> np.ndarray | None:
    """随机采样与所有真值框 IoU < 0.1 的背景块，用作负样本。

    参数:
        img: 原图
        gt_boxes: (m, 4) 真值框
        rng: 随机数生成器
        min_size, max_size: 背景块边长范围（像素）
        max_tries: 单张图最大尝试次数

    返回:
        [x1, y1, x2, y2] 框；采样失败返回 None
    """
    h, w = img.shape[:2]
    for _ in range(max_tries):
        pw = int(rng.integers(min_size, min(max_size, w // 2) + 1))
        ph = int(rng.integers(min_size, min(max_size, h // 2) + 1))
        x1 = int(rng.integers(0, max(1, w - pw + 1)))
        y1 = int(rng.integers(0, max(1, h - ph + 1)))
        box = np.array([x1, y1, x1 + pw, y1 + ph], dtype=float)
        if len(gt_boxes) == 0 or max(iou(box, g) for g in gt_boxes) < IOU_NEG_TH:
            return box
    return None


def random_edge_strip(
    img: np.ndarray, rng: np.random.Generator,
    thickness: tuple[int, int] = (16, 48), length: tuple[int, int] = (128, 400),
    max_tries: int = 20,
) -> np.ndarray | None:
    """采样图像边缘条带块（覆盖轴承端面/边缘轮廓纹理的负样本形态）。

    轴承表面图像四周存在端面台阶、倒角等加工纹理，与划痕在 HOG/LBP
    特征空间中有一定混淆（SVM 曾把右下角扁平条带判为缺陷，conf≈0.99）。
    方形随机块覆盖不到这类扁平长条形态，故单独采样边缘条带。

    参数:
        img: 原图
        rng: 随机数生成器
        thickness: 条带厚度范围（垂直边的宽度 / 水平边的高度，像素）
        length: 条带沿边长度范围（像素）
        max_tries: 最大尝试次数

    返回:
        [x1, y1, x2, y2] 框；采样失败返回 None
    """
    h, w = img.shape[:2]
    for _ in range(max_tries):
        edge = rng.integers(0, 4)          # 0上 1下 2左 3右
        t = int(rng.integers(*thickness))
        ln = int(rng.integers(*length))
        if edge == 0:
            x1 = int(rng.integers(0, max(1, w - ln + 1)))
            box = np.array([x1, 0, min(w, x1 + ln), min(h, t)], dtype=float)
        elif edge == 1:
            x1 = int(rng.integers(0, max(1, w - ln + 1)))
            box = np.array([x1, max(0, h - t), min(w, x1 + ln), h], dtype=float)
        elif edge == 2:
            y1 = int(rng.integers(0, max(1, h - ln + 1)))
            box = np.array([0, y1, min(w, t), min(h, y1 + ln)], dtype=float)
        else:
            y1 = int(rng.integers(0, max(1, h - ln + 1)))
            box = np.array([max(0, w - t), y1, w, min(h, y1 + ln)], dtype=float)
        return box
    return None


def extract_split(
    model: YOLO, split: str, conf_th: float, device: str,
    rng: np.random.Generator, top_k: int = 20,
) -> dict[str, np.ndarray]:
    """对一个划分执行 YOLO 推理并提取全部候选框特征。

    参数:
        model: 已加载的 YOLO 模型
        split: train / val / test
        conf_th: YOLO 置信度阈值（默认 0.05，尽量收集候选，GA 阶段再筛）
        device: torch 设备字符串
        rng: 随机数生成器（背景块采样）

    返回:
        {"X", "y", "conf", "boxes", "names"} 特征集
    """
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    conf_rows: list[float] = []
    box_rows: list[np.ndarray] = []
    name_rows: list[str] = []

    n_cand = n_missed = n_random = 0
    # 第一遍：逐图推理，收集候选框并标定正负样本。
    # 注意：必须用 io_utils.imread_unicode 自行读图并逐张 predict——
    # model.predict(source=目录) 内部走 cv2.imread，中文路径会静默失败
    #（读图返回 None → 0 个候选框，train/val 的 GT 会全部变成"漏检"）。
    for img_path in sorted(img_dir.glob("*.jpg")):
        img = io_utils.imread_unicode(img_path)
        if img is None:
            print(f"[警告] 读取失败，跳过: {img_path.name}")
            continue
        r = model.predict(img, conf=conf_th, imgsz=640,
                          device=device, verbose=False)[0]
        name = img_path.name
        h, w = img.shape[:2]
        gt = load_gt_boxes(lbl_dir / f"{img_path.stem}.txt", w, h)

        # YOLO 检出框（无检出时为空数组）
        if r.boxes is not None and len(r.boxes) > 0:
            det = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
        else:
            det = np.empty((0, 4))
            confs = np.empty((0,))

        # 候选框：IoU ≥ 0.5 为正样本，其余为负样本（YOLO 误检）。
        # 困难负样本挖掘：正样本无条件保留；负样本按 conf 降序只取
        # 前 top_k 个——训练后 YOLO 置信度标定差（conf 普遍 < 0.02、
        # 单图候选上百），全量保留会使特征规模与 SVM 训练成本失控。
        if len(det) > 0:
            n_kept_neg = 0
            for idx in np.argsort(-confs):          # conf 降序
                box, c = det[idx], float(confs[idx])
                crop = crop_box(img, box)
                if crop is None:
                    continue
                label = 1 if (len(gt) > 0 and max(iou(box, g) for g in gt) >= IOU_POS_TH) else 0
                if label == 0 and n_kept_neg >= top_k:
                    continue
                X_rows.append(svm_core.compute_features(crop))
                y_rows.append(label)
                conf_rows.append(c)
                box_rows.append(box.astype(np.float32))
                name_rows.append(name)
                if label == 0:
                    n_kept_neg += 1
                n_cand += 1

        # YOLO 漏检的真值框 → 追加为正样本（保证 SVM 见过全部缺陷形态）
        for g in gt:
            if len(det) == 0 or max(iou(g, b) for b in det) < IOU_POS_TH:
                crop = crop_box(img, g)
                if crop is None:
                    continue
                X_rows.append(svm_core.compute_features(crop))
                y_rows.append(1)
                conf_rows.append(1.0)   # 真值框直接加入，不参与置信度筛选
                box_rows.append(g.astype(np.float32))
                name_rows.append(name)
                n_missed += 1

    # 第二遍：无缺陷图像负样本增强（M3 修订的关键决策）。
    # 早期版本只在负样本不足时补背景块，导致 SVM 从未见过正常件特有
    # 纹理（端面/倒角/加工纹理），线上测试时把正常件边缘纹理误判为
    # 划痕（如 kos41_Part0 假阳性 7 框，SVM 置信度最高 0.996）。
    # 修订：每个 split 的每张无缺陷图像都采样 3 个随机块作为负样本，
    # 覆盖正常纹理多样性；有缺陷图像的背景块仍参与补充（保持类别平衡）。
    n_pos = sum(y_rows)
    n_neg = len(y_rows) - n_pos
    need = n_pos - n_neg
    for img_path in sorted(img_dir.glob("*.jpg")):
        img = io_utils.imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt_boxes(lbl_dir / f"{img_path.stem}.txt", w, h)
        # 无缺陷图：2 个方形随机块 + 2 个边缘条带（覆盖端面纹理形态）；
        # 有缺陷图仅在负样本不足时补充（4 方块 + 2 条带，受 need 限制）
        if len(gt) == 0:
            patch_boxes = ([random_negative_patch(img, gt, rng) for _ in range(2)]
                           + [random_edge_strip(img, rng) for _ in range(2)])
        elif need > 0:
            patch_boxes = ([random_negative_patch(img, gt, rng) for _ in range(4)]
                           + [random_edge_strip(img, rng) for _ in range(2)])
        else:
            patch_boxes = []
        for patch_box in patch_boxes:
            if patch_box is None:
                continue
            crop = crop_box(img, patch_box)
            if crop is None:
                continue
            X_rows.append(svm_core.compute_features(crop))
            y_rows.append(0)
            conf_rows.append(1.0)   # 背景块不参与置信度筛选
            box_rows.append(patch_box.astype(np.float32))
            name_rows.append(img_path.name)
            n_random += 1
            if len(gt) > 0:
                need -= 1

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int8)
    conf = np.asarray(conf_rows, dtype=np.float32)
    boxes = np.asarray(box_rows, dtype=np.float32)
    names = np.asarray(name_rows, dtype=object)

    print(f"[{split}] 候选 {n_cand} | 正样本 {int(y.sum())} | 负样本 {int((y == 0).sum())} "
          f"| 漏检补充 {n_missed} | 背景补充 {n_random}")
    return {"X": X, "y": y, "conf": conf, "boxes": boxes, "names": names}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="缺陷候选框特征提取与缓存")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_DIR / "yolo_best.pt",
                        help="YOLO 权重路径")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="YOLO 置信度阈值（0.001 最大化召回：训练后模型 conf 普遍 < 0.02）")
    parser.add_argument("--top-k", type=int, default=20,
                        help="每图保留的负样本上限（困难负样本挖掘，按 conf 取前 K）")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="要提取的划分列表")
    parser.add_argument("--device", type=str, default=None,
                        help="设备：0=GPU / cpu=CPU，默认自动检测")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """特征提取入口。"""
    io_utils.force_utf8_stdout()
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"[错误] 未找到 YOLO 权重 {args.weights}，请先运行 train_yolo.py")

    device = detect_device(args.device)
    rng = np.random.default_rng(args.seed)
    model = YOLO(str(args.weights))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("缺陷候选框特征提取（HOG + LBP + 统计）")
    print("=" * 60)
    for split in args.splits:
        feats = extract_split(model, split, args.conf, device, rng, args.top_k)
        out = ARTIFACTS_DIR / f"svm_features_{split}.npz"
        np.savez_compressed(out, **feats)
        print(f"[保存] {out}（{feats['X'].shape[0]} 样本 × {feats['X'].shape[1]} 维）")
    print("特征提取完成")


if __name__ == "__main__":
    main()
