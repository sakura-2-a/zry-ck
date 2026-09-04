"""
M3/M4 共享核心模块：SVM 特征计算、特征集加载、训练与评估。

被 train_svm.py（M3）与 ga_optimize.py（M4）共同引用。
两个脚本通过 `sys.path.insert(0, 脚本目录)` 后 `import svm_core` 使用，
因此本模块必须与它们放在同一目录（backend/scripts/）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# 特征配置（M3 设计决策）
#   候选框统一缩放到 64×64 后提取三类特征：
#     HOG ：梯度方向直方图，描述局部纹理方向分布（1764 维）
#     LBP ：旋转不变均匀模式（LBP^riu2），描述局部纹理模式（59 维）
#     统计：均值 / 标准差 / 对比度（3 维）
#   合计 1826 维。HOG/LBP 参数在 GA 阶段保持固定（特征缓存一次），
#   GA 只优化 SVM 超参数与两级阈值，保证每代评估在秒级完成。
# ---------------------------------------------------------------------------
PATCH_SIZE = 64
HOG_WIN = (64, 64)
HOG_BLOCK = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL = (8, 8)
HOG_NBINS = 9
HOG_DIM = 1764       # ((64-16)/8+1)^2 × 4 cell × 9 bins = 1764
LBP_DIM = 59         # 均匀模式 58 类 + 非均匀 1 类
STAT_DIM = 3
FEATURE_DIM = HOG_DIM + LBP_DIM + STAT_DIM

FEATURE_CONFIG = {
    "patch_size": PATCH_SIZE,
    "hog": {"win": HOG_WIN, "block": HOG_BLOCK, "stride": HOG_BLOCK_STRIDE,
            "cell": HOG_CELL, "nbins": HOG_NBINS, "dim": HOG_DIM},
    "lbp": {"radius": 1, "points": 8, "dim": LBP_DIM},
    "stat": {"dim": STAT_DIM},
    "total_dim": FEATURE_DIM,
}


def _uniform_lbp_table(n_points: int = 8) -> np.ndarray:
    """生成 256 个 LBP 编码到 59 维均匀模式直方图索引的映射表。

    原理：8 位 LBP 编码按循环位序统计 0/1 跳变次数，跳变 ≤ 2 的称为
    均匀模式（按旋转不变代表编码编号 0~57，共 58 类），其余非均匀
    模式统一归入第 58 维（最后一维）。

    参数:
        n_points: 邻域采样点数（8）

    返回:
        长度为 256 的映射表，table[code] = 直方图索引
    """
    mask = (1 << n_points) - 1
    table = np.full(256, 58, dtype=np.uint8)  # 58 = 均匀模式数（非均匀全部归入此维）
    reps: dict[int, int] = {}
    for code in range(256):
        bits = [(code >> p) & 1 for p in range(n_points)]
        transitions = sum(bits[p] != bits[(p + 1) % n_points] for p in range(n_points))
        if transitions > 2:
            continue  # 非均匀模式
        # 旋转不变：循环移位取最小编码作为该模式的代表
        rep = min(((code >> k) | ((code << (n_points - k)) & mask)) for k in range(n_points))
        if rep not in reps:
            reps[rep] = len(reps)
        table[code] = reps[rep]
    return table


def _compute_lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """计算旋转不变均匀模式 LBP（LBP^riu2）的 59 维归一化直方图。

    参数:
        gray: 灰度图（实际输入为 64×64）

    返回:
        59 维 float32 直方图（元素和恒为 1）
    """
    table = _uniform_lbp_table()
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge")  # 半径 1 的 8 邻域，边界复制填充
    codes = np.zeros((h, w), dtype=np.uint8)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    for p, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
        codes |= ((neighbor >= gray).astype(np.uint8) << p)  # 邻域 ≥ 中心记为 1
    hist = np.bincount(table[codes].ravel(), minlength=59).astype(np.float32)
    return hist / (hist.sum() + 1e-9)


def _compute_hog(gray: np.ndarray) -> np.ndarray:
    """手写 HOG（梯度方向直方图）特征：64×64 灰度图 → 1764 维。

    与 cv2.HOGDescriptor(win=(64,64), block=(16,16), stride=(8,8),
    cell=(8,8), nbins=9) 的网格几何完全一致：
        7×7 个滑动块 × 4 个 cell × 9 个方向 = 1764 维。
    手写原因：opencv-python 4.10 起移除了 HOGDescriptor 的 Python 绑定。

    计算步骤：
      1) Sobel 算子求梯度幅值与无符号方向（[0°, 180°)）
      2) 每个 8×8 cell 内按最近邻方向 bin 投票，得 9 维直方图
      3) 相邻 2×2 cell 组成 16×16 块（36 维），L2-Hys 归一化
         （L2 归一化 → 截断 0.2 → 再归一化，抑制纹理变化）
      4) 按 8×8 步长滑动拼接全部块

    与 OpenCV 实现的差异：方向 bin 采用最近邻投票（无插值），
    因 SVM/GA 全程使用同一实现，特征空间自洽即可。

    参数:
        gray: 64×64 灰度图（uint8 或 float32）

    返回:
        (1764,) 的 float32 特征向量
    """
    h, w = gray.shape
    # 1) 梯度幅值与无符号方向（[0, 180)）
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0).astype(np.float32)

    # 2) 每个 cell 的 9 方向幅值直方图
    n_cx, n_cy = w // HOG_CELL[0], h // HOG_CELL[1]
    cell_hists = np.zeros((n_cy, n_cx, HOG_NBINS), dtype=np.float32)
    row_idx = np.arange(h)[:, None] // HOG_CELL[1]   # (h,1) 像素所在 cell 行号
    col_idx = np.arange(w)[None, :] // HOG_CELL[0]   # (1,w) 像素所在 cell 列号
    bin_idx = (ang / (180.0 / HOG_NBINS)).astype(np.int32) % HOG_NBINS
    for b in range(HOG_NBINS):
        votes = np.where(bin_idx == b, mag, 0.0)
        np.add.at(cell_hists[:, :, b], (row_idx, col_idx), votes)

    # 3) 2×2 cell 组成块 → 36 维向量 → L2-Hys 归一化 → 拼接
    block_cells = HOG_BLOCK[0] // HOG_CELL[0]   # 2
    feats: list[np.ndarray] = []
    eps = 1e-6
    for by in range(n_cy - block_cells + 1):
        for bx in range(n_cx - block_cells + 1):
            vec = cell_hists[by:by + block_cells, bx:bx + block_cells, :].ravel()
            vec = vec / (np.sqrt(float((vec ** 2).sum()) + eps))
            vec = np.minimum(vec, 0.2)                     # Hys 截断
            vec = vec / (np.sqrt(float((vec ** 2).sum()) + eps))
            feats.append(vec.astype(np.float32))
    return np.concatenate(feats)


def compute_features(crop: np.ndarray) -> np.ndarray:
    """从缺陷候选框裁剪图计算 1826 维特征向量（HOG + LBP + 统计）。

    参数:
        crop: 候选框裁剪图（BGR 或灰度，任意尺寸，内部统一处理）

    返回:
        (1826,) 的 float32 特征向量
    """
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    patch = cv2.resize(gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

    # 1. HOG 梯度方向直方图（描述局部纹理方向分布）
    hog_feat = _compute_hog(patch)

    # 2. LBP 直方图（描述局部纹理模式，对光照变化不敏感）
    lbp_feat = _compute_lbp_histogram(patch)

    # 3. 统计特征（整体灰度分布）
    mean = float(patch.mean())
    std = float(patch.std())
    contrast = std / (mean + 1e-6)   # 对比度：相对灰度波动程度
    stat_feat = np.array([mean, std, contrast], dtype=np.float32)

    return np.concatenate([hog_feat, lbp_feat, stat_feat]).astype(np.float32)


def load_feature_set(npz_path: Path) -> dict[str, np.ndarray]:
    """加载 extract_features.py 缓存的特征集。

    参数:
        npz_path: 特征缓存文件（*.npz）

    返回:
        键为 X / y / conf / boxes / names 的字典
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"特征缓存不存在: {npz_path}\n"
            "请先运行: python backend/scripts/extract_features.py"
        )
    data = np.load(npz_path, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"],
        "conf": data["conf"],
        "boxes": data["boxes"],
        "names": data["names"],
    }


def train_svm(
    X: np.ndarray, y: np.ndarray, C: float = 1.0,
    gamma: float | str = "scale", kernel: str = "rbf",
) -> tuple[StandardScaler, SVC]:
    """在给定超参数下训练 SVM 二级判别器。

    参数:
        X: 特征矩阵 (n, 1826)
        y: 样本标签（1=真缺陷，0=误检/背景）
        C: 惩罚系数（越大越倾向拟合训练数据，过大会过拟合）
        gamma: RBF 核宽度（'scale' 表示 sklearn 自动按特征数计算）
        kernel: 核函数（rbf / linear / poly）

    返回:
        (已拟合的 StandardScaler, 已拟合的 SVC)
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # class_weight=balanced：自动按类别比例加权，缓解负样本过多导致的偏向
    svm = SVC(C=C, gamma=gamma, kernel=kernel, probability=True,
              class_weight="balanced", random_state=42)
    svm.fit(Xs, y)
    return scaler, svm


def evaluate_samples(
    svm: SVC, scaler: StandardScaler, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """评估 SVM 分类器本身的性能（准确率 / P / R / F1 / AUC）。

    参数:
        svm, scaler: 已训练模型
        X, y: 评估集特征与标签

    返回:
        指标字典
    """
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    Xs = scaler.transform(X)
    pred = svm.predict(Xs)
    prob = svm.predict_proba(Xs)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y, prob)), 4),
    }


def evaluate_pipeline(
    svm: SVC, scaler: StandardScaler, X: np.ndarray, y: np.ndarray,
    conf: np.ndarray, conf_th: float, prob_th: float,
) -> dict[str, float]:
    """评估「YOLO 置信度阈值 + SVM 判别」两级流水线的样本级检测指标。

    判定规则（与后端推理流水线完全一致）：
      候选框保留条件 = YOLO 置信度 ≥ conf_th 且 SVM 概率 ≥ prob_th
    样本标签 y 在特征提取阶段由「候选框与真值框 IoU ≥ 0.5」定义，
    因此 TP/FP/FN 可直接在样本上统计（近似：一个真值框至多对应一个 TP）。

    参数:
        svm, scaler: 已训练模型
        X, y, conf: 评估集特征 / 标签 / YOLO 置信度
        conf_th: YOLO 置信度阈值
        prob_th: SVM 概率阈值

    返回:
        {"precision", "recall", "f1", "tp", "fp", "fn"} 字典
    """
    keep_conf = conf >= conf_th
    Xs = scaler.transform(X[keep_conf])
    prob = svm.predict_proba(Xs)[:, 1]

    detected = keep_conf.copy()
    detected[keep_conf] = prob >= prob_th

    tp = int((detected & (y == 1)).sum())
    fp = int((detected & (y == 0)).sum())
    fn = int(((~detected) & (y == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": round(float(precision), 4), "recall": round(float(recall), 4),
            "f1": round(float(f1), 4), "tp": tp, "fp": fp, "fn": fn}
