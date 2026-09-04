"""临时诊断脚本：检查 YOLO 在不同置信度阈值下的候选框分布（用完即删）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io_utils
from ultralytics import YOLO

io_utils.force_utf8_stdout()

IMG = r"c:/Users/Administrator/Desktop/轴承缺陷项目/datasets/kolektor_sdd/yolo_format/images/train/kos01_Part5.jpg"
WEIGHTS = r"c:/Users/Administrator/Desktop/轴承缺陷项目/backend/weights/yolo_best.pt"

img = io_utils.imread_unicode(IMG)
print(f"图像尺寸: {img.shape}")
model = YOLO(WEIGHTS)
for conf in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25]:
    r = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    n = len(r.boxes) if r.boxes is not None else 0
    if n:
        confs = r.boxes.conf.cpu().numpy()
        print(f"conf={conf:<6} 候选数={n:3d}  conf范围=[{confs.min():.4f}, {confs.max():.4f}] 平均={confs.mean():.4f}")
    else:
        print(f"conf={conf:<6} 候选数={n:3d}")
