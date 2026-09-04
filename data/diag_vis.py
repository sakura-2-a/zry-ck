"""临时诊断脚本：对指定图像运行两级流水线并画框可视化（用完即删）。"""
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
from app.services.detector import DetectionPipeline  # noqa: E402

import io_utils  # noqa: E402
io_utils.force_utf8_stdout()

IMG = r"c:/Users/Administrator/Desktop/轴承缺陷项目/datasets/kolektor_sdd/yolo_format/images/test/kos41_Part0.jpg"
OUT = BACKEND_DIR / "artifacts" / "diag_part0.jpg"

p = DetectionPipeline()
p.load()
img = io_utils.imread_unicode(IMG)
result = p.detect(Path(IMG).read_bytes(), Path(IMG).name)
print(f"检出 {result['num_defects']} 框:")
vis = img.copy()
if vis.ndim == 2:   # 单通道灰度 → BGR 三通道（便于画彩色框）
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
for b in result["boxes"]:
    print(f"  ({b['x1']:.0f},{b['y1']:.0f})-({b['x2']:.0f},{b['y2']:.0f}) conf={b['confidence']:.3f}")
    cv2.rectangle(vis, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])),
                  (0, 0, 255), 2)
    cv2.putText(vis, f"{b['confidence']:.2f}", (int(b["x1"]), max(15, int(b["y1"]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
scale = 600 / vis.shape[1]
vis = cv2.resize(vis, (600, int(vis.shape[0] * scale)))
io_utils.imwrite_unicode(OUT, vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"已保存: {OUT}")
