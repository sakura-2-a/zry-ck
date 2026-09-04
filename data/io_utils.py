"""
跨脚本共享的 IO 工具：Windows 中文路径安全的图像读写 + UTF-8 控制台输出。

背景（踩坑记录）：
  - cv2.imread / cv2.imwrite 在 Windows 上对含中文的绝对路径会失败（内部走
    ANSI 接口），np.fromfile + cv2.imdecode / imencode 则完全安全；
  - Windows 控制台默认 GBK 编码，打印 ⚠ 等非 GBK 字符会抛 UnicodeEncodeError，
    统一把 stdout/stderr 重配置为 UTF-8（VSCode 终端默认 UTF-8，显示正常）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def force_utf8_stdout() -> None:
    """把 stdout/stderr 重配置为 UTF-8（Windows 控制台默认 GBK）。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:  # 防御：某些嵌入式环境下 reconfigure 不可用
                pass


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读取图像（兼容含中文的绝对路径）。

    参数:
        path: 图像路径
        flags: cv2 读取标志（如 cv2.IMREAD_GRAYSCALE）

    返回:
        图像数组；文件不存在或无法解码返回 None
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(
    path: str | Path, img: np.ndarray, params: list[int] | None = None
) -> bool:
    """保存图像（兼容含中文的绝对路径）。

    参数:
        path: 输出路径（按扩展名选择编码格式）
        img: 图像数组
        params: cv2.imencode 参数（如 [cv2.IMWRITE_JPEG_QUALITY, 95]）

    返回:
        是否保存成功
    """
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True
