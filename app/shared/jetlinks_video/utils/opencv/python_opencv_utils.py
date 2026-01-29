'''
Author: 13594053100@163.com
Date: 2025-11-10 17:19:24
LastEditTime: 2025-11-11 11:28:49
'''
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import os
from app.shared.jetlinks_video.utils import logger_utils

__all__ = [ 
    "imwrite_jpg",
    "get_video_meta",
    "resize_keep_w",
    "ssim_gray",
    "bgr_ratio_score",
    "dbg_img",
] 

logger = logger_utils.get_logger(__name__)

def imwrite_jpg(path: str, img, quality: int = 90) -> str:
    """安全写 JPG（带质量参数）。"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', ...) failed")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path


def get_video_meta(video_path: str) -> Tuple[float, int]:
    """读取视频 fps 和总帧数（失败时 fps=25.0, total=0）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 25.0, 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
    if fps <= 0:
        fps = 25.0
    return fps, total


def resize_keep_w(frame_bgr, resize_w: int) -> np.ndarray:
    """保持宽度等比缩放。"""
    h, w = frame_bgr.shape[:2]
    small_h = max(1, int(h * (resize_w / max(1, w))))
    return cv2.resize(frame_bgr, (resize_w, small_h), interpolation=cv2.INTER_AREA)


def ssim_gray(g1: np.ndarray, g2: np.ndarray) -> float:
    """简化 SSIM（全图），返回 0~1（越大越相似）。"""
    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = g1.mean()
    mu2 = g2.mean()
    sigma1_sq = g1.var()
    sigma2_sq = g2.var()
    sigma12 = ((g1 - mu1) * (g2 - mu2)).mean()
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2)
    if den <= 0:
        return 0.0
    v = num / den
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def bgr_ratio_score(b1: np.ndarray, b2: np.ndarray) -> float:
    """BGR 三通道绝对差的非零比例（0~1，越大变化越大）。"""
    diff = cv2.absdiff(b1, b2)
    return float(np.count_nonzero(diff)) / float(diff.size)

def dbg_img(tag: str, img: Optional[np.ndarray] = None, path: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
    info = {"tag": tag}
    if path:
        info["path"] = path
        try:
            info["exists"] = os.path.exists(path)
            if info["exists"]:
                # 用 imread 再确认一次真实尺寸（防止你手里 img 不是同一张）
                im = cv2.imread(path)
                info["imread_shape"] = None if im is None else list(im.shape)
        except Exception as e:
            info["imread_err"] = str(e)

    if img is not None:
        info["shape"] = list(img.shape)
        info["dtype"] = str(img.dtype)
        info["minmax"] = [float(np.min(img)), float(np.max(img))] if img.size else None
        info["contiguous"] = bool(img.flags["C_CONTIGUOUS"])

    if extra:
        info.update(extra)

    logger.info("[DBG_IMG] %s", info)


