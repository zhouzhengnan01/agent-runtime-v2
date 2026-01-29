from __future__ import annotations
"""
根据 VLM 返回的 objects 数组，在证据帧上绘制目标检测框，并导出到静态目录。

✅ 最新规范：
- VLM 输出 box 为归一化坐标 [0,1]，并且保留 4 位小数（提示词侧约束）
- 画框侧严格校验：任一坐标点不在 [0,1] -> 直接丢弃该框，并打印 warning
- 同时校验 bx1<bx2 && by1<by2，否则丢弃并打印 warning

✅ 颜色策略（你确认的）：
- 同一个 seg# 内：按“首次出现 label 的顺序”分配颜色（绿、紫、青…）
- 后续帧遇到同 label：复用同一颜色（跨帧一致）

✅ 视觉优化：
- label 文本背景框使用同色“半透明”填充（addWeighted）
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import uuid

import cv2

from app.shared.jetlinks_video.configs.vlm_config import VlmConfig
from app.shared.jetlinks_video.utils import logger_utils
from app.shared.jetlinks_video.utils.opencv.python_opencv_utils import dbg_img

logger = logger_utils.get_logger(__name__)


def _is_http_url(p: str) -> bool:
    return isinstance(p, str) and p.lower().startswith(("http://", "https://"))


def _uri_to_local_path(uri: str) -> Optional[str]:
    if not uri:
        return None
    if uri.lower().startswith("file://"):
        return uri[len("file://"):]
    return uri


def _ensure_box_dir(vlm_config: Optional[VlmConfig]) -> Tuple[str, str]:
    if vlm_config is not None and getattr(vlm_config, "vlm_static_evidence_images_box_dir", None):
        box_dir = vlm_config.vlm_static_evidence_images_box_dir
    else:
        box_dir = os.path.join(os.getcwd(), "storage", "evidence_images_box")

    if vlm_config is not None and getattr(vlm_config, "vlm_static_evidence_images_box_url_prefix", None):
        url_prefix = vlm_config.vlm_static_evidence_images_box_url_prefix
    else:
        url_prefix = "/storage/evidence_images_box"

    url_prefix = url_prefix.rstrip("/")
    os.makedirs(box_dir, exist_ok=True)
    return box_dir, url_prefix


def _collect_objects_for_image(
    events: List[Dict[str, Any]],
    image_index: int,
    total_images: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if total_images <= 1:
        for ev in events:
            for obj in ev.get("objects") or []:
                if isinstance(obj, dict):
                    out.append(obj)
        return out

    for ev in events:
        objs = ev.get("objects") or []
        if not isinstance(objs, list):
            continue
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            idx = obj.get("image_index", obj.get("frame_index"))
            if idx is None:
                target = 0
            else:
                try:
                    target = int(idx)
                except Exception:
                    continue
            if target == image_index:
                out.append(obj)
    return out


def _pick_text_style(img_w: int) -> Tuple[float, int]:
    """根据分辨率自动放大字体/描边。"""
    if img_w >= 3000:
        return 1.10, 2
    if img_w >= 2400:
        return 1.00, 2
    if img_w >= 1800:
        return 0.90, 2
    if img_w >= 1200:
        return 0.80, 2
    return 0.70, 2


_LABEL_COLOR_SEQUENCE: List[Tuple[int, int, int]] = [
    (0, 255, 0),      # 1 绿
    (255, 0, 255),    # 2 紫
    (255, 255, 0),    # 3 青
    (0, 255, 255),    # 4 黄
    (255, 0, 0),      # 5 蓝
    (0, 165, 255),    # 6 橙
    (0, 0, 255),      # 7 红
    (128, 128, 128),  # 8 灰
    (255, 127, 80),   # 9 珊瑚
    (50, 205, 50),    # 10 亮绿
    (138, 43, 226),   # 11 蓝紫
    (64, 224, 208),   # 12 绿松石
]


def _get_color_for_label_in_order(label: str, label2color: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    key = (label or "").strip()
    if not key:
        key = "_unknown_"

    if key in label2color:
        return label2color[key]

    idx = len(label2color)
    color = _LABEL_COLOR_SEQUENCE[idx % len(_LABEL_COLOR_SEQUENCE)]
    label2color[key] = color
    return color


def _draw_transparent_rect(
    img: Any,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    alpha: float = 0.45,
) -> None:
    """
    在 img 上画一个“半透明填充矩形”：
    - alpha 越大越不透明（0~1）
    - 只对 ROI 做 overlay，性能更好
    """
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 <= x1 or y2 <= y1:
        return

    h, w = img.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return

    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), color_bgr, thickness=-1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def export_evidence_images_with_boxes(
    evidence_images: List[str],
    events: List[Dict[str, Any]],
    seg_idx: int,
    vlm_config: Optional[VlmConfig] = None,
) -> List[str]:
    if not evidence_images or not events:
        return []

    box_dir, url_prefix = _ensure_box_dir(vlm_config)
    urls: List[str] = []

    total = len(evidence_images)

    # ✅ seg 内共享 label->color 映射（跨帧一致）
    seg_label2color: Dict[str, Tuple[int, int, int]] = {}

    for i, uri in enumerate(evidence_images):
        if _is_http_url(uri):
            logger.debug("[BBox] seg#%d 第 %d 帧为 http URL(%s)，跳过画框。", seg_idx, i, uri)
            continue

        local_path = _uri_to_local_path(uri)
        if not local_path or not os.path.exists(local_path):
            logger.warning("[BBox] seg#%d 第 %d 帧本地文件不存在: %s", seg_idx, i, local_path)
            continue

        img = cv2.imread(local_path)
        if img is None:
            logger.warning("[BBox] seg#%d 第 %d 帧无法读取图像: %s", seg_idx, i, local_path)
            continue

        h, w = img.shape[:2]
        objs = _collect_objects_for_image(events, i, total_images=total)

        dbg_img(
            "bbox:imread",
            img=img,
            path=local_path,
            extra={"seg_idx": seg_idx, "image_index": i, "total": total},
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = _pick_text_style(w)
        rect_thickness = 3 if w >= 1800 else 2
        alpha_bg = 0.45  # ✅ 半透明背景强度（你想更清晰就调到 0.55~0.65）

        kept = 0
        dropped = 0

        for obj in objs:
            box = obj.get("box") or obj.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                dropped += 1
                logger.warning(
                    "[BBox] seg#%d img#%d drop_box(invalid_format): label=%s conf=%s box=%s path=%s",
                    seg_idx, i, str(obj.get("label", "")), str(obj.get("confidence", "")), str(box), local_path
                )
                continue

            try:
                bx1 = float(box[0])
                by1 = float(box[1])
                bx2 = float(box[2])
                by2 = float(box[3])
            except Exception:
                dropped += 1
                logger.warning(
                    "[BBox] seg#%d img#%d drop_box(non_numeric): label=%s conf=%s box=%s path=%s",
                    seg_idx, i, str(obj.get("label", "")), str(obj.get("confidence", "")), str(box), local_path
                )
                continue

            # 任一坐标点不在 [0,1] -> 丢弃
            if not (0.0 <= bx1 <= 1.0 and 0.0 <= by1 <= 1.0 and 0.0 <= bx2 <= 1.0 and 0.0 <= by2 <= 1.0):
                dropped += 1
                logger.warning(
                    "[BBox] seg#%d img#%d drop_box(out_of_range_0_1): label=%s conf=%s box=%s path=%s",
                    seg_idx, i, str(obj.get("label", "")), str(obj.get("confidence", "")), str(box), local_path
                )
                continue

            if not (bx1 < bx2 and by1 < by2):
                dropped += 1
                logger.warning(
                    "[BBox] seg#%d img#%d drop_box(invalid_order): label=%s conf=%s box=%s path=%s",
                    seg_idx, i, str(obj.get("label", "")), str(obj.get("confidence", "")), str(box), local_path
                )
                continue

            # 0~1 -> 像素
            x1 = int(round(bx1 * w))
            x2 = int(round(bx2 * w))
            y1 = int(round(by1 * h))
            y2 = int(round(by2 * h))

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                dropped += 1
                logger.warning(
                    "[BBox] seg#%d img#%d drop_box(after_clamp_invalid): label=%s conf=%s box=%s px=[%d,%d,%d,%d] wh=(%d,%d) path=%s",
                    seg_idx, i, str(obj.get("label", "")), str(obj.get("confidence", "")), str(box),
                    x1, y1, x2, y2, w, h, local_path
                )
                continue

            label = str(obj.get("label", "") or "").strip()
            conf = obj.get("confidence", None)

            # ✅ 用 seg 级映射：跨帧一致；首个 label=绿，第二个=紫...
            color = _get_color_for_label_in_order(label, seg_label2color)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, rect_thickness)

            if isinstance(conf, (int, float)):
                label_text = f"{label} {conf:.2f}" if label else f"{conf:.2f}"
            else:
                label_text = label

            if label_text:
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                tx = x1
                ty = y1 - 6
                if ty - th - 8 < 0:
                    ty = y1 + th + 8  # 放到框内/下方，避免出界

                # 背景矩形坐标（加 padding）
                pad_x = 4
                pad_y = 3
                x_bg1 = tx
                y_bg1 = ty - th - pad_y * 2
                x_bg2 = tx + tw + pad_x * 2
                y_bg2 = ty + pad_y

                # clamp 背景到图内
                x_bg1 = max(0, min(x_bg1, w - 1))
                y_bg1 = max(0, min(y_bg1, h - 1))
                x_bg2 = max(0, min(x_bg2, w))
                y_bg2 = max(0, min(y_bg2, h))

                # ✅ 半透明同色背景
                _draw_transparent_rect(img, (x_bg1, y_bg1), (x_bg2, y_bg2), color, alpha=alpha_bg)

                # 文本（黑字）
                cv2.putText(
                    img,
                    label_text,
                    (x_bg1 + pad_x, y_bg2 - pad_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    lineType=cv2.LINE_AA,
                )

            kept += 1

        logger.info(
            "[BBox] seg#%d img#%d draw_summary kept=%d dropped=%d unique_labels(seg_scope)=%d path=%s",
            seg_idx, i, kept, dropped, len(seg_label2color), local_path
        )

        # 保存带框图片
        ext = os.path.splitext(local_path)[1] or ".jpg"
        uid = uuid.uuid4().hex
        fname = f"seg{seg_idx:04d}_boxed_{i:02d}_{uid}{ext}"
        dst = os.path.join(box_dir, fname)

        dbg_img(
            "bbox:before_write",
            img=img,
            path=dst,
            extra={"seg_idx": seg_idx, "image_index": i, "dst": dst},
        )

        try:
            cv2.imwrite(dst, img)
            url = f"{url_prefix}/{fname}"
            urls.append(url)
            dbg_img(
                "bbox:after_write",
                path=dst,
                extra={"seg_idx": seg_idx, "image_index": i, "url": url},
            )
        except Exception as e:
            logger.warning("[BBox] seg#%d 第 %d 帧写入带框图片失败: %s", seg_idx, i, e)

    return urls
