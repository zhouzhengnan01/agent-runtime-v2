# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import math
from typing import Any, List, Optional, Tuple
from queue import Queue, Empty

from app.shared.jetlinks_video.utils import logger_utils
from app.shared.jetlinks_video.all_enum import MODEL
from app.shared.jetlinks_video.configs.cv_config import CvConfig, CvPoseConfig, CvRoiConfig, build_cv_config
from app.shared.jetlinks_video.configs.rtsp_batch_config import RTSPBatchConfig
from app.shared.jetlinks_video.configs.vlm_config import VlmConfig
from app.shared.jetlinks_video.workers.worker_b_vlm import (
    _ctrl_stop_requested,
    _emit_done,
    _export_evidence_list_to_static,
    _q_put_with_retry,
)

logger = logger_utils.get_logger(__name__)

_TASK_LABELS = {
    "fall": "跌倒",
    "smoke": "抽烟",
    "fight": "吵架",
}


def _tensor_to_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    try:
        value = value.cpu()
    except Exception:
        pass
    try:
        return value.tolist()
    except Exception:
        return None


def _get_image_shape(res: Any) -> Tuple[int, int]:
    orig_shape = getattr(res, "orig_shape", None) or getattr(res, "orig_img", None)
    if isinstance(orig_shape, (list, tuple)) and len(orig_shape) >= 2:
        return int(orig_shape[1]), int(orig_shape[0])
    if hasattr(orig_shape, "shape"):
        return int(orig_shape.shape[1]), int(orig_shape.shape[0])
    return 0, 0


def _get_kpt_point(kpts: List[List[float]], confs: List[float], idx: int, min_conf: float) -> Optional[Tuple[float, float]]:
    if idx >= len(kpts) or idx >= len(confs):
        return None
    if confs[idx] < min_conf:
        return None
    x, y = kpts[idx]
    return float(x), float(y)


def _midpoint(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not a or not b:
        return None
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _extract_pose_people(res: Any, min_kpt_conf: float) -> List[dict]:
    boxes = getattr(res, "boxes", None)
    keypoints = getattr(res, "keypoints", None)
    if boxes is None or keypoints is None:
        return []

    w, h = _get_image_shape(res)
    if w <= 0 or h <= 0:
        return []

    xyxy = _tensor_to_list(getattr(boxes, "xyxy", None))
    confs = _tensor_to_list(getattr(boxes, "conf", None)) or []
    if not xyxy:
        return []

    if hasattr(keypoints, "xyn"):
        kpts_xy = _tensor_to_list(keypoints.xyn)
        normalized = True
    else:
        kpts_xy = _tensor_to_list(getattr(keypoints, "xy", None))
        normalized = False

    if not kpts_xy:
        return []

    if not normalized:
        kpts_xy = [
            [[float(x) / w, float(y) / h] for x, y in person]
            for person in kpts_xy
        ]

    kpts_conf = _tensor_to_list(getattr(keypoints, "conf", None))
    if not kpts_conf:
        kpts_conf = [[1.0 for _ in person] for person in kpts_xy]

    n = min(len(xyxy), len(kpts_xy), len(kpts_conf))
    people: List[dict] = []
    for i in range(n):
        box = xyxy[i]
        if not box or len(box) != 4:
            continue
        box_norm = _normalize_box_xyxy(
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
            w,
            h,
        )
        if not box_norm:
            continue
        x1, y1, x2, y2 = box_norm
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        conf = float(confs[i]) if i < len(confs) else 0.0
        people.append(
            {
                "box": box_norm,
                "conf": max(0.0, min(1.0, conf)),
                "kpts": kpts_xy[i],
                "kpt_conf": kpts_conf[i],
                "center": center,
                "size": (bw, bh),
            }
        )
    return people


def _detect_fall(person: dict, pose_cfg: CvPoseConfig) -> Tuple[bool, float]:
    kpts = person["kpts"]
    confs = person["kpt_conf"]
    x1, y1, x2, y2 = person["box"]
    bw, bh = person["size"]
    ratio = bw / (bh + 1e-6)

    l_shoulder = _get_kpt_point(kpts, confs, 5, pose_cfg.min_kpt_conf)
    r_shoulder = _get_kpt_point(kpts, confs, 6, pose_cfg.min_kpt_conf)
    l_hip = _get_kpt_point(kpts, confs, 11, pose_cfg.min_kpt_conf)
    r_hip = _get_kpt_point(kpts, confs, 12, pose_cfg.min_kpt_conf)
    mid_shoulder = _midpoint(l_shoulder, r_shoulder)
    mid_hip = _midpoint(l_hip, r_hip)

    angle_ratio = 0.0
    if mid_shoulder and mid_hip:
        dx = abs(mid_shoulder[0] - mid_hip[0])
        dy = abs(mid_shoulder[1] - mid_hip[1])
        angle_ratio = dx / (dy + 1e-6)

    l_ankle = _get_kpt_point(kpts, confs, 15, pose_cfg.min_kpt_conf)
    r_ankle = _get_kpt_point(kpts, confs, 16, pose_cfg.min_kpt_conf)
    ankle_y = max([p[1] for p in (l_ankle, r_ankle) if p], default=y2)
    hip_y = mid_hip[1] if mid_hip else y2

    is_horizontal = ratio > pose_cfg.fall_ratio or angle_ratio > pose_cfg.fall_angle_ratio
    is_low = ankle_y > pose_cfg.fall_low_y or hip_y > (pose_cfg.fall_low_y - 0.05)
    if not (is_horizontal and is_low):
        return False, 0.0

    kp_indices = [5, 6, 11, 12, 15, 16]
    kp_scores = [confs[i] for i in kp_indices if i < len(confs)]
    kp_score = sum(kp_scores) / len(kp_scores) if kp_scores else 0.0
    score = max(person["conf"], kp_score)
    return True, min(1.0, score)


def _detect_smoke(person: dict, pose_cfg: CvPoseConfig) -> Tuple[bool, float]:
    kpts = person["kpts"]
    confs = person["kpt_conf"]
    bw, bh = person["size"]

    nose = _get_kpt_point(kpts, confs, 0, pose_cfg.min_kpt_conf)
    l_eye = _get_kpt_point(kpts, confs, 1, pose_cfg.min_kpt_conf)
    r_eye = _get_kpt_point(kpts, confs, 2, pose_cfg.min_kpt_conf)
    face = nose or _midpoint(l_eye, r_eye)
    if not face:
        return False, 0.0

    l_wrist = _get_kpt_point(kpts, confs, 9, pose_cfg.min_kpt_conf)
    r_wrist = _get_kpt_point(kpts, confs, 10, pose_cfg.min_kpt_conf)
    wrists = [p for p in (l_wrist, r_wrist) if p]
    if not wrists:
        return False, 0.0

    l_shoulder = _get_kpt_point(kpts, confs, 5, pose_cfg.min_kpt_conf)
    r_shoulder = _get_kpt_point(kpts, confs, 6, pose_cfg.min_kpt_conf)
    l_hip = _get_kpt_point(kpts, confs, 11, pose_cfg.min_kpt_conf)
    r_hip = _get_kpt_point(kpts, confs, 12, pose_cfg.min_kpt_conf)
    mid_shoulder = _midpoint(l_shoulder, r_shoulder)
    mid_hip = _midpoint(l_hip, r_hip)
    torso = _distance(mid_shoulder, mid_hip) if (mid_shoulder and mid_hip) else max(bh, 1e-6)

    wrist_dist = min(_distance(face, w) for w in wrists)
    if wrist_dist > pose_cfg.smoke_dist_ratio * torso:
        return False, 0.0

    if mid_shoulder:
        wrist_y = min(w[1] for w in wrists)
        if wrist_y > mid_shoulder[1] + pose_cfg.smoke_wrist_y_margin:
            return False, 0.0

    face_score = confs[0] if len(confs) > 0 else 0.0
    wrist_scores = [confs[i] for i in (9, 10) if i < len(confs)]
    wrist_score = max(wrist_scores) if wrist_scores else 0.0
    score = max(person["conf"], (face_score + wrist_score) * 0.5)
    return True, min(1.0, score)


def _hand_close_to_target(source: dict, target: dict, min_kpt_conf: float, hand_ratio: float) -> bool:
    s_kpts = source["kpts"]
    s_confs = source["kpt_conf"]
    t_kpts = target["kpts"]
    t_confs = target["kpt_conf"]

    l_wrist = _get_kpt_point(s_kpts, s_confs, 9, min_kpt_conf)
    r_wrist = _get_kpt_point(s_kpts, s_confs, 10, min_kpt_conf)
    wrists = [p for p in (l_wrist, r_wrist) if p]
    if not wrists:
        return False

    t_face = _get_kpt_point(t_kpts, t_confs, 0, min_kpt_conf)
    t_l_shoulder = _get_kpt_point(t_kpts, t_confs, 5, min_kpt_conf)
    t_r_shoulder = _get_kpt_point(t_kpts, t_confs, 6, min_kpt_conf)
    t_mid_shoulder = _midpoint(t_l_shoulder, t_r_shoulder)
    t_l_hip = _get_kpt_point(t_kpts, t_confs, 11, min_kpt_conf)
    t_r_hip = _get_kpt_point(t_kpts, t_confs, 12, min_kpt_conf)
    t_mid_hip = _midpoint(t_l_hip, t_r_hip)
    targets = [p for p in (t_face, t_mid_shoulder, t_mid_hip) if p]
    if not targets:
        return False

    torso = _distance(t_mid_shoulder, t_mid_hip) if (t_mid_shoulder and t_mid_hip) else max(target["size"][1], 1e-6)
    min_dist = min(_distance(w, t) for w in wrists for t in targets)
    return min_dist < hand_ratio * torso


def _detect_fight(people: List[dict], pose_cfg: CvPoseConfig) -> List[float]:
    scores = [0.0 for _ in people]
    if len(people) < 2:
        return scores

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1 = people[i]
            p2 = people[j]
            c1 = p1["center"]
            c2 = p2["center"]
            diag1 = math.hypot(p1["size"][0], p1["size"][1])
            diag2 = math.hypot(p2["size"][0], p2["size"][1])
            avg_diag = max((diag1 + diag2) * 0.5, 1e-6)
            center_dist = _distance(c1, c2)
            if center_dist > pose_cfg.fight_center_ratio * avg_diag:
                continue

            hit = _hand_close_to_target(
                p1, p2, pose_cfg.min_kpt_conf, pose_cfg.fight_hand_ratio
            ) or _hand_close_to_target(
                p2, p1, pose_cfg.min_kpt_conf, pose_cfg.fight_hand_ratio
            )
            if not hit:
                continue

            score = max(p1["conf"], p2["conf"])
            if score < pose_cfg.fight_min_score:
                continue
            scores[i] = max(scores[i], score)
            scores[j] = max(scores[j], score)

    return scores


def _normalize_roi(
    roi_cfg: Optional[CvRoiConfig],
    w: int,
    h: int,
) -> Optional[Tuple[float, float, float, float]]:
    if not roi_cfg:
        return None
    rect = roi_cfg.rect
    if not rect:
        return None
    try:
        x1, y1, x2, y2 = rect
    except Exception:
        return None
    if not roi_cfg.normalized:
        if w <= 0 or h <= 0:
            return None
        x1 /= w
        x2 /= w
        y1 /= h
        y2 /= h

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    pad = max(0.0, float(roi_cfg.padding or 0.0))
    if pad:
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _roi_hit(
    box: List[float],
    roi: Tuple[float, float, float, float],
    mode: str,
) -> bool:
    try:
        bx1, by1, bx2, by2 = box
    except Exception:
        return False
    rx1, ry1, rx2, ry2 = roi
    if mode in {"inside", "contain", "containment"}:
        return bx1 >= rx1 and by1 >= ry1 and bx2 <= rx2 and by2 <= ry2
    if mode in {"intersect", "overlap", "iou"}:
        return not (bx2 <= rx1 or bx1 >= rx2 or by2 <= ry1 or by1 >= ry2)
    # default: center
    cx = (bx1 + bx2) * 0.5
    cy = (by1 + by2) * 0.5
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _select_roi_config(cv_config: CvConfig, source_id: Optional[Any]) -> Optional[CvRoiConfig]:
    if source_id is not None:
        key = str(source_id)
        roi = cv_config.roi_by_id.get(key)
        if roi:
            return roi
    return cv_config.roi


def _coerce_pose_objects(
    results: List[Any],
    task: str,
    max_boxes: int,
    pose_cfg: CvPoseConfig,
    roi_cfg: Optional[CvRoiConfig],
) -> List[dict]:
    objects: List[dict] = []
    label = _TASK_LABELS.get(task, task)

    for image_index, res in enumerate(results):
        people = _extract_pose_people(res, pose_cfg.min_kpt_conf)
        if not people:
            continue
        w, h = _get_image_shape(res)
        roi_norm = _normalize_roi(roi_cfg, w, h) if roi_cfg else None
        roi_mode = (roi_cfg.mode if roi_cfg else "center").strip().lower()

        if task == "fight":
            scores = _detect_fight(people, pose_cfg)
            for idx, score in enumerate(scores):
                if score <= 0:
                    continue
                if roi_norm and not _roi_hit(people[idx]["box"], roi_norm, roi_mode):
                    continue
                objects.append(
                    {
                        "label": label,
                        "confidence": round(min(1.0, score), 2),
                        "box": people[idx]["box"],
                        "image_index": image_index,
                    }
                )
            continue

        for person in people:
            if task == "fall":
                hit, score = _detect_fall(person, pose_cfg)
            elif task == "smoke":
                hit, score = _detect_smoke(person, pose_cfg)
            else:
                hit, score = False, 0.0

            if not hit:
                continue
            if roi_norm and not _roi_hit(person["box"], roi_norm, roi_mode):
                continue
            objects.append(
                {
                    "label": label,
                    "confidence": round(min(1.0, score), 2),
                    "box": person["box"],
                    "image_index": image_index,
                }
            )

    objects.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    if max_boxes > 0 and len(objects) > max_boxes:
        objects = objects[:max_boxes]
    return objects


def _normalize_box_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> Optional[List[float]]:
    if w <= 0 or h <= 0:
        return None
    x1n = max(0.0, min(1.0, x1 / float(w)))
    y1n = max(0.0, min(1.0, y1 / float(h)))
    x2n = max(0.0, min(1.0, x2 / float(w)))
    y2n = max(0.0, min(1.0, y2 / float(h)))
    if x2n <= x1n or y2n <= y1n:
        return None
    return [round(x1n, 4), round(y1n, 4), round(x2n, 4), round(y2n, 4)]


def _coerce_objects(
    results: List[Any],
    max_boxes: int,
    roi_cfg: Optional[CvRoiConfig],
) -> List[dict]:
    objects: List[dict] = []
    for image_index, res in enumerate(results):
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        names = getattr(res, "names", None) or {}
        if isinstance(names, (list, tuple)):
            names = {i: n for i, n in enumerate(names)}
        elif not isinstance(names, dict):
            names = {}
        w, h = _get_image_shape(res)
        roi_norm = _normalize_roi(roi_cfg, w, h) if roi_cfg else None
        roi_mode = (roi_cfg.mode if roi_cfg else "center").strip().lower()

        try:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            clss = boxes.cls.cpu().tolist()
        except Exception:
            continue

        for idx, box in enumerate(xyxy):
            if len(box) != 4:
                continue
            norm = _normalize_box_xyxy(
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                w,
                h,
            )
            if not norm:
                continue
            cls_id = int(clss[idx]) if idx < len(clss) else -1
            label = str(names.get(cls_id, cls_id))
            conf = float(confs[idx]) if idx < len(confs) else 0.0
            if roi_norm and not _roi_hit(norm, roi_norm, roi_mode):
                continue
            objects.append(
                {
                    "label": label,
                    "confidence": round(max(0.0, min(1.0, conf)), 2),
                    "box": norm,
                    "image_index": image_index,
                }
            )

    objects.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    if max_boxes > 0 and len(objects) > max_boxes:
        objects = objects[:max_boxes]
    return objects


def worker_b_cv(
    q_video: Queue,
    q_vlm: Queue,
    q_ctrl: Queue,
    stop: object,
    model: MODEL,
    rtsp_batch_config: Optional[RTSPBatchConfig] = None,
    cv_config: Optional[CvConfig] = None,
    vlm_config: Optional[VlmConfig] = None,
):
    running = False
    paused = False
    # If caller doesn't provide a config, build from env with safe device defaults.
    cv_config = cv_config or build_cv_config()

    model_tag = f"cv:{os.path.basename(cv_config.model_path)}"

    use_ascend_om = str(cv_config.model_path).lower().endswith(".om")
    detector = None
    om_detector = None

    if use_ascend_om:
        if cv_config.task in {"fall", "smoke", "fight"}:
            err = "Ascend OM 后端暂不支持 pose 任务（fall/smoke/fight），请先用 .pt 或提供 pose 的 .om"
            logger.error("[B-CV] %s", err)
            _emit_done(
                q_vlm,
                seg_idx=-1,
                full_text="",
                model=model_tag,
                item={},
                usage={"status": "model_load_error", "backend": "ascend_om", "error": err},
                latency_ms=0,
                streaming=False,
                is_task=True,
                is_json_format=True,
                evidence_images=[],
                evidence_image_urls=[],
                q_ctrl=q_ctrl,
                stop=stop,
            )
            return
        try:
            from app.shared.jetlinks_video.utils.ascend_om_yolo import AscendOmYoloV8Detector

            om_detector = AscendOmYoloV8Detector(cv_config.model_path)
            logger.info("[B-CV] using Ascend OM backend: %s", cv_config.model_path)
        except Exception as e:
            logger.error("[B-CV] Ascend OM init failed: %s", e)
            _emit_done(
                q_vlm,
                seg_idx=-1,
                full_text="",
                model=model_tag,
                item={},
                usage={"status": "import_error", "backend": "ascend_om", "error": str(e)},
                latency_ms=0,
                streaming=False,
                is_task=True,
                is_json_format=True,
                evidence_images=[],
                evidence_image_urls=[],
                q_ctrl=q_ctrl,
                stop=stop,
            )
            return
    else:
        try:
            from ultralytics import YOLO
        except Exception as e:
            logger.error("[B-CV] ultralytics import failed: %s", e)
            _emit_done(
                q_vlm,
                seg_idx=-1,
                full_text="",
                model=model_tag,
                item={},
                usage={"status": "import_error", "error": str(e)},
                latency_ms=0,
                streaming=False,
                is_task=True,
                is_json_format=True,
                evidence_images=[],
                evidence_image_urls=[],
                q_ctrl=q_ctrl,
                stop=stop,
            )
            return

        try:
            detector = YOLO(cv_config.model_path)
        except Exception as e:
            logger.error("[B-CV] model load failed: %s", e)
            _emit_done(
                q_vlm,
                seg_idx=-1,
                full_text="",
                model=model_tag,
                item={},
                usage={"status": "model_load_error", "error": str(e)},
                latency_ms=0,
                streaming=False,
                is_task=True,
                is_json_format=True,
                evidence_images=[],
                evidence_image_urls=[],
                q_ctrl=q_ctrl,
                stop=stop,
            )
            return

    try:
        while True:
            if (not running) or paused:
                try:
                    msg = q_ctrl.get(timeout=0.2)
                except Empty:
                    continue

                if msg is stop:
                    logger.info("[B-CV] STOP received, exit")
                    return

                if isinstance(msg, dict):
                    typ = msg.get("type")
                    if typ in ("START", "RESUME"):
                        running, paused = True, False
                        logger.info("[B-CV] started")
                    elif typ == "PAUSE":
                        paused = True
                        logger.info("[B-CV] paused")
                    elif typ in ("STOP", "SHUTDOWN"):
                        logger.info("[B-CV] STOP received, exit")
                        return
                continue

            try:
                item = q_video.get(timeout=0.1)
            except Empty:
                continue

            try:
                if item is stop:
                    logger.info("[B-CV] data STOP received, exit")
                    return

                if _ctrl_stop_requested(q_ctrl, stop):
                    continue

                seg_idx = int(item.get("segment_index", -1))
                keyframes = item.get("keyframes") or []
                if not keyframes:
                    logger.warning("[B-CV] seg#%s no keyframes, skip", seg_idx)
                    continue

                images = [p for p in keyframes if isinstance(p, str) and p and os.path.exists(p)]
                if not images:
                    logger.warning("[B-CV] seg#%s no valid images, skip", seg_idx)
                    continue

                t_start = time.time()
                roi_cfg = _select_roi_config(cv_config, item.get("stream_rtsp_id"))
                roi_norm = None
                roi_mode = "center"

                if om_detector is not None:
                    try:
                        objects: List[dict] = []
                        for image_index, image_path in enumerate(images):
                            w0, h0, boxes_xyxy, scores, cls = om_detector.predict_one(
                                image_path,
                                conf=cv_config.conf,
                                iou=cv_config.iou,
                                classes=cv_config.classes,
                                max_det=cv_config.max_det,
                            )
                            if roi_cfg:
                                roi_norm = _normalize_roi(roi_cfg, w0, h0)
                                roi_mode = (roi_cfg.mode if roi_cfg else "center").strip().lower()

                            for j in range(int(scores.shape[0])):
                                box = boxes_xyxy[j]
                                norm = _normalize_box_xyxy(
                                    float(box[0]),
                                    float(box[1]),
                                    float(box[2]),
                                    float(box[3]),
                                    w0,
                                    h0,
                                )
                                if not norm:
                                    continue
                                if roi_norm and not _roi_hit(norm, roi_norm, roi_mode):
                                    continue

                                cls_id = int(cls[j]) if j < cls.shape[0] else -1
                                label = om_detector.label_name(cls_id)
                                conf = float(scores[j]) if j < scores.shape[0] else 0.0
                                objects.append(
                                    {
                                        "label": label,
                                        "confidence": round(max(0.0, min(1.0, conf)), 2),
                                        "box": norm,
                                        "image_index": image_index,
                                    }
                                )

                        objects.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
                        if cv_config.max_boxes > 0 and len(objects) > cv_config.max_boxes:
                            objects = objects[: cv_config.max_boxes]
                    except Exception as e:
                        logger.error("[B-CV] Ascend OM inference failed seg#%s: %s", seg_idx, e)
                        _emit_done(
                            q_vlm,
                            seg_idx=seg_idx,
                            full_text="",
                            model=model_tag,
                            item=item,
                            usage={"status": "infer_error", "backend": "ascend_om", "error": str(e)},
                            latency_ms=int((time.time() - t_start) * 1000),
                            streaming=False,
                            is_task=True,
                            is_json_format=True,
                            evidence_images=images,
                            evidence_image_urls=_export_evidence_list_to_static(images, seg_idx, vlm_config),
                            q_ctrl=q_ctrl,
                            stop=stop,
                        )
                        continue
                else:
                    try:
                        results = detector.predict(
                            source=images,
                            conf=cv_config.conf,
                            iou=cv_config.iou,
                            imgsz=cv_config.imgsz,
                            max_det=cv_config.max_det,
                            classes=cv_config.classes,
                            device=cv_config.device,
                            verbose=False,
                        )
                    except Exception as e:
                        logger.error("[B-CV] inference failed seg#%s: %s", seg_idx, e)
                        _emit_done(
                            q_vlm,
                            seg_idx=seg_idx,
                            full_text="",
                            model=model_tag,
                            item=item,
                            usage={"status": "infer_error", "error": str(e)},
                            latency_ms=int((time.time() - t_start) * 1000),
                            streaming=False,
                            is_task=True,
                            is_json_format=True,
                            evidence_images=images,
                            evidence_image_urls=_export_evidence_list_to_static(images, seg_idx, vlm_config),
                            q_ctrl=q_ctrl,
                            stop=stop,
                        )
                        continue

                    if cv_config.task in {"fall", "smoke", "fight"}:
                        objects = _coerce_pose_objects(
                            list(results or []),
                            cv_config.task,
                            cv_config.max_boxes,
                            cv_config.pose,
                            roi_cfg,
                        )
                    else:
                        objects = _coerce_objects(list(results or []), cv_config.max_boxes, roi_cfg)
                evidence_urls = _export_evidence_list_to_static(images, seg_idx, vlm_config)

                _emit_done(
                    q_vlm,
                    seg_idx=seg_idx,
                    full_text=objects,
                    model=model_tag,
                    item=item,
                    usage={
                        "backend": "cv",
                        "engine": ("ascend_om" if om_detector is not None else "ultralytics"),
                        "model": cv_config.model_path,
                        "objects": len(objects),
                    },
                    latency_ms=int((time.time() - t_start) * 1000),
                    streaming=False,
                    is_task=True,
                    is_json_format=True,
                    evidence_images=images,
                    evidence_image_urls=evidence_urls,
                    q_ctrl=q_ctrl,
                    stop=stop,
                )

            finally:
                try:
                    q_video.task_done()
                except Exception:
                    pass
    finally:
        if om_detector is not None:
            try:
                om_detector.close()
            except Exception:
                pass
