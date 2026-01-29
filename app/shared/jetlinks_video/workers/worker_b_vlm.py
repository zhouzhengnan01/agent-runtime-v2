# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import uuid
import json
import re
from typing import Any, List, Optional, Tuple
from queue import Queue, Empty
from datetime import datetime
from pathlib import Path

from app.shared.jetlinks_video.utils import logger_utils
from app.shared.jetlinks_video.utils.vlm_client.vlm_curl_client import VlmCurlClient
from app.shared.jetlinks_video.configs.rtsp_batch_config import RTSPBatchConfig
from app.shared.jetlinks_video.configs.vlm_config import VlmConfig
from app.shared.jetlinks_video.all_enum import MODEL

logger = logger_utils.get_logger(__name__)


# =============================================================================
# .env 读取（不依赖 os.environ；若找不到 .env 则只返回空配置）
# =============================================================================
def _find_project_root(start_file: str) -> Path:
    """
    从当前文件向上找项目根（以包含 app/ 目录为准），找不到就回退到当前文件父目录。
    """
    p = Path(start_file).resolve()
    cur = p.parent
    for _ in range(10):
        if (cur / "app").exists() and (cur / "app").is_dir():
            return cur
        cur = cur.parent
    return p.parent


def _parse_dotenv_file(dotenv_path: Path) -> dict:
    """
    极简 .env 解析：KEY=VALUE，支持引号，忽略注释行。
    """
    out: dict = {}
    try:
        text = dotenv_path.read_text(encoding="utf-8")
    except Exception:
        return out

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        # 去掉行内注释（非常保守：仅在未被引号包住时处理）
        if v and v[0] not in ("'", '"'):
            v = v.split("#", 1)[0].strip()
        # 去引号
        if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
            v = v[1:-1]
        out[k] = v
    return out


_PROJECT_ROOT = _find_project_root(__file__)
_ENV_PATH = (_PROJECT_ROOT / ".env").resolve()
_ENV_MAP = _parse_dotenv_file(_ENV_PATH) if _ENV_PATH.exists() else {}


def _env_get_bool(key: str, default: bool = False) -> bool:
    raw = _ENV_MAP.get(key, None)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _safe_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _dump_raw_append(dump_id: str, text: str, usage: Optional[dict] = None) -> None:
    """
    追加写入：storage/logs/<id>
    - 由 .env 的 VLM_DUMP_RAW 控制开关
    - 每次记录：时间戳 + usage(可选) + 原文 + 空行
    """
    if not _env_get_bool("VLM_DUMP_RAW", default=False):
        return

    try:
        logs_dir = (_PROJECT_ROOT / "storage" / "logs").resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)

        fname = _safe_filename(dump_id)
        fp = logs_dir / fname  # 按你的要求：storage/logs/<id>（不加扩展名）

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with fp.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}]")
            if isinstance(usage, dict) and usage:
                # 不要太啰嗦：写一行精简 usage（可自行删）
                try:
                    brief = json.dumps(usage, ensure_ascii=False)
                    f.write(f" usage={brief}")
                except Exception:
                    pass
            f.write("\n")
            f.write(text or "")
            f.write("\n\n")
    except Exception:
        # dump 失败不能影响主流程
        return


# ----------------- 文件与 URI 辅助 -----------------
def _is_http_url(p: str) -> bool:
    return isinstance(p, str) and p.lower().startswith(("http://", "https://"))


def _to_file_uri(path_or_uri: str) -> Optional[str]:
    if not path_or_uri:
        return None
    if _is_http_url(path_or_uri) or path_or_uri.lower().startswith("file://"):
        return path_or_uri
    abs_path = os.path.abspath(path_or_uri).replace("\\", "/")
    return f"file://{abs_path}"


def _uri_to_local_path(file_uri: str) -> Optional[str]:
    if not file_uri:
        return None
    if file_uri.lower().startswith("file://"):
        return file_uri[len("file://") :]
    return None


def _exists_local_from_uri(file_uri_or_path: str) -> bool:
    if not file_uri_or_path:
        return False
    if _is_http_url(file_uri_or_path):
        return True
    if file_uri_or_path.lower().startswith("file://"):
        local = file_uri_or_path[len("file://") :]
        return os.path.exists(local)
    return os.path.exists(file_uri_or_path)


def _export_evidence_list_to_static(
    evidence_images: List[str],
    seg_idx: int,
    vlm_config: Optional[VlmConfig] = None,
) -> List[str]:
    out_urls: List[str] = []
    if not evidence_images:
        return out_urls

    if vlm_config is not None and getattr(vlm_config, "vlm_static_evidence_images_dir", None):
        static_dir = vlm_config.vlm_static_evidence_images_dir
    else:
        static_dir = os.path.join(os.getcwd(), "static", "evidence_images")

    if vlm_config is not None and getattr(vlm_config, "vlm_static_evidence_images_url_prefix", None):
        url_prefix = vlm_config.vlm_static_evidence_images_url_prefix
    else:
        url_prefix = "/static/evidence_images"

    url_prefix = url_prefix.rstrip("/")
    os.makedirs(static_dir, exist_ok=True)

    for i, uri in enumerate(evidence_images):
        if _is_http_url(uri):
            out_urls.append(uri)
            continue

        local_path = _uri_to_local_path(uri) or uri
        if not os.path.exists(local_path):
            continue

        ext = os.path.splitext(local_path)[1] or ".jpg"
        fname = f"seg{seg_idx:04d}_evdimg{i:02d}_{uuid.uuid4().hex}{ext}"
        dst = os.path.join(static_dir, fname)

        try:
            if os.path.abspath(local_path) != os.path.abspath(dst):
                import shutil

                shutil.copy2(local_path, dst)
            out_urls.append(f"{url_prefix}/{fname}")
        except Exception as e:
            logger.warning(f"[B] 导出证据帧失败：{e}")

    return out_urls


# ----------------- STOP 感知 & 安全投递 -----------------
def _ctrl_stop_requested(q_ctrl: Queue | None, stop: object | None) -> bool:
    if q_ctrl is None or stop is None:
        return False
    try:
        while True:
            msg = q_ctrl.get_nowait()
            if (msg is stop) or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                logger.warning("[B] 检测到控制队列 STOP 哨兵, 停止本次投递")
                return True
            try:
                q_ctrl.put_nowait(msg)
            except Exception:
                pass
            return False
    except Empty:
        return False


def _q_put_with_retry(
    q: Queue,
    obj: Any,
    *,
    tries: int = 3,
    timeout: float = 0.5,
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
    drop_on_stop: bool = True,
    drop_on_timeout: bool = True,
) -> bool:
    if _ctrl_stop_requested(q_ctrl, stop):
        return not drop_on_stop
    for i in range(1, tries + 1):
        try:
            q.put(obj, timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"[B] q_vlm.put 超时（第{i}/{tries}次）：{e}")
            if _ctrl_stop_requested(q_ctrl, stop):
                return not drop_on_stop
    if drop_on_timeout:
        return False
    try:
        q.put(obj, timeout=0.2)
        return True
    except Exception:
        return False


# ----------------- 事件发包 -----------------
def _emit_delta(
    q_vlm: Queue,
    seg_idx: int,
    delta: str,
    seq: int,
    model: str,
    item: dict,
    *,
    streaming: bool,
    is_task: bool,
    is_json_format: bool,
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
) -> None:
    _q_put_with_retry(
        q_vlm,
        {
            "type": "vlm_stream_delta",
            "segment_index": seg_idx,
            "delta": delta,
            "seq": seq,
            "model": model,
            "streaming": bool(streaming),
            "is_task": bool(is_task),
            "is_json_format": bool(is_json_format),
            "produce_ts": time.time(),
            "clip_t0": item.get("t0"),
            "clip_t1": item.get("t1"),
            "frame_pts": item.get("frame_pts") or [],
            "frame_indices": item.get("frame_indices") or [],
            "t0_iso": item.get("t0_iso"),
            "t1_iso": item.get("t1_iso"),
            "t0_epoch": item.get("t0_epoch"),
            "t1_epoch": item.get("t1_epoch"),
            "frame_epoch": item.get("frame_epoch") or [],
            "frame_iso": item.get("frame_iso") or [],
            "small_video_fps": (item.get("policy") or {}).get("encode", {}).get("fps"),
            "origin_policy": (item.get("policy") or {}).get("policy_used"),
            "stream_url": item.get("stream_url"),
            "stream_index": item.get("stream_index"),
            "stream_segment_index": item.get("stream_segment_index"),
            "window_index_in_stream": item.get("window_index_in_stream"),
            "polling_round_index": item.get("polling_round_index"),
            "stream_rtsp_id": item.get("stream_rtsp_id"),
        },
        q_ctrl=q_ctrl,
        stop=stop,
        drop_on_stop=True,
        drop_on_timeout=True,
    )


def _emit_done(
    q_vlm: Queue,
    seg_idx: int,
    full_text: str,
    model: str,
    item: dict,
    *,
    usage: dict | None,
    latency_ms: int,
    streaming: bool,
    is_task: bool,
    is_json_format: bool,
    evidence_images: Optional[List[str]] = None,
    evidence_image_urls: Optional[List[str]] = None,
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
) -> None:
    _q_put_with_retry(
        q_vlm,
        {
            "type": "vlm_stream_done",
            "segment_index": seg_idx,
            "full_text": full_text or "",
            "usage": usage,
            "model": model,
            "streaming": bool(streaming),
            "is_task": bool(is_task),
            "is_json_format": bool(is_json_format),
            "latency_ms": latency_ms,
            "produce_ts": time.time(),
            "clip_t0": item.get("t0"),
            "clip_t1": item.get("t1"),
            "frame_pts": item.get("frame_pts") or [],
            "frame_indices": item.get("frame_indices") or [],
            "t0_iso": item.get("t0_iso"),
            "t1_iso": item.get("t1_iso"),
            "t0_epoch": item.get("t0_epoch"),
            "t1_epoch": item.get("t1_epoch"),
            "frame_epoch": item.get("frame_epoch") or [],
            "frame_iso": item.get("frame_iso") or [],
            "evidence_images": evidence_images or [],
            "evidence_image_urls": evidence_image_urls or [],
            "small_video_fps": (item.get("policy") or {}).get("encode", {}).get("fps"),
            "origin_policy": (item.get("policy") or {}).get("policy_used"),
            "stream_url": item.get("stream_url"),
            "stream_index": item.get("stream_index"),
            "stream_segment_index": item.get("stream_segment_index"),
            "window_index_in_stream": item.get("window_index_in_stream"),
            "polling_round_index": item.get("polling_round_index"),
            "stream_rtsp_id": item.get("stream_rtsp_id"),
        },
        q_ctrl=q_ctrl,
        stop=stop,
        drop_on_stop=True,
        drop_on_timeout=True,
    )


def _model_tag_from_cfg(cfg: VlmConfig) -> str:
    m = getattr(cfg, "vlm_model_name", None)
    if m is None:
        return "unknown"
    return getattr(m, "value", None) or str(m)


def _pick_dump_id(item: dict, vlm_config: Optional[VlmConfig], model_tag: str) -> str:
    """
    你要求 storage/logs/<id>：这里尽量选“任务级”id，避免不同任务混写。
    优先级：
      1) item.task_id / item.rpc_id / item.session_id
      2) item.stream_rtsp_id
      3) model_tag
    """
    for k in ("task_id", "rpc_id", "session_id", "taskId", "rpcId", "sessionId"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    v = item.get("stream_rtsp_id")
    if v is not None and str(v).strip():
        return str(v).strip()
    return model_tag or "unknown"


# ----------------- 入口（B 线程主体） -----------------
def worker_b_vlm(
    q_video: Queue,
    q_vlm: Queue,
    q_ctrl: Queue,
    stop: object,
    model: MODEL,
    rtsp_batch_config: Optional[RTSPBatchConfig] = None,
    vlm_config: Optional[VlmConfig] = None,
):
    """
    B 线程（VlmCurlClient 统一调用版）：

    ✅ 改动点（按你的要求）：
    1) 保留 task + json 时短 user_prompt（强制只吐 JSON，提升遵从率）
    2) 完全移除“task + json 校验并输出严格 JSON”的逻辑：B 侧不再解析/校验/修复模型输出
    3) 调用失败时不再按 task/json 强制返回 "[]"：直接透传空字符串或错误信息给上层（这里采用空字符串 + usage.error）
    4) 增加原文落盘开关：读取 .env 的 VLM_DUMP_RAW
       - False/缺省：不落盘
       - True：写入 storage/logs/<id>（单文件追加，带时间戳，每条记录后空行）
    """

    running = False
    paused = False

    vlm_config = vlm_config or VlmConfig()

    is_task = model in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING)
    is_json_format = bool(getattr(vlm_config, "is_json_format", False))
    max_frames_cap = int(getattr(vlm_config, "vlm_max_frames", 1) or 1)
    model_tag = _model_tag_from_cfg(vlm_config)

    # ✅ task + json 时给短 user_prompt（强制只吐 JSON），提升遵从率
    user_prompt = None
    if is_task and is_json_format:
        user_prompt = "只输出严格可解析的 JSON 数组，不要输出任何解释、Markdown、代码块。"

    try:
        while True:
            # ---------- 等待 START/RESUME ----------
            if (not running) or paused:
                try:
                    msg = q_ctrl.get(timeout=0.2)
                except Empty:
                    continue

                if msg is stop:
                    logger.info("[B] 收到 STOP，退出")
                    return

                if isinstance(msg, dict):
                    typ = msg.get("type")
                    if typ in ("START", "RESUME"):
                        running, paused = True, False
                        logger.info("[B] 启动视觉解析")
                    elif typ == "PAUSE":
                        paused = True
                        logger.info("[B] 暂停视觉解析")
                    elif typ in ("STOP", "SHUTDOWN"):
                        logger.info("[B] 收到 STOP，退出。")
                        return
                continue

            # ---------- 取片段 ----------
            try:
                item = q_video.get(timeout=0.1)
            except Empty:
                continue

            try:
                if item is stop:
                    logger.info("[B] 收到数据队列 STOP，退出")
                    return

                seg_idx = int(item.get("segment_index", -1))

                evidence_images: List[str] = []
                evidence_image_urls: List[str] = []

                small_video = item.get("small_video")
                keyframes = item.get("keyframes") or []
                frame_pts = item.get("frame_pts") or []

                if is_task:
                    logger.info(
                        "[B] 正在处理片段：seg#%d | 流索引=%s | 流地址=%s | 当前流切片序号=%s | 轮询轮次=%s | rtsp_id=%s | json=%s",
                        seg_idx,
                        item.get("stream_index", "-"),
                        item.get("stream_url", "-") or "-",
                        item.get("stream_segment_index", "-"),
                        item.get("polling_round_index", "-"),
                        item.get("stream_rtsp_id", "-"),
                        is_json_format,
                    )

                # ---------- 1) system_prompt：上游优先，兼容旧字段 ----------
                system_prompt = item.get("system_prompt")
                if not system_prompt:
                    if model == MODEL.OFFLINE:
                        system_prompt = getattr(vlm_config, "offline_system_prompt", "") or ""
                    elif model in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING):
                        system_prompt = item.get("rtsp_system_prompt")
                        if (not system_prompt) and rtsp_batch_config:
                            try:
                                stream_url = item.get("stream_url")
                                if stream_url and rtsp_batch_config.polling_list:
                                    for it in rtsp_batch_config.polling_list:
                                        if getattr(it, "rtsp_url", None) == stream_url:
                                            system_prompt = (
                                                getattr(it, "rtsp_system_prompt", None)
                                                or getattr(it, "rtsp_prompt", None)
                                            )
                                            break
                            except Exception:
                                pass
                        system_prompt = system_prompt or ""
                    else:
                        system_prompt = ""

                logger.info("[B] seg#%d, 最终 system_prompt 前 400 字：%s", seg_idx, (system_prompt or "")[:400])

                # ---------- 2) 选择图片 / 视频输入 ----------
                images: List[str] = []
                videos: Optional[List[str]] = None

                if keyframes and frame_pts:
                    n = min(len(keyframes), len(frame_pts))
                    if n > 0:
                        cap = min(max_frames_cap, n) if max_frames_cap > 0 else n
                        for p in keyframes[:cap]:
                            u = _to_file_uri(p)
                            if u and (_exists_local_from_uri(u) or _is_http_url(u)):
                                images.append(u)

                        evidence_images = list(images)
                        evidence_image_urls = _export_evidence_list_to_static(evidence_images, seg_idx, vlm_config)
                    else:
                        logger.warning("[B] seg#%d keyframes/frame_pts 长度为 0，回退 small_video。", seg_idx)

                if (not images) and small_video:
                    vuri = _to_file_uri(small_video)
                    if vuri and _exists_local_from_uri(vuri):
                        videos = [vuri]

                if (not images) and (not videos):
                    logger.warning("[B] seg#%d 无可用的视频/图片 URI，跳过该段。", seg_idx)
                    continue

                # ---------- 3) 统一调用 VlmCurlClient ----------
                t_start = time.time()
                try:
                    mode_used, iter_pair, nonstream_pair = VlmCurlClient(
                        vlm_config=vlm_config,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        text=None,
                        images=images,
                        videos=videos,
                        q_ctrl=q_ctrl,
                        stop=stop,
                    ).infer()
                except Exception as e:
                    logger.error("[B] VLM 调用失败（seg#%d）：%s", seg_idx, e)
                    # ✅ 不再 task/json 强制 "[]"
                    usage = {"status": "call_error", "error": str(e)}
                    out_text = ""  # 让上层决定怎么兜底/重试/降级
                    # dump 原文（这里是“失败信息”）
                    dump_id = _pick_dump_id(item, vlm_config, model_tag)
                    _dump_raw_append(dump_id, f"[CALL_ERROR] {e}", usage=usage)

                    _emit_done(
                        q_vlm,
                        seg_idx,
                        full_text=out_text,
                        model=model_tag,
                        item=item,
                        usage=usage,
                        latency_ms=int((time.time() - t_start) * 1000),
                        streaming=False,
                        is_task=is_task,
                        is_json_format=is_json_format,
                        evidence_images=evidence_images,
                        evidence_image_urls=evidence_image_urls,
                        q_ctrl=q_ctrl,
                        stop=stop,
                    )
                    continue

                # ---------- 4) 输出（流式/非流式） ----------
                usage: dict | None = None
                final_text = ""
                streaming_flag = False

                if mode_used == "stream":
                    streaming_flag = True
                    seq = 1
                    buf: List[str] = []
                    for delta, usage_part in iter_pair:  # type: ignore
                        if delta:
                            buf.append(delta)
                            _emit_delta(
                                q_vlm,
                                seg_idx,
                                delta,
                                seq,
                                model_tag,
                                item,
                                streaming=True,
                                is_task=is_task,
                                is_json_format=is_json_format,
                                q_ctrl=q_ctrl,
                                stop=stop,
                            )
                            seq += 1
                        if usage_part:
                            usage = usage_part
                    final_text = "".join(buf)
                else:
                    final_text, usage = nonstream_pair or ("", None)  # type: ignore

                out_text = (final_text or "").strip()

                # ---------- 5) ✅ B侧不做任何校验：直接透传 ----------
                # 但支持按 .env 开关做原文落盘（单文件追加）
                dump_id = _pick_dump_id(item, vlm_config, model_tag)
                _dump_raw_append(dump_id, out_text, usage=usage)

                # ---------- 6) done 投递 ----------
                _emit_done(
                    q_vlm,
                    seg_idx,
                    full_text=out_text,
                    model=model_tag,
                    item=item,
                    usage=usage,
                    latency_ms=int((time.time() - t_start) * 1000),
                    streaming=streaming_flag,
                    is_task=is_task,
                    is_json_format=is_json_format,
                    evidence_images=evidence_images,
                    evidence_image_urls=evidence_image_urls,
                    q_ctrl=q_ctrl,
                    stop=stop,
                )

            finally:
                try:
                    q_video.task_done()
                except Exception:
                    pass

    finally:
        logger.info("[B] 线程退出清理完成。")
