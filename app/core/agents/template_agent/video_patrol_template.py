# -*- coding: utf-8 -*-
"""
JetLinks 视频巡检智能体模板（只从 .env 文件读取配置，不读取 OS 环境变量）

- 聊天侧只支持：OFFLINE / SECURITY_SINGLE
- 机器视觉侧只支持：SECURITY_SINGLE / SECURITY_POLLING
- 统一并发控制：VLM_MAX_THREAD（聊天 + 机器视觉共用）

B 侧输出：/storage/xxx.jpg
本模板负责：拼成 http(s)://host:port/storage/xxx.jpg

HTTP 离线（OFFLINE）：
- http(s) URL -> 下载到 storage/http_tmp/<sessionId>/... -> 传给底层 OFFLINE 本地路径

清理逻辑：
1) Chat 任务：会话结束（收到“停止分析/巡检” 或 到了停止时间 或 任务自然结束）就按 sessionId 清理：
   - storage/evidence_images/<sessionId>
   - storage/out/<sessionId>
   - storage/http_tmp/<sessionId>
2) 机器视觉任务：
   - 收到 {"state":"stop"}：立即按 rpc_id 清理并注销：
     - storage/evidence_images/<id>
     - storage/evidence_images_box/<id>
     - storage/out/<id>
   - _handle_machine_request finally：再兜底清理一次（即使异常也确保清掉）
3) 断链（ws disconnect）：
   - 立即 stop analyzer + 立即清理 + 立即注销

额外：Machine 运行中防爆盘 GC（关键需求）
- 只关注“正在运行”的 rpc_id
- 每 _MACHINE_GC_INTERVAL_SEC 秒触发一次
- 对每个正在运行 rpc_id：不删 <id> 根目录，但删除 <id> 目录内 mtime <= (now - _MACHINE_GC_EXPIRE_SEC) 的旧文件，并清空空子目录
  目录范围：
  - storage/evidence_images/<id>
  - storage/evidence_images_box/<id>
  - storage/out/<id>

说明（重要）：
- 底层B 侧已经改成“完全透传模型原始输出”，不再做 JSON 校验/修复。
- 因此：本文件（上层）负责对 full_text 做“宽松解析 + 修复常见格式错误 + 统一为 objects[]”，
  以便后续画框逻辑稳定运行（即使模型偶尔输出 ```json ...```、多余文本、尾逗号、box 少 '[' 等）。
"""

from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

# --------------------
# 动态加入项目根目录
# --------------------
current_file = Path(__file__).resolve()
if current_file.name == "video_patrol_template.py":
    project_root = current_file.parent.parent.parent.parent
    if project_root.name == "app":
        project_root = project_root.parent
else:
    project_root = current_file.parent.parent.parent.parent
    if project_root.name == "app":
        project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# =============================================================================
# .env 只读加载（不注入 os.environ）
# =============================================================================
ENV_FILE_PATH = (project_root / ".env").resolve()
try:
    from dotenv import dotenv_values  # type: ignore
except Exception as e:  # noqa: BLE001
    raise RuntimeError("缺少依赖 python-dotenv：请安装 pip install python-dotenv") from e

if not ENV_FILE_PATH.exists():
    raise RuntimeError(f".env 不存在：{str(ENV_FILE_PATH)}（当前实现仅支持从该路径读取）")

_ENV_MAP = dotenv_values(str(ENV_FILE_PATH))  # 只读文件，不会写入 os.environ


def _env_get_optional(key: str) -> Optional[str]:
    raw = _ENV_MAP.get(key)
    if raw is None:
        return None
    v = str(raw).strip()
    return v if v else None


def _env_get_str(key: str, default: str = "") -> str:
    v = _env_get_optional(key)
    return v if v is not None else default


def _env_get_int(key: str, default: int) -> int:
    v = _env_get_optional(key)
    if v is None:
        return default
    try:
        n = int(float(v))
        return n
    except Exception:
        return default


def _env_get_float(key: str, default: float) -> float:
    v = _env_get_optional(key)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_get_bool(key: str, default: bool = False) -> bool:
    v = _env_get_optional(key)
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
    return default


# --------------------
# 内部模块引入
# --------------------
from app.core.agents.template_agent.base_template import BaseTemplateAgent
from app.shared.jetlinks_video.all_enum import MODEL
from app.shared.jetlinks_video.configs.cv_config import (
    CvConfig,
    CvPoseConfig,
    CvRoiConfig,
    build_cv_config,
)
from app.shared.jetlinks_video.configs.rtsp_batch_config import RTSPBatchConfig, RTSP
from app.shared.jetlinks_video.configs.runtime_machine_config import RuntimeMachineConfig
from app.shared.jetlinks_video.configs.vlm_config import VlmConfig
from app.shared.jetlinks_video.utils.logger_utils import get_logger
from app.shared.jetlinks_video.utils.object_detection_bounding_box import export_evidence_images_with_boxes
from app.shared.streaming_analyze import StreamingAnalyze

logger = get_logger(__name__)
logger.info("[ENV] Loaded .env values from %s (file-only mode)", str(ENV_FILE_PATH))

# =============================================================================
# Machine 任务：追加到每路 source.prompt 后的统一约束（对象检测框）
# =============================================================================

# 最终返回值中的 full_text 形式如下：
"""
full_text:[{
        "objects": [
            {
                "label": "对象标签",
                "confidence": 0.00,
                "box": [x1, y1, x2, y2]
            }
        ]
    }]
"""

VLM_OBJECT_DETECT_JSON_CONSTRAINTS_8B_Q8 = r"""
【任务定位：画框数据（Bounding Boxes）输出器】
你不是巡检报告生成器，也不输出事件/风险等级/处置建议。
你只输出“用于画框”的检测结果（类似 YOLO 输出）。

【强制输出格式】
- 你的最终回答必须是“严格可解析”的 JSON 数组（JSONArray）。
- 除 JSON 数组本体外，禁止输出任何其它字符：禁止解释、禁止 Markdown、禁止代码块、禁止前后缀文字。

【只允许的字段】
数组内每个元素必须是一个 JSON 对象，并且“只允许”包含以下 3 个字段：
1) "label"      : string
2) "confidence" : number（0.00~1.00，保留两位小数）
3) "box"        : [x1, y1, x2, y2]（四个数值坐标，采用 0~1 归一化坐标系；每个值保留 4 位小数）

【严禁输出的字段（出现即视为错误）】
- 严禁输出任何巡检报告/事件结构字段，包括但不限于：
  "type", "describe", "level", "suggestion", "objects", "events",
  "risk", "severity", "reason", "action", "advice", "summary",
  以及任何不在【只允许的字段】列表中的字段。
- 也不要输出嵌套结构，例如 {"objects":[...]} 或 [{"objects":[...]}]。

【box 字段规则（0~1 归一化坐标系）】
- box 必须是长度为 4 的数组：[x1, y1, x2, y2]，每项为 0~1 之间的数值，保留 4 位小数。
- (x1, y1) 为左上角；(x2, y2) 为右下角。
- 必须满足 x1 < x2 且 y1 < y2。
- 任一坐标若超出 [0,1]，必须裁剪到 [0,1]（且仍需满足 x1<x2, y1<y2）。
- 重要：坐标必须基于你“实际看到的画面内容”比例输出即可（不要输出像素坐标）。

【数量/截断保护（非常重要）】
- 最多只允许输出 12 个目标（JSONArray 长度 <= 12）。
- 若检测到超过 12 个目标：只保留 confidence 最高的 12 个，其余全部丢弃。
- 输出前必须按 confidence 从高到低排序（降序）。
- 不要生成任何“均匀排列/规则网格/填充式”的虚构框；只对真实可见目标输出框。

【label 规则】
- 若上方任务描述给出了 label 枚举/候选值：必须从中选择。
- 若未给出：使用最贴切的简短英文对象名称（例如：person/vehicle/fire/smoke等），不要写长句。

【没有目标】
- 若没有任何目标：输出空数组 []（只输出这两个字符及中括号内容，不加其它任何字）。

【多帧/多图输入时】
- 即使输入包含多张图片或多帧，你也只能输出“一个” JSON 数组。
- 把所有帧中检测到的目标合并到同一个数组中输出（允许重复；不要做复杂去重）。
- 合并后仍必须满足“最多 12 个、按 confidence 排序、只取 top12”。

【示例（正确）】
[
  {"label":"person","confidence":0.93,"box":[0.0469,0.0625,0.1719,0.2812]},
  {"label":"person","confidence":0.88,"box":[0.2031,0.0703,0.3125,0.2891]}
]

【再次强调】
- 最终只输出 JSON 数组本体。
- 数组元素只允许 label/confidence/box 三字段，其它字段一律禁止。
"""

VLM_OBJECT_DETECT_JSON_CONSTRAINTS_8B_Q4 = r"""
你是“画框(BBox)JSON 输出器”。只做目标检测框输出。

必须只输出：JSON数组。禁止任何其它字符/解释/Markdown。

数组元素只能有3个字段（多一个都算错）：
- label: string
- confidence: number(0.00~1.00, 保留2位小数)
- box: [x1,y1,x2,y2]  0~1归一化坐标，保留4位小数

box规则：
- 0.0000 <= x1,y1,x2,y2 <= 1.0000
- x1 < x2 且 y1 < y2
- 任一坐标越界必须裁剪回[0,1]后再输出

数量限制：
- 最多输出12个框
- 必须按confidence降序，只保留top12

禁止字段（出现即视为不合格，必须改为输出[]）：
bbox_2d, timestamp, objects, events, type, describe, level, suggestion, summary

若没有目标：输出 [] （只输出这两个字符和括号）

【示例（正确）】
[
  {"label":"person","confidence":0.93,"box":[0.0469,0.0625,0.1719,0.2812]},
  {"label":"person","confidence":0.88,"box":[0.2031,0.0703,0.3125,0.2891]}
]

最终只输出JSON数组本体。
"""

# 路径与清理工具相关路径
_STORAGE_ROOT = (project_root / "storage").resolve()
_EVIDENCE_IMAGES_ROOT = (_STORAGE_ROOT / "evidence_images").resolve()
_EVIDENCE_IMAGES_BOX_ROOT = (_STORAGE_ROOT / "evidence_images_box").resolve()
_OUT_ROOT = (_STORAGE_ROOT / "out").resolve()
_HTTP_TMP_ROOT = (_STORAGE_ROOT / "http_tmp").resolve()

# Machine：运行中防爆盘 GC（目录内按时间删旧文件，不删根目录）相关配置
_MACHINE_GC_INTERVAL_SEC = _env_get_int("MACHINE_GC_INTERVAL_SEC", 3600)  # 触发频率, 单位秒（0=禁用）
_MACHINE_GC_EXPIRE_SEC = _env_get_int("MACHINE_GC_EXPIRE_SEC", 216000)  # 删除 mtime <= now - 阈值 的旧文件（单位秒）

# HTTP 离线下载配置
_HTTP_TMP_MAX_MB = _env_get_str("HTTP_OFFLINE_MAX_MB", "")
_HTTP_DOWNLOAD_TIMEOUT_SEC = _env_get_str("HTTP_OFFLINE_DOWNLOAD_TIMEOUT_SEC", "60")

for _p in (_STORAGE_ROOT, _EVIDENCE_IMAGES_ROOT, _EVIDENCE_IMAGES_BOX_ROOT, _OUT_ROOT, _HTTP_TMP_ROOT):
    _p.mkdir(parents=True, exist_ok=True)


def _sanitize_dir_name(x: str) -> str:
    """
    仅用于 http_tmp 的 sessionId 目录名（避免奇怪字符导致目录异常）
    注意：底层 out/evidence_images 是直接用 task_id（未 sanitize），这里不要对它们做 sanitize，否则删不到。
    """
    return re.sub(r"[^0-9A-Za-z._-]+", "_", x or "")


def _safe_rmtree(p: Path) -> None:
    """
    只允许删除 storage 目录下的子目录；并且不抛异常（避免上游 websocket 断连）。
    """
    try:
        p = p.resolve()
        if _STORAGE_ROOT not in p.parents and p != _STORAGE_ROOT:
            logger.warning("[CLEANUP] refuse to delete path outside storage: %s", str(p))
            return
        if not p.exists():
            return
        shutil.rmtree(str(p), ignore_errors=True)
        logger.info("[CLEANUP] deleted: %s", str(p))
    except Exception as e:
        logger.warning("[CLEANUP] delete failed: path=%s err=%s", str(p), e)


def cleanup_chat_storage_sync(session_id: str) -> None:
    """
    Chat：按 sessionId 删除
    - storage/evidence_images/<sessionId>
    - storage/out/<sessionId>
    - storage/http_tmp/<sessionId>   （http_tmp 目录名使用 sanitize 规则）
    """
    sid = str(session_id or "").strip()
    if not sid:
        return

    _safe_rmtree(_EVIDENCE_IMAGES_ROOT / sid)
    _safe_rmtree(_OUT_ROOT / sid)
    _safe_rmtree(_HTTP_TMP_ROOT / _sanitize_dir_name(sid))


def cleanup_machine_storage_sync(rpc_id: Any) -> None:
    """
    Machine：按 rpc 响应 id 删除
    - storage/evidence_images/<id>
    - storage/evidence_images_box/<id>
    - storage/out/<id>
    """
    tid = str(rpc_id).strip()
    if not tid:
        return
    _safe_rmtree(_EVIDENCE_IMAGES_ROOT / tid)
    _safe_rmtree(_EVIDENCE_IMAGES_BOX_ROOT / tid)
    _safe_rmtree(_OUT_ROOT / tid)


def _safe_unlink(fp: Path) -> bool:
    try:
        fp.unlink(missing_ok=True)  # py3.8+
        return True
    except TypeError:
        try:
            if fp.exists():
                fp.unlink()
            return True
        except Exception:
            return False
    except Exception:
        return False


def _safe_rmdir_if_empty(dp: Path) -> bool:
    try:
        if not dp.exists() or not dp.is_dir():
            return False
        if any(dp.iterdir()):
            return False
        dp.rmdir()
        return True
    except Exception:
        return False


def cleanup_running_task_dir_older_than_sync(task_dir: Path, threshold_ts: float) -> int:
    """
    对单个 task_dir（例如 storage/out/<id>）做“目录内清理”：
    - 删除 mtime <= threshold_ts 的文件
    - 递归清理空目录（但保留 task_dir 根目录）
    返回：删除的文件数量
    """
    deleted_files = 0
    try:
        task_dir = task_dir.resolve()
        if not task_dir.exists() or not task_dir.is_dir():
            return 0

        # 安全：只允许在 storage 下
        if _STORAGE_ROOT not in task_dir.parents and task_dir != _STORAGE_ROOT:
            logger.warning("[MACHINE_GC] refuse to clean outside storage: %s", str(task_dir))
            return 0

        for root, _, files in os.walk(str(task_dir), topdown=False):
            root_p = Path(root)

            # 删旧文件
            for fn in files:
                fp = root_p / fn
                try:
                    mtime = fp.stat().st_mtime
                except Exception:
                    continue
                if mtime <= threshold_ts:
                    if _safe_unlink(fp):
                        deleted_files += 1

            # 清空空目录（不删根）
            if root_p != task_dir:
                _safe_rmdir_if_empty(root_p)

        return deleted_files
    except Exception as e:
        logger.warning("[MACHINE_GC] cleanup_running_task_dir failed: dir=%s err=%s", str(task_dir), e)
        return deleted_files


def cleanup_running_machine_storage_older_than_sync(rpc_id: Any, threshold_ts: float) -> Dict[str, int]:
    """
    对“正在运行的 rpc_id”做目录内按时间清理。
    返回每个根目录删了多少文件。
    """
    tid = str(rpc_id).strip()
    if not tid:
        return {"evidence_images": 0, "evidence_images_box": 0, "out": 0}

    d1 = (_EVIDENCE_IMAGES_ROOT / tid).resolve()
    d2 = (_EVIDENCE_IMAGES_BOX_ROOT / tid).resolve()
    d3 = (_OUT_ROOT / tid).resolve()

    c1 = cleanup_running_task_dir_older_than_sync(d1, threshold_ts)
    c2 = cleanup_running_task_dir_older_than_sync(d2, threshold_ts)
    c3 = cleanup_running_task_dir_older_than_sync(d3, threshold_ts)

    return {"evidence_images": c1, "evidence_images_box": c2, "out": c3}


# =============================================================================
# HTTP 离线资源：下载到本地 tmp，再传给底层 OFFLINE
# =============================================================================

try:
    HTTP_OFFLINE_MAX_BYTES: Optional[int] = (
        int(float(_HTTP_TMP_MAX_MB) * 1024 * 1024) if _HTTP_TMP_MAX_MB else None
    )
except Exception:
    HTTP_OFFLINE_MAX_BYTES = None

try:
    HTTP_OFFLINE_DOWNLOAD_TIMEOUT_SEC = max(5, int(float(_HTTP_DOWNLOAD_TIMEOUT_SEC)))
except Exception:
    HTTP_OFFLINE_DOWNLOAD_TIMEOUT_SEC = 60


def normalize_http_url(url: str) -> str:
    """
    把可能含中文/空格的 http(s) url 变成 ASCII 可用的 percent-encoded url
    同时兼容 IDN 域名（中文域名 -> punycode）
    """
    p = urllib.parse.urlsplit(url)

    netloc = p.netloc
    try:
        netloc = netloc.encode("idna").decode("ascii")
    except Exception:
        pass

    path = urllib.parse.quote(p.path, safe="/%")
    query = urllib.parse.quote(p.query, safe="=&%")
    fragment = urllib.parse.quote(p.fragment, safe="")

    return urllib.parse.urlunsplit((p.scheme, netloc, path, query, fragment))


def _guess_ext_from_url(url: str) -> str:
    try:
        path = urllib.parse.urlparse(url).path or ""
        base = os.path.basename(path)
        _, ext = os.path.splitext(base)
        ext = (ext or "").lower()
        if ext in (".mp4", ".mov", ".mkv", ".avi", ".flv", ".ts", ".m3u8", ".webm"):
            return ext
    except Exception:
        pass
    return ".mp4"


def _download_http_file_sync(url: str, dst_path: str) -> None:
    safe_url = normalize_http_url(url)

    req = urllib.request.Request(
        safe_url,
        headers={"User-Agent": "JetLinks-VideoPatrol/1.0", "Accept": "*/*"},
        method="GET",
    )

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    tmp_path = dst_path + ".part"

    try:
        with urllib.request.urlopen(req, timeout=HTTP_OFFLINE_DOWNLOAD_TIMEOUT_SEC) as resp:
            try:
                length = resp.headers.get("Content-Length")
                if length and HTTP_OFFLINE_MAX_BYTES is not None:
                    if int(length) > HTTP_OFFLINE_MAX_BYTES:
                        raise RuntimeError(f"HTTP_OFFLINE_MAX_MB exceeded: {int(length)} bytes")
            except Exception as e:
                if "exceeded" in str(e):
                    raise

            written = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if HTTP_OFFLINE_MAX_BYTES is not None and written > HTTP_OFFLINE_MAX_BYTES:
                        raise RuntimeError(f"HTTP_OFFLINE_MAX_MB exceeded while downloading: {written} bytes")

        os.replace(tmp_path, dst_path)

        if not os.path.exists(dst_path) or os.path.getsize(dst_path) <= 0:
            raise RuntimeError("downloaded file is empty")

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


async def download_http_offline_to_tmp(url: str, session_id: str) -> Tuple[str, str]:
    """
    http(s) URL -> 下载到 storage/http_tmp/<sessionId_sanitized>/<uuid>.<ext>
    返回：(local_path, tmp_dir)
    """
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("missing sessionId for http offline download")

    sid_dir = _sanitize_dir_name(sid)
    tmp_dir = str((_HTTP_TMP_ROOT / sid_dir).resolve())
    os.makedirs(tmp_dir, exist_ok=True)

    ext = _guess_ext_from_url(url)
    fname = f"{uuid.uuid4().hex}{ext}"
    local_path = os.path.join(tmp_dir, fname)

    logger.info("[HTTP_OFFLINE] downloading url=%s -> %s", url, local_path)
    await asyncio.to_thread(_download_http_file_sync, url, local_path)
    logger.info("[HTTP_OFFLINE] downloaded ok, size=%d bytes, path=%s", os.path.getsize(local_path), local_path)
    return local_path, tmp_dir


def _noop_stream_delta(_: Dict[str, Any]) -> None:
    return


# =============================================================================
# 同步 generator -> 异步流（线程 + asyncio.Queue）
# =============================================================================
async def _iterate_analyzer_events_async(analyzer: StreamingAnalyze) -> AsyncIterator[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def _put_threadsafe(item: Any) -> None:
        if stop_event.is_set() or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception:
            return

    def _worker() -> None:
        try:
            batch_chars = _env_get_int("ANALYZER_EVENT_BATCH_CHARS", 256)
            batch_interval_s = _env_get_float("ANALYZER_EVENT_BATCH_INTERVAL_S", 0.1)
            if batch_chars <= 0:
                batch_chars = 256
            if batch_interval_s <= 0:
                batch_interval_s = 0.1

            buf_ev: Optional[Dict[str, Any]] = None
            buf_delta_parts: List[str] = []
            buf_delta_len = 0
            last_flush_ts = time.time()

            def _flush_delta(now: Optional[float] = None) -> None:
                nonlocal buf_ev, buf_delta_len, last_flush_ts
                if not buf_ev or buf_delta_len <= 0:
                    buf_ev = None
                    buf_delta_parts.clear()
                    buf_delta_len = 0
                    return
                merged = dict(buf_ev)
                merged["delta"] = "".join(buf_delta_parts)
                _put_threadsafe(merged)
                buf_ev = None
                buf_delta_parts.clear()
                buf_delta_len = 0
                last_flush_ts = now if now is not None else time.time()

            for ev in analyzer.run():
                if stop_event.is_set():
                    break

                ev_type = ev.get("type")
                if ev_type in ("vlm_stream_delta", "asr_stream_delta"):
                    delta = ev.get("delta") or ""
                    if not delta:
                        continue

                    seg_idx = ev.get("segment_index")
                    if (
                        buf_ev is not None
                        and buf_ev.get("type") == ev_type
                        and buf_ev.get("segment_index") == seg_idx
                    ):
                        buf_delta_parts.append(delta)
                        buf_delta_len += len(delta)
                    else:
                        _flush_delta()
                        buf_ev = ev
                        buf_delta_parts.append(delta)
                        buf_delta_len = len(delta)
                        last_flush_ts = time.time()

                    now = time.time()
                    if buf_delta_len >= batch_chars or (now - last_flush_ts) >= batch_interval_s:
                        _flush_delta(now)
                    continue

                _flush_delta()
                _put_threadsafe(ev)

        except Exception as e:
            logger.exception("StreamingAnalyze.run() 线程异常: %s", e)
            _put_threadsafe({"type": "error", "message": str(e)})
        finally:
            try:
                # 兜底 flush
                pass
            finally:
                _put_threadsafe(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield ev
    finally:
        stop_event.set()
        try:
            analyzer.force_stop("async iterator cancelled")
        except Exception:
            pass
        if t.is_alive():
            logger.debug("background analyzer thread still alive on exit")


# =============================================================================
# 任务元信息
# =============================================================================
@dataclass
class TaskMeta:
    key: str
    kind: str  # "chat" | "machine"
    mode: str  # "OFFLINE" | "SECURITY_SINGLE" | "SECURITY_POLLING"
    started_at: float

    run_token: str = field(default_factory=lambda: uuid.uuid4().hex)

    session_id: Optional[str] = None
    rpc_id: Optional[Any] = None
    url: Optional[str] = None

    analyzer: Optional[StreamingAnalyze] = None

    # HTTP 离线下载相关
    local_url: Optional[str] = None
    http_tmp_dir: Optional[str] = None

    # Chat 自动停止控制
    stop_after_sec: Optional[int] = None
    stop_deadline_ts: Optional[float] = None
    stop_task: Optional[asyncio.Task] = None

    # 绑定 websocket client_id（断链时按 client 精确停止并清理）
    client_id: Optional[str] = None


# =============================================================================
# VideoPatrolAgent
# =============================================================================
class VideoPatrolAgent(BaseTemplateAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        name = config.get("name") or "VideoPatrolAgent"
        desc = config.get("description") or "JetLinks 视频巡检智能体"
        version = config.get("version") or "2.0.1"

        super().__init__(name=name, description=desc, version=version, config=config)

        self.agent_name = name
        self.last_active = datetime.now()

        self.max_concurrent = self._load_max_concurrent()
        self.evidence_base_url = self._load_evidence_base_url()
        self.chat_security_single_stop_time = self._load_chat_security_single_stop_time()

        self._task_registry: Dict[str, TaskMeta] = {}

        self.patrol_status: str = "idle"
        self.current_task_id: Optional[str] = None

        self.event_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # ✅ Machine：运行中防爆盘 GC
        self._machine_gc_task: Optional[asyncio.Task] = None
        self._machine_gc_lock = asyncio.Lock()

        logger.info(
            "[VideoPatrol] Initialized VLM_MAX_THREAD=%d, evidence_base_url=%r, chat_security_single_stop=%ss",
            self.max_concurrent,
            self.evidence_base_url,
            self.chat_security_single_stop_time,
        )

    # ------------------------------------------------------------------
    # 并发与 stop timer（全部从 .env 读取）
    # ------------------------------------------------------------------
    def _load_max_concurrent(self) -> int:
        n = _env_get_int("VLM_MAX_THREAD", 6)
        if n <= 0:
            logger.warning("无法解析 VLM_MAX_THREAD（.env），使用默认值 6")
            return 6
        return n

    def _load_chat_security_single_stop_time(self) -> int:
        sec_raw = (_env_get_optional("CHAT_SECURITY_SINGLE_STOP_SEC") or "").strip()
        min_raw = (_env_get_optional("CHAT_SECURITY_SINGLE_STOP_MIN") or "").strip()

        if sec_raw:
            try:
                v = int(float(sec_raw))
                return max(10, v)
            except Exception:
                pass

        if min_raw:
            try:
                v = int(float(min_raw)) * 60
                return max(10, v)
            except Exception:
                pass

        return 60 * 10

    def _make_chat_task_key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    def _make_machine_task_key(self, rpc_id: Any) -> str:
        return f"rpc:{rpc_id}"

    def _get_active_task_count(self) -> int:
        return sum(1 for meta in self._task_registry.values() if meta.analyzer is not None)

    # ------------------------------------------------------------------
    # Machine：GC loop（统一管理当前所有正在运行 rpc_id）
    # ------------------------------------------------------------------
    def _active_machine_rpc_ids(self) -> List[str]:
        out: List[str] = []
        for m in self._task_registry.values():
            if m.kind == "machine" and m.analyzer is not None and m.rpc_id is not None:
                out.append(str(m.rpc_id))
        return out

    async def _run_machine_gc_loop(self) -> None:
        if _MACHINE_GC_INTERVAL_SEC <= 0 or _MACHINE_GC_EXPIRE_SEC <= 0:
            logger.info(
                "[MACHINE_GC] disabled: interval=%ss expire=%ss",
                _MACHINE_GC_INTERVAL_SEC,
                _MACHINE_GC_EXPIRE_SEC,
            )
            return
        logger.info(
            "[MACHINE_GC] loop started: interval=%ss expire=%ss",
            _MACHINE_GC_INTERVAL_SEC,
            _MACHINE_GC_EXPIRE_SEC,
        )
        try:
            while True:
                await asyncio.sleep(_MACHINE_GC_INTERVAL_SEC)

                now = time.time()
                threshold = now - _MACHINE_GC_EXPIRE_SEC
                active_ids = self._active_machine_rpc_ids()

                if not active_ids:
                    logger.info("[MACHINE_GC] no active machine task, skip sweep")
                    continue

                total_deleted = {"evidence_images": 0, "evidence_images_box": 0, "out": 0}

                for rid in active_ids:
                    try:
                        stats = await asyncio.to_thread(
                            cleanup_running_machine_storage_older_than_sync,
                            rid,
                            threshold,
                        )
                        logger.info(
                            "[MACHINE_GC] sweep rpc_id=%s threshold=%s deleted=%s",
                            rid,
                            int(threshold),
                            stats,
                        )
                        total_deleted["evidence_images"] += int(stats.get("evidence_images", 0) or 0)
                        total_deleted["evidence_images_box"] += int(stats.get("evidence_images_box", 0) or 0)
                        total_deleted["out"] += int(stats.get("out", 0) or 0)
                    except Exception as e:
                        logger.warning("[MACHINE_GC] sweep rpc_id=%s failed: %s", rid, e)

                logger.info(
                    "[MACHINE_GC] sweep done: active=%d threshold=%s total_deleted=%s",
                    len(active_ids),
                    int(threshold),
                    total_deleted,
                )

        except asyncio.CancelledError:
            logger.info("[MACHINE_GC] loop cancelled")
            return
        except Exception as e:
            logger.warning("[MACHINE_GC] loop crashed: %s", e)

    async def _ensure_machine_gc_started(self) -> None:
        async with self._machine_gc_lock:
            if self._machine_gc_task and not self._machine_gc_task.done():
                return
            self._machine_gc_task = asyncio.create_task(self._run_machine_gc_loop())

    # ------------------------------------------------------------------
    # stop timer 与 cleanup helpers
    # ------------------------------------------------------------------
    def _cancel_auto_stop(self, meta: Optional[TaskMeta]) -> None:
        if not meta:
            return
        t = meta.stop_task
        meta.stop_task = None
        meta.stop_after_sec = None
        meta.stop_deadline_ts = None
        if t and not t.done():
            try:
                t.cancel()
            except Exception:
                pass

    async def _cleanup_chat_and_unregister_if_still_same(self, key: str, token: str, session_id: str) -> None:
        try:
            meta = self._task_registry.get(key)
            if not meta or meta.run_token != token:
                return

            try:
                if meta.analyzer:
                    meta.analyzer.force_stop("cleanup fallback")
            except Exception:
                pass

            await asyncio.to_thread(cleanup_chat_storage_sync, session_id)
            self._unregister_task(key)
        except Exception as e:
            logger.warning("[CLEANUP][CHAT] cleanup failed: %s", e)

    async def _cleanup_machine_and_unregister_if_still_same(self, key: str, token: str, rpc_id: Any) -> None:
        try:
            meta = self._task_registry.get(key)
            if not meta or meta.run_token != token:
                return

            try:
                if meta.analyzer:
                    meta.analyzer.force_stop("machine stop cleanup")
            except Exception:
                pass

            await asyncio.to_thread(cleanup_machine_storage_sync, rpc_id)
            self._unregister_task(key)
        except Exception as e:
            logger.warning("[CLEANUP][MACHINE] cleanup failed: %s", e)

    def _arm_auto_stop(self, meta: TaskMeta, stop_after_sec: int, reason: str) -> None:
        self._cancel_auto_stop(meta)

        stop_after_sec = int(stop_after_sec or 0)
        if stop_after_sec <= 0:
            return

        meta.stop_after_sec = stop_after_sec
        meta.stop_deadline_ts = time.time() + stop_after_sec
        key = meta.key
        token = meta.run_token
        session_id = meta.session_id or ""

        async def _job() -> None:
            try:
                await asyncio.sleep(stop_after_sec)
                m = self._task_registry.get(key)
                if not m or m.run_token != token or not m.analyzer:
                    return

                logger.info("[Chat][AUTO_STOP] key=%s after=%ss reason=%s", key, stop_after_sec, reason)
                try:
                    m.analyzer.force_stop(reason)
                except Exception as e:
                    logger.warning("[Chat][AUTO_STOP] force_stop error: %s", e)

                asyncio.create_task(self._cleanup_chat_and_unregister_if_still_same(key, token, session_id))

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("[Chat][AUTO_STOP] timer error: %s", e)

        meta.stop_task = asyncio.create_task(_job())

    def _cleanup_task_resources(self, meta: Optional[TaskMeta]) -> None:
        if not meta:
            return
        self._cancel_auto_stop(meta)

    def _register_task(self, meta: TaskMeta) -> None:
        self._task_registry[meta.key] = meta
        self.current_task_id = meta.key
        self.patrol_status = "running"
        logger.info(
            "注册任务: key=%s, kind=%s, mode=%s, active=%d/%d client_id=%s",
            meta.key,
            meta.kind,
            meta.mode,
            self._get_active_task_count(),
            self.max_concurrent,
            meta.client_id,
        )

    def _unregister_task(self, key: str) -> None:
        meta = self._task_registry.pop(key, None)
        if meta:
            self._cleanup_task_resources(meta)
            logger.info("注销任务: key=%s", key)
        if not self._task_registry:
            self.patrol_status = "idle"
            self.current_task_id = None

    # ------------------------------------------------------------------
    # 断链立即 stop 下游任务并清理 + 注销（更强：立即）
    # ------------------------------------------------------------------
    def stop_all_tasks_by_client_id(self, client_id: str, reason: str = "ws disconnected") -> None:
        cid = (client_id or "").strip()
        if not cid:
            return

        metas = [(k, m) for k, m in list(self._task_registry.items()) if m.client_id == cid]
        if not metas:
            return

        logger.warning("[STOP_BY_CLIENT] client_id=%s tasks=%d reason=%s", cid, len(metas), reason)

        try:
            loop = asyncio.get_running_loop()
        except Exception:
            loop = None

        for key, meta in metas:
            # 1) 先停 analyzer
            if meta.analyzer:
                try:
                    meta.analyzer.force_stop(reason)
                except Exception:
                    pass

            # 2) 立即清理 + 立即注销
            if loop and loop.is_running():
                try:
                    if meta.kind == "chat" and meta.session_id:
                        loop.create_task(
                            self._cleanup_chat_and_unregister_if_still_same(key, meta.run_token, meta.session_id)
                        )
                    elif meta.kind == "machine" and meta.rpc_id is not None:
                        loop.create_task(
                            self._cleanup_machine_and_unregister_if_still_same(key, meta.run_token, meta.rpc_id)
                        )
                    else:
                        self._unregister_task(key)
                except Exception:
                    try:
                        self._unregister_task(key)
                    except Exception:
                        pass
            else:
                # 无事件循环：同步清理 + 注销
                try:
                    if meta.kind == "chat" and meta.session_id:
                        cleanup_chat_storage_sync(meta.session_id)
                    elif meta.kind == "machine" and meta.rpc_id is not None:
                        cleanup_machine_storage_sync(meta.rpc_id)
                except Exception:
                    pass
                try:
                    self._unregister_task(key)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 事件回调
    # ------------------------------------------------------------------
    def add_event_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.event_callbacks.append(callback)

    def _emit_event(self, event: Dict[str, Any]) -> None:
        for cb in self.event_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning("event callback error: %s", e)

    # ------------------------------------------------------------------
    # 证据帧 URL 拼接（模板负责）（全部从 .env 读取）
    # ------------------------------------------------------------------
    @staticmethod
    def _is_http_url(s: Any) -> bool:
        return isinstance(s, str) and s.lower().startswith(("http://", "https://"))

    @staticmethod
    def _load_evidence_base_url() -> str:
        raw = (_env_get_optional("VLM_EVIDENCE_IMAGE_BASE_URL") or "").strip()
        if not raw:
            logger.warning("未配置 VLM_EVIDENCE_IMAGE_BASE_URL（.env），将无法把 /storage/... 拼成可访问的 http(s):// 链接")
            return ""
        if not raw.lower().startswith(("http://", "https://")):
            logger.warning("VLM_EVIDENCE_IMAGE_BASE_URL=%r 非法：必须以 http:// 或 https:// 开头。将忽略该配置。", raw)
            return ""
        return raw.rstrip("/")

    @staticmethod
    def _join_base_url(base: str, path: str) -> str:
        base = (base or "").strip().rstrip("/")
        path = (path or "").strip()
        if not base:
            return path
        if not path:
            return base
        if path.startswith(("http://", "https://")):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _absolutize_evidence_urls(self, urls: List[Any]) -> List[str]:
        out: List[str] = []
        for u in urls or []:
            if not u:
                continue
            s = str(u).strip()
            if not s:
                continue
            if self._is_http_url(s):
                out.append(s)
            else:
                out.append(self._join_base_url(self.evidence_base_url, s) if self.evidence_base_url else s)
        return out

    # ------------------------------------------------------------------
    # 前端 Markdown 块：image / video
    # ------------------------------------------------------------------
    @staticmethod
    def _format_image_block(url: str) -> str:
        return f":::image\n```json\n{url}\n```\n:::\n"

    @staticmethod
    def _format_video_block(url: str) -> str:
        return f":::video\n```json\n{url}\n```\n:::\n"

    def _local_path_to_storage_url(self, local_path: str) -> Optional[str]:
        if not local_path:
            return None
        try:
            p = Path(local_path).resolve()
            rel = p.relative_to(_STORAGE_ROOT)
            return "/storage/" + rel.as_posix()
        except Exception:
            return None

    def _extract_done_image_urls(self, ev: Dict[str, Any]) -> List[str]:
        raw: List[Any] = []
        v1 = ev.get("evidence_image_urls") or []
        v2 = ev.get("evidence_image_box_urls") or []
        if isinstance(v1, list):
            raw.extend(v1)
        elif v1:
            raw.append(v1)
        if isinstance(v2, list):
            raw.extend(v2)
        elif v2:
            raw.append(v2)

        seen = set()
        uniq: List[str] = []
        for x in raw:
            if not x:
                continue
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        return self._absolutize_evidence_urls(uniq)

    # ------------------------------------------------------------------
    # envelope 判定
    # ------------------------------------------------------------------
    @staticmethod
    def _is_chat_envelope(original_params: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(original_params, dict):
            return False
        return ("sessionId" in original_params and "content" in original_params)

    @staticmethod
    def _extract_chat_message_and_system_prompt(original_params: Dict[str, Any]) -> Tuple[str, str]:
        content = (original_params.get("content") or "").strip()
        ctx = original_params.get("context") or {}
        system_prompt = (ctx.get("system_prompt") or "").strip()
        return content, system_prompt

    @staticmethod
    def _extract_rpc_id_and_params(original_params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if not isinstance(original_params, dict):
            return None, {}
        if "jsonrpc" in original_params:
            rpc_id = original_params.get("id")
            params = original_params.get("params") or {}
            return rpc_id, params
        return original_params.get("id"), original_params

    # ------------------------------------------------------------------
    # Chat：命令解析
    # ------------------------------------------------------------------
    _URL_PATTERN = re.compile(
        r"""(
            https?://[^\s'"]+
            |rtsp://[^\s'"]+
            |rtsps://[^\s'"]+
            |rtmp://[^\s'"]+
            |rtmps://[^\s'"]+
            |file://[^\s'"]+
            |/[^\s'"]+
            |[A-Za-z]:\\[^\s'"]+
        )""",
        re.IGNORECASE | re.VERBOSE,
    )
    _BACKEND_NAME_RE = re.compile(r"^\s*(?P<backend>cv|yolo|vlm)(?::(?P<model>[^\s]+))?", re.IGNORECASE)

    @staticmethod
    def _normalize_backend_name(backend: Optional[str]) -> Optional[str]:
        if not backend:
            return None
        b = str(backend).strip().lower()
        if b in ("cv", "yolo"):
            return "cv"
        if b in ("vlm", "llm"):
            return "vlm"
        return None

    @classmethod
    def _parse_backend_from_name(cls, name: str) -> Tuple[Optional[str], Optional[str]]:
        if not name:
            return None, None
        m = cls._BACKEND_NAME_RE.match(str(name).strip())
        if not m:
            return None, None
        backend = cls._normalize_backend_name(m.group("backend"))
        model = (m.group("model") or "").strip() or None
        return backend, model

    def _resolve_backend_and_model(
        self,
        params: Dict[str, Any],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        backend_hint = self._normalize_backend_name(
            params.get("backend") or params.get("engine") or params.get("detector")
        )
        model_hint = (
            (params.get("model") or params.get("modelName") or params.get("model_path") or params.get("modelPath"))
            or ""
        ).strip() or None

        name_hint = (params.get("name") or "").strip()
        if not name_hint:
            cfg = params.get("configuration") or {}
            if isinstance(cfg, dict):
                name_hint = (cfg.get("name") or "").strip()
                if not backend_hint:
                    backend_hint = self._normalize_backend_name(
                        cfg.get("backend") or cfg.get("engine") or cfg.get("detector")
                    )
                if not model_hint:
                    model_hint = (cfg.get("model") or cfg.get("modelName") or "").strip() or None

        if name_hint:
            b_from_name, m_from_name = self._parse_backend_from_name(name_hint)
            if not backend_hint and b_from_name:
                backend_hint = b_from_name
            if not model_hint and m_from_name:
                model_hint = m_from_name

        source_backends = set()
        sources = params.get("source") or []
        if isinstance(sources, list):
            for src in sources:
                if not isinstance(src, dict):
                    continue
                sb = self._normalize_backend_name(src.get("backend") or src.get("engine") or src.get("detector"))
                if sb:
                    source_backends.add(sb)
                if not model_hint:
                    sm = (src.get("model") or src.get("modelName") or "").strip() or None
                    if sm:
                        model_hint = sm

                if not backend_hint:
                    sn = (src.get("name") or "").strip()
                    if sn:
                        b_from_name, m_from_name = self._parse_backend_from_name(sn)
                        if b_from_name:
                            source_backends.add(b_from_name)
                        if not model_hint and m_from_name:
                            model_hint = m_from_name

        if backend_hint:
            source_backends.add(backend_hint)

        if len(source_backends) > 1:
            return "vlm", model_hint, "backend conflict in params/source"

        backend = source_backends.pop() if source_backends else "vlm"
        return backend, model_hint, None

    @staticmethod
    def _normalize_cv_task(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        s = str(value).strip().lower()
        if not s:
            return None

        # explicit canonical values
        if s in {"fall", "fallen", "falling"}:
            return "fall"
        if s in {"smoke", "smoking", "cigarette"}:
            return "smoke"
        if s in {"fight", "fighting", "brawl"}:
            return "fight"

        # keyword match (zh/en)
        fall_keywords = ("跌倒", "摔倒", "倒地", "fall")
        smoke_keywords = ("抽烟", "吸烟", "smoke", "smoking", "cigarette")
        fight_keywords = ("吵架", "打架", "斗殴", "fight", "fighting", "brawl")

        if any(k in s for k in fall_keywords):
            return "fall"
        if any(k in s for k in smoke_keywords):
            return "smoke"
        if any(k in s for k in fight_keywords):
            return "fight"

        return None

    def _resolve_cv_task(self, params: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        tasks = set()

        def _add_task(raw: Any) -> None:
            t = self._normalize_cv_task(raw)
            if t:
                tasks.add(t)

        # explicit task fields
        _add_task(params.get("cvTask"))
        _add_task(params.get("task"))
        _add_task(params.get("taskType"))
        _add_task(params.get("action"))
        _add_task(params.get("scene"))

        name_hint = params.get("name")
        _add_task(name_hint)

        cfg = params.get("configuration") or {}
        if isinstance(cfg, dict):
            _add_task(cfg.get("name"))
            _add_task(cfg.get("task"))
            for key in ("cv", "cvConfig", "cv_config", "vision", "yolo"):
                sub = cfg.get(key)
                if isinstance(sub, dict):
                    _add_task(sub.get("task"))
                    _add_task(sub.get("name"))

        for key in ("cv", "cvConfig", "cv_config", "vision", "yolo"):
            sub = params.get(key)
            if isinstance(sub, dict):
                _add_task(sub.get("task"))
                _add_task(sub.get("name"))

        sources = params.get("source") or []
        if isinstance(sources, list):
            for src in sources:
                if not isinstance(src, dict):
                    continue
                _add_task(src.get("name"))
                _add_task(src.get("task"))

        if len(tasks) > 1:
            return None, "cv task conflict in params/name/source"
        if tasks:
            return tasks.pop(), None
        return None, None

    @staticmethod
    def _coerce_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off", ""):
                return False
        return default

    @staticmethod
    def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        if value is None:
            return default
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _coerce_int_list(value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            out = []
            for v in value:
                try:
                    out.append(int(float(v)))
                except Exception:
                    continue
            return out or None
        if isinstance(value, str):
            parts = value.replace(";", ",").split(",")
            out = []
            for part in parts:
                p = part.strip()
                if not p:
                    continue
                try:
                    out.append(int(float(p)))
                except Exception:
                    continue
            return out or None
        return None

    @classmethod
    def _parse_roi_config(cls, raw: Any) -> Optional[CvRoiConfig]:
        if raw is None:
            return None

        rect = None
        normalized = None
        mode = None
        draw = None
        padding = None

        if isinstance(raw, dict):
            normalized = raw.get("normalized")
            rect = raw.get("rect")
            if rect is None and all(k in raw for k in ("x1", "y1", "x2", "y2")):
                rect = [raw.get("x1"), raw.get("y1"), raw.get("x2"), raw.get("y2")]
            if rect is None and all(k in raw for k in ("x", "y", "w", "h")):
                x = raw.get("x")
                y = raw.get("y")
                w = raw.get("w")
                h = raw.get("h")
                rect = [x, y, None if w is None else x + w, None if h is None else y + h]
            if rect is None and all(k in raw for k in ("left", "top", "right", "bottom")):
                rect = [raw.get("left"), raw.get("top"), raw.get("right"), raw.get("bottom")]
            mode = raw.get("mode") or raw.get("roiMode") or raw.get("roi_mode")
            draw = raw.get("draw") or raw.get("drawRoi") or raw.get("roiDraw")
            padding = raw.get("padding") or raw.get("pad")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
            rect = raw

        if not rect:
            return None

        try:
            x1 = float(rect[0])
            y1 = float(rect[1])
            x2 = float(rect[2])
            y2 = float(rect[3])
        except Exception:
            return None

        if normalized is None:
            vals = [x1, y1, x2, y2]
            normalized = min(vals) >= 0.0 and max(vals) <= 1.0
        else:
            normalized = cls._coerce_bool(normalized, default=True)

        mode_str = (mode or "center").strip().lower()
        draw_flag = bool(cls._coerce_bool(draw, default=False))
        pad_val = float(padding) if padding is not None else 0.0

        return CvRoiConfig(
            rect=(x1, y1, x2, y2),
            normalized=bool(normalized),
            mode=mode_str or "center",
            draw=draw_flag,
            padding=max(0.0, pad_val),
        )

    @staticmethod
    def _looks_like_cv_config(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        keys = {
            "conf",
            "confidence",
            "score",
            "iou",
            "imgsz",
            "max_det",
            "maxDet",
            "max_boxes",
            "maxBoxes",
            "device",
            "classes",
            "class_ids",
            "roi",
            "roiRect",
            "roiMode",
            "roi_mode",
            "pose",
            "poseConfig",
            "rule",
            "ruleConfig",
        }
        return any(k in obj for k in keys)

    def _apply_pose_overrides(self, pose_cfg: CvPoseConfig, pose_raw: Any) -> bool:
        if not isinstance(pose_raw, dict):
            return False
        touched = False

        def _set_float(attr: str, raw_key: str) -> None:
            nonlocal touched
            if raw_key not in pose_raw:
                return
            val = self._coerce_float(pose_raw.get(raw_key))
            if val is None:
                return
            setattr(pose_cfg, attr, float(val))
            touched = True

        _set_float("min_kpt_conf", "minKptConf")
        _set_float("min_kpt_conf", "min_kpt_conf")
        _set_float("fall_ratio", "fallRatio")
        _set_float("fall_ratio", "fall_ratio")
        _set_float("fall_angle_ratio", "fallAngleRatio")
        _set_float("fall_angle_ratio", "fall_angle_ratio")
        _set_float("fall_low_y", "fallLowY")
        _set_float("fall_low_y", "fall_low_y")
        _set_float("smoke_dist_ratio", "smokeDistRatio")
        _set_float("smoke_dist_ratio", "smoke_dist_ratio")
        _set_float("smoke_wrist_y_margin", "smokeWristYMargin")
        _set_float("smoke_wrist_y_margin", "smoke_wrist_y_margin")
        _set_float("fight_center_ratio", "fightCenterRatio")
        _set_float("fight_center_ratio", "fight_center_ratio")
        _set_float("fight_hand_ratio", "fightHandRatio")
        _set_float("fight_hand_ratio", "fight_hand_ratio")
        _set_float("fight_min_score", "fightMinScore")
        _set_float("fight_min_score", "fight_min_score")

        return touched

    def _apply_cv_config_dict(self, cv_cfg: CvConfig, cfg: Any, *, flags: Dict[str, bool]) -> None:
        if not isinstance(cfg, dict):
            return

        if "conf" in cfg or "confidence" in cfg or "score" in cfg:
            val = self._coerce_float(cfg.get("conf") or cfg.get("confidence") or cfg.get("score"))
            if val is not None:
                cv_cfg.conf = float(val)
                flags["conf"] = True

        if "iou" in cfg:
            val = self._coerce_float(cfg.get("iou"))
            if val is not None:
                cv_cfg.iou = float(val)

        if "imgsz" in cfg:
            val = self._coerce_int(cfg.get("imgsz"))
            if val is not None:
                cv_cfg.imgsz = int(val)

        if "max_det" in cfg or "maxDet" in cfg:
            val = self._coerce_int(cfg.get("max_det") or cfg.get("maxDet"))
            if val is not None:
                cv_cfg.max_det = int(val)

        if "max_boxes" in cfg or "maxBoxes" in cfg:
            val = self._coerce_int(cfg.get("max_boxes") or cfg.get("maxBoxes"))
            if val is not None:
                cv_cfg.max_boxes = int(val)

        if "device" in cfg:
            val = str(cfg.get("device") or "").strip()
            if val:
                cv_cfg.device = val

        if "classes" in cfg or "class_ids" in cfg:
            cls_list = self._coerce_int_list(cfg.get("classes") or cfg.get("class_ids"))
            if cls_list is not None:
                cv_cfg.classes = cls_list

        roi_raw = cfg.get("roi") or cfg.get("roiRect")
        roi_cfg = self._parse_roi_config(roi_raw)
        if roi_cfg:
            cv_cfg.roi = roi_cfg

        pose_raw = (
            cfg.get("pose")
            or cfg.get("poseConfig")
            or cfg.get("rule")
            or cfg.get("ruleConfig")
        )
        if pose_raw:
            self._apply_pose_overrides(cv_cfg.pose, pose_raw)

    def _apply_cv_overrides(self, cv_cfg: CvConfig, params: Dict[str, Any]) -> Dict[str, bool]:
        flags = {"conf": False}

        # 1) top-level params
        self._apply_cv_config_dict(cv_cfg, params, flags=flags)

        # 2) configuration block
        cfg = params.get("configuration")
        if isinstance(cfg, dict):
            self._apply_cv_config_dict(cv_cfg, cfg, flags=flags)
            for key in ("cv", "cvConfig", "cv_config", "vision", "yolo"):
                if isinstance(cfg.get(key), dict):
                    self._apply_cv_config_dict(cv_cfg, cfg.get(key), flags=flags)

        # 3) explicit cv config on params
        for key in ("cv", "cvConfig", "cv_config", "vision", "yolo"):
            if isinstance(params.get(key), dict):
                self._apply_cv_config_dict(cv_cfg, params.get(key), flags=flags)

        # 4) roi from sources (by id)
        sources = params.get("source") or []
        if isinstance(sources, list):
            for src in sources:
                if not isinstance(src, dict):
                    continue
                src_id = src.get("id")
                if src_id is None:
                    continue
                roi_raw = src.get("roi") or src.get("roiRect")
                cfg = src.get("configuration") or {}
                if not roi_raw and isinstance(cfg, dict):
                    roi_raw = cfg.get("roi") or cfg.get("roiRect")
                roi_cfg = self._parse_roi_config(roi_raw)
                if roi_cfg:
                    cv_cfg.roi_by_id[str(src_id)] = roi_cfg

        return flags

    @staticmethod
    def _pick_cv_roi(cv_cfg: Optional[CvConfig], source_id: Any) -> Optional[CvRoiConfig]:
        if not cv_cfg:
            return None
        if source_id is not None:
            key = str(source_id)
            roi = cv_cfg.roi_by_id.get(key)
            if roi:
                return roi
        return cv_cfg.roi

    @staticmethod
    def _parse_interval_seconds(text: str) -> Optional[int]:
        m = re.search(r"(?:轮询间隔|轮询|间隔)\s*[-:：=]?\s*(\d+)\s*(?:秒|s|S)?", text)
        if m:
            try:
                v = int(m.group(1))
                return v if v > 0 else None
            except Exception:
                return None

        m = re.search(r"(?:每)\s*(\d+)\s*秒", text)
        if m:
            try:
                v = int(m.group(1))
                return v if v > 0 else None
            except Exception:
                return None

        m = re.search(r"(\d+)\s*秒", text)
        if m:
            try:
                v = int(m.group(1))
                return v if v > 0 else None
            except Exception:
                return None

        return None

    @staticmethod
    def _parse_stop_minutes(text: str) -> Optional[int]:
        m = re.search(r"(?:停止时间)\s*[-:：=]?\s*(\d+)\s*(?:分|分钟|min|m)?", text, re.IGNORECASE)
        if m:
            try:
                v = int(m.group(1))
                return v if v > 0 else None
            except Exception:
                return None

        m = re.search(r"(?:停止)\s*[-:：=]?\s*(\d+)\s*(?:分|分钟|min|m)?", text, re.IGNORECASE)
        if m:
            try:
                v = int(m.group(1))
                return v if v > 0 else None
            except Exception:
                return None

        return None

    @classmethod
    def _parse_chat_command(cls, content: str) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
        text = content.strip()

        cmd_type = "unknown"
        if any(k in text for k in ("启动分析", "开始分析", "启动巡检", "开始巡检")):
            cmd_type = "start"
        elif any(k in text for k in ("停止分析", "停止巡检", "终止分析", "结束分析", "结束巡检")):
            cmd_type = "stop"
        elif any(k in text for k in ("查看状态", "查看进度", "当前状态", "当前进度", "任务状态")):
            cmd_type = "status"

        url = None
        m = cls._URL_PATTERN.search(text)
        if m:
            url = m.group(1)

        interval = cls._parse_interval_seconds(text)
        stop_minutes = cls._parse_stop_minutes(text)

        logger.info(
            "[Chat] 解析命令: cmd_type=%s, url=%s, interval=%s, stop_minutes=%s, raw=%r",
            cmd_type,
            url,
            interval,
            stop_minutes,
            content,
        )
        return cmd_type, url, interval, stop_minutes

    # ------------------------------------------------------------------
    # Chat：构造 StreamingAnalyze（必须传 task_id=sessionId）
    # ------------------------------------------------------------------
    def _build_vlm_config_for_chat_offline(self, system_prompt: str) -> VlmConfig:
        vlm_cfg = VlmConfig()
        vlm_cfg.vlm_streaming = True
        vlm_cfg.is_json_format = False
        vlm_cfg.vlm_temperature = 0.5
        if hasattr(vlm_cfg, "offline_system_prompt"):
            vlm_cfg.offline_system_prompt = system_prompt or ""
        return vlm_cfg

    def _build_vlm_config_for_chat_security_single(self, system_prompt: str) -> VlmConfig:
        vlm_cfg = VlmConfig()
        vlm_cfg.vlm_streaming = True
        vlm_cfg.is_json_format = False
        vlm_cfg.vlm_temperature = 0.5
        return vlm_cfg

    def _create_chat_offline_analyzer(self, url: str, system_prompt: str, *, task_id: str) -> StreamingAnalyze:
        vlm_cfg = self._build_vlm_config_for_chat_offline(system_prompt)
        analyzer = StreamingAnalyze(
            mode=MODEL.OFFLINE,
            task_id=task_id,
            url=url,
            enable_b=True,
            enable_c=False,
            vlm_config=vlm_cfg,
        )
        analyzer.on_vlm_delta = _noop_stream_delta
        analyzer.on_asr_delta = _noop_stream_delta
        return analyzer

    def _create_chat_security_single_analyzer(
        self, rtsp_url: str, system_prompt: str, interval_sec: Optional[int] = None, *, task_id: str
    ) -> StreamingAnalyze:
        rtsp = RTSP(rtsp_id=None, rtsp_url=rtsp_url, rtsp_system_prompt=system_prompt or "")

        if interval_sec and interval_sec > 0:
            batch_cfg = RTSPBatchConfig(polling_list=[rtsp], polling_batch_interval=float(interval_sec))
        else:
            batch_cfg = RTSPBatchConfig(polling_list=[rtsp])

        vlm_cfg = self._build_vlm_config_for_chat_security_single(system_prompt)
        analyzer = StreamingAnalyze(
            mode=MODEL.SECURITY_SINGLE,
            task_id=task_id,
            enable_b=True,
            enable_c=False,
            rtsp_batch_config=batch_cfg,
            vlm_config=vlm_cfg,
        )
        analyzer.on_vlm_delta = _noop_stream_delta
        analyzer.on_asr_delta = _noop_stream_delta
        return analyzer

    # ------------------------------------------------------------------
    # Chat：状态描述
    # ------------------------------------------------------------------
    def _describe_chat_status(self, key: str) -> List[str]:
        meta = self._task_registry.get(key)
        if not meta or not meta.analyzer:
            return ["当前会话没有正在运行的分析或巡检任务。"]

        lines = ["📊 当前会话分析状态："]
        lines.append(f"  - 模式: {meta.mode}")
        if meta.url:
            lines.append(f"  - 视频地址: {meta.url}")
        if meta.local_url:
            lines.append(f"  - 本地缓存: {meta.local_url}")
        dur = int(time.time() - meta.started_at)
        lines.append(f"  - 已运行时间: {dur} 秒（近似）")
        if meta.stop_deadline_ts:
            remain = max(0, int(meta.stop_deadline_ts - time.time()))
            lines.append(f"  - 自动停止倒计时: {remain} 秒（约 {remain//60} 分钟）")
        return lines

    # ------------------------------------------------------------------
    # Chat：主流程（流式输出）
    # ------------------------------------------------------------------
    async def _handle_chat_request(self, original_params: Dict[str, Any]) -> AsyncIterator[str]:
        session_id = str((original_params.get("sessionId") or "")).strip()
        if not session_id:
            yield "❌ 缺少 sessionId，无法创建 task_id 目录，请检查上游入参。\n"
            return

        # websocket client_id（断链 stop/cleanup 用）
        ws_client_id = None
        try:
            ws_client_id = (original_params.get("_ws_client_id") or "").strip() or None
        except Exception:
            ws_client_id = None

        content, system_prompt = self._extract_chat_message_and_system_prompt(original_params)

        cmd_type, url, interval, stop_minutes = self._parse_chat_command(content)
        key = self._make_chat_task_key(session_id)

        if cmd_type == "status":
            for line in self._describe_chat_status(key):
                yield line
            return

        if cmd_type == "stop":
            meta = self._task_registry.get(key)
            if not meta or not meta.analyzer:
                yield "当前会话没有正在运行的分析或巡检任务。\n请先发送：启动分析 + URL\n"
                return

            try:
                meta.analyzer.force_stop("chat stop command")
            except Exception as e:
                logger.warning("[Chat] force_stop 异常: %s", e)

            # ✅ 立即清理 + 注销
            asyncio.create_task(self._cleanup_chat_and_unregister_if_still_same(key, meta.run_token, session_id))

            yield "✅ 已请求停止当前会话的分析/巡检任务（后台完成收尾与清理）。\n"
            return

        if cmd_type != "start":
            yield (
                "未能识别您的指令，请使用以下格式之一：\n"
                "- 启动巡检 rtsp://... 轮询间隔=30秒 停止时间=5分钟\n"
                "- 停止分析/巡检\n"
                "- 查看状态/进度\n"
            )
            return

        if not url:
            yield (
                "未检测到有效的视频 URL/路径，请在命令中包含：\n"
                "- rtsp(s):// / rtmp(s)://\n"
                "- http(s)://\n"
                "- file://\n"
                "- 本地绝对路径（/xxx 或 C:\\xxx）\n"
                "\n格式示例：\n"
                "启动巡检 rtsp://... 轮询间隔-30 停止时间-10\n"
            )
            return

        active_count = self._get_active_task_count()
        if active_count >= self.max_concurrent and key not in self._task_registry:
            yield "当前任务繁忙，暂无法启动新的分析任务，请稍后重试（建议指数退避）。\n"
            return

        old_meta = self._task_registry.get(key)
        if old_meta and old_meta.analyzer:
            try:
                old_meta.analyzer.force_stop("new task in same chat session")
            except Exception:
                pass
            asyncio.create_task(self._cleanup_chat_and_unregister_if_still_same(key, old_meta.run_token, session_id))
            yield "检测到当前会话已有任务，已请求停止旧任务，并启动新的分析任务。\n"

        url_lower = url.lower()
        is_live = url_lower.startswith(("rtsp://", "rtsps://", "rtmp://", "rtmps://"))

        http_tmp_dir: Optional[str] = None
        local_url: Optional[str] = None
        display_video_http_url: Optional[str] = None

        if is_live:
            analyzer = self._create_chat_security_single_analyzer(
                url, system_prompt, interval_sec=interval, task_id=session_id
            )
            mode_label = "实时安全巡检 (SECURITY_SINGLE)"
            display_video_http_url = url
        else:
            effective_url = url
            if self._is_http_url(url):
                yield "🔽 检测到离线 HTTP/HTTPS 视频，正在下载到本地临时目录...\n"
                try:
                    local_path, tmp_dir = await download_http_offline_to_tmp(url, session_id)
                    effective_url = local_path
                    local_url = local_path
                    http_tmp_dir = tmp_dir
                    yield "✅ 下载完成\n"

                    storage_path = self._local_path_to_storage_url(local_path)
                    if storage_path:
                        display_video_http_url = (
                            self._join_base_url(self.evidence_base_url, storage_path)
                            if self.evidence_base_url
                            else storage_path
                        )
                except Exception as e:
                    logger.exception("[HTTP_OFFLINE] download failed: %s", e)
                    yield f"❌ 离线资源下载失败：{e}\n"
                    return
            else:
                storage_path = self._local_path_to_storage_url(url)
                if storage_path:
                    display_video_http_url = (
                        self._join_base_url(self.evidence_base_url, storage_path)
                        if self.evidence_base_url
                        else storage_path
                    )

            analyzer = self._create_chat_offline_analyzer(effective_url, system_prompt, task_id=session_id)
            mode_label = "离线视频分析 (OFFLINE)"

        auto_stop_sec: Optional[int] = None
        if is_live:
            auto_stop_sec = (int(stop_minutes) * 60) if stop_minutes else int(self.chat_security_single_stop_time)
        else:
            auto_stop_sec = (int(stop_minutes) * 60) if stop_minutes else None

        meta = TaskMeta(
            key=key,
            kind="chat",
            mode=("SECURITY_SINGLE" if is_live else "OFFLINE"),
            started_at=time.time(),
            session_id=session_id,
            url=url,
            local_url=local_url,
            http_tmp_dir=http_tmp_dir,
            analyzer=analyzer,
            client_id=ws_client_id,  # ✅ 绑定 client_id
        )
        self._register_task(meta)

        yield f"✅ 已接收任务，类型：{mode_label}\n"
        yield f"📺 视频地址：{url}\n"
        if interval and is_live:
            yield f"⏱ 轮询间隔：{interval} 秒\n"

        if auto_stop_sec and auto_stop_sec > 0:
            hint = "用户指定停止时间" if stop_minutes else "使用默认停止时间"
            self._arm_auto_stop(meta, auto_stop_sec, reason=f"chat auto stop ({hint})")
            yield f"🛑 自动停止：{auto_stop_sec//60} 分钟后（也可手动发送“停止巡检/停止分析”提前结束）\n"
        else:
            if not is_live:
                yield "🛑 自动停止：未设置，离线任务将根据视频长度自动结束。\n"

        if display_video_http_url:
            yield self._format_video_block(display_video_http_url)

        try:
            # ✅ flush 参数从 .env 读取
            flush_chars = _env_get_int("VLM_STREAM_FLUSH_CHARS", 64)
            flush_interval_s = _env_get_float("VLM_STREAM_FLUSH_INTERVAL_S", 0.2)
            if flush_chars <= 0:
                flush_chars = 64
            if flush_interval_s <= 0:
                flush_interval_s = 0.2

            delta_buf: List[str] = []
            delta_buf_len = 0
            last_flush_ts = time.time()

            def _flush_delta_buf(now: Optional[float] = None) -> Optional[str]:
                nonlocal delta_buf_len, last_flush_ts
                if delta_buf_len <= 0:
                    return None
                out = "".join(delta_buf)
                delta_buf.clear()
                delta_buf_len = 0
                last_flush_ts = now if now is not None else time.time()
                return out

            async for ev in _iterate_analyzer_events_async(analyzer):
                ev_type = ev.get("type")

                if ev_type == "vlm_stream_delta":
                    delta = ev.get("delta") or ""
                    if delta:
                        delta_buf.append(delta)
                        delta_buf_len += len(delta)
                        now = time.time()
                        if delta_buf_len >= flush_chars or (now - last_flush_ts) >= flush_interval_s:
                            flushed = _flush_delta_buf(now)
                            if flushed:
                                yield flushed

                elif ev_type == "vlm_stream_done":
                    flushed = _flush_delta_buf()
                    if flushed:
                        yield flushed

                    img_urls = self._extract_done_image_urls(ev)

                    seg_idx = ev.get("segment_index")
                    t0 = ev.get("clip_t0")
                    t1 = ev.get("clip_t1")
                    header = "📷 本轮证据帧："
                    if seg_idx is not None:
                        header += f" segment={seg_idx}"
                    if t0 is not None and t1 is not None:
                        header += f" window=[{t0},{t1}]"
                    yield "\n---\n" + header + "\n"

                    if img_urls:
                        for u in img_urls[:12]:
                            yield self._format_image_block(u)
                    else:
                        yield "（本轮未产出证据帧）\n"

                elif ev_type == "error":
                    flushed = _flush_delta_buf()
                    if flushed:
                        yield flushed
                    msg = ev.get("message") or "分析过程中发生未知错误\n"
                    yield f"❌ 分析错误：{msg}\n"

        finally:
            # ✅ finally 兜底清理
            try:
                await asyncio.to_thread(cleanup_chat_storage_sync, session_id)
            except Exception as e:
                logger.warning("[CLEANUP][CHAT] cleanup failed: %s", e)

            self._unregister_task(key)

            try:
                analyzer.force_stop("chat task finished")
            except Exception:
                pass

            yield "\n🔚 分析任务已结束\n"

    # ------------------------------------------------------------------
    # Machine：VLM config
    # ------------------------------------------------------------------
    def _build_vlm_config_for_machine(self, backend: str) -> Any:
        if backend == "cv":
            return SimpleNamespace(
                vlm_streaming=False,
                is_json_format=True,
                vlm_temperature=0.0,
                vlm_max_frames=8,
                vlm_static_evidence_images_dir=str(_EVIDENCE_IMAGES_ROOT),
                vlm_static_evidence_images_url_prefix="/storage/evidence_images",
            )

        vlm_cfg = VlmConfig()
        vlm_cfg.vlm_streaming = False
        vlm_cfg.is_json_format = True
        vlm_cfg.vlm_temperature = 0.2
        return vlm_cfg

    # ------------------------------------------------------------------
    # Machine：StreamingAnalyze（必须传 task_id=rpc_id）
    # ------------------------------------------------------------------
    def _create_machine_analyzer(
        self,
        mode: MODEL,
        sources: List[Dict[str, Any]],
        interval: int,
        *,
        task_id: str,
        backend: str = "vlm",
        cv_config: Optional[CvConfig] = None,
    ) -> StreamingAnalyze:
        rtsp_list: List[RTSP] = []
        for src in sources:
            raw_id = src.get("id", None)
            if raw_id is None:
                rtsp_id: Optional[Union[int, str]] = None
            elif isinstance(raw_id, (int, str)):
                rtsp_id = raw_id
            else:
                raise ValueError(f"source.id 必须为 int | str | None，当前类型={type(raw_id).__name__}, 值={raw_id!r}")

            rtsp_url = src.get("rtsp") or src.get("rtmp")
            if not rtsp_url:
                logger.warning("[Machine] source 缺少 rtsp/rtmp，跳过: %r", src)
                continue

            prompt = (src.get("prompt") or "")
            if backend == "vlm":
                prompt += VLM_OBJECT_DETECT_JSON_CONSTRAINTS_8B_Q8
                # prompt = (src.get("prompt") or "") + VLM_OBJECT_DETECT_JSON_CONSTRAINTS_8B_Q4
            rtsp_list.append(RTSP(rtsp_id=rtsp_id, rtsp_url=rtsp_url, rtsp_system_prompt=prompt))

        if not rtsp_list:
            raise ValueError("source 列表中未找到有效的 rtsp/rtmp 地址")

        if len(rtsp_list) >= 2 and interval < 10:
            interval = 10

        batch_cfg = RTSPBatchConfig(polling_list=rtsp_list, polling_batch_interval=float(interval))
        vlm_cfg = self._build_vlm_config_for_machine(backend)
        runtime_machine_cfg = RuntimeMachineConfig()

        return StreamingAnalyze(
            mode=mode,
            task_id=task_id,
            enable_b=True,
            enable_c=False,
            rtsp_batch_config=batch_cfg,
            vlm_config=vlm_cfg,
            cv_config=cv_config,
            runtime_machine_config=runtime_machine_cfg,
            b_backend=backend,
        )

    # ------------------------------------------------------------------
    # Machine：full_text 宽松解析 + 统一转换为 objects[]
    # ------------------------------------------------------------------
    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

    @staticmethod
    def _clean_text_for_json(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
        # 全角括号兼容
        s = s.translate(
            str.maketrans({"｛": "{", "｝": "}", "［": "[", "］": "]", "﹛": "{", "﹜": "}", "【": "[", "】": "]"})
        )
        return s.strip()

    @classmethod
    def _strip_code_fence(cls, s: str) -> str:
        """如果有 ```json ... ```，优先取 fence 内部；否则原样返回。"""
        s = cls._clean_text_for_json(s)
        if not s:
            return s
        m = cls._JSON_FENCE_RE.search(s)
        if m:
            return (m.group(1) or "").strip()
        return s

    @classmethod
    def _repair_common_json_mistakes(cls, s: str) -> str:
        """
        尝试修复常见“几乎是 JSON，但差一点点”的情况（不保证覆盖所有错误）：
        - 代码块 ```json ... ```
        - 尾逗号：,] 或 ,}
        - box 写成 "box":1,2,3,4]（缺 '['）  -> "box":[1,2,3,4]
        - 单引号 JSON / True False None
        """
        s = cls._strip_code_fence(s)
        s = cls._clean_text_for_json(s)
        if not s:
            return s

        # 1) 去尾逗号
        s = re.sub(r",(\s*[\]}])", r"\1", s)

        # 2) 修复 box: 缺 '[' 但后面带了 ']' 的情况，例如："box":686,440,799,550]
        s = re.sub(
            r'("box"\s*:\s*)(?!\s*\[)\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(\s*\])',
            r'\1[\2,\3,\4,\5]\6',
            s,
        )

        # 3) 修复 box: 缺 '[' 且也没有 ']' 的情况
        s = re.sub(
            r'("box"\s*:\s*)(?!\s*\[)\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?!\s*\])',
            r'\1[\2,\3,\4,\5]',
            s,
        )

        # 4) True/False/None -> true/false/null
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)

        # 5) 单引号 -> 双引号（尽量宽松）
        if "'" in s and '"' not in s:
            s = s.replace("'", '"')

        return s

    @staticmethod
    def _extract_json_fragment(s: str) -> Optional[str]:
        """
        宽松提取字符串中的第一个 JSON 片段（对象或数组）
        - 支持从任意位置开始的 {...} 或 [...]
        - 支持字符串引号、转义，避免误判括号
        """
        if not s:
            return None

        s = VideoPatrolAgent._clean_text_for_json(s)

        # fence 优先
        m = VideoPatrolAgent._JSON_FENCE_RE.search(s)
        if m:
            inner = (m.group(1) or "").strip()
            if inner:
                s = inner

        i_obj = s.find("{")
        i_arr = s.find("[")
        candidates = [i for i in (i_obj, i_arr) if i >= 0]
        if not candidates:
            return None
        start = min(candidates)

        stack: List[str] = []
        in_str = False
        esc = False

        def _match(open_ch: str, close_ch: str) -> bool:
            return (open_ch == "{" and close_ch == "}") or (open_ch == "[" and close_ch == "]")

        for idx in range(start, len(s)):
            ch = s[idx]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch in "{[":
                stack.append(ch)
                continue

            if ch in "}]":
                if not stack:
                    continue
                top = stack[-1]
                if _match(top, ch):
                    stack.pop()
                    if not stack:
                        return s[start : idx + 1]
                else:
                    continue

        return None

    @classmethod
    def _try_load_json(cls, x: Any) -> Any:
        """
        宽松 JSON 解析（适配 B 侧“原始输出透传”）：
        - x 是 dict/list：直接返回
        - x 是 str：
          1) 直接 json.loads
          2) 修复常见错误后 json.loads
          3) 抽取第一个 JSON 片段后 json.loads（含修复）
          4) ast.literal_eval 兜底（用于单引号/类 Python 结构）
        """
        if x is None:
            return None
        if isinstance(x, (list, dict)):
            return x
        if isinstance(x, str):
            s0 = x.strip()
            if not s0:
                return None

            # ① 直接 loads
            try:
                return json.loads(s0)
            except Exception:
                pass

            # ② 修复后 loads
            s1 = cls._repair_common_json_mistakes(s0)
            if s1 and s1 != s0:
                try:
                    return json.loads(s1)
                except Exception:
                    pass

            # ③ 提取 fragment 后 loads
            frag = cls._extract_json_fragment(s0)
            if frag:
                frag1 = cls._repair_common_json_mistakes(frag)
                try:
                    return json.loads(frag1)
                except Exception:
                    pass

            # ④ literal_eval 兜底
            try:
                py = (s1 or s0).replace("true", "True").replace("false", "False").replace("null", "None")
                return ast.literal_eval(py)
            except Exception:
                return None
        return None

    @staticmethod
    def _coerce_box_to_float4(box: Any) -> Optional[List[float]]:
        if not isinstance(box, list) or len(box) != 4:
            return None
        try:
            return [float(v) for v in box]
        except Exception:
            return None

    def _coerce_to_objects_list(self, full_text_raw: Any) -> List[Dict[str, Any]]:
        """把模型输出统一转换为 objects[]（list[dict]），并过滤掉不含 box 的项。"""
        parsed = self._try_load_json(full_text_raw)
        out: List[Dict[str, Any]] = []

        def _append_obj(o: Dict[str, Any]) -> None:
            box4 = self._coerce_box_to_float4(o.get("box"))
            if not box4:
                return
            label = str(o.get("label") or "").strip() or "object"
            conf = o.get("confidence", 0.0)
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            conf_f = max(0.0, min(1.0, conf_f))
            out.append({"label": label, "confidence": round(conf_f, 2), "box": box4})

        def _extend_from_list(lst: Any) -> None:
            if not isinstance(lst, list):
                return
            for o in lst:
                if isinstance(o, dict):
                    if "objects" in o and isinstance(o.get("objects"), list):
                        for oo in o.get("objects") or []:
                            if isinstance(oo, dict):
                                _append_obj(oo)
                    else:
                        _append_obj(o)

        if isinstance(parsed, list):
            _extend_from_list(parsed)
            return out

        if isinstance(parsed, dict):
            if "objects" in parsed:
                _extend_from_list(parsed.get("objects"))
                return out
            if "box" in parsed:
                _append_obj(parsed)
                return out

        return out

    # ------------------------------------------------------------------
    # Machine：构造返回
    # ------------------------------------------------------------------
    def _build_machine_success_payload(
        self,
        source_id: Any,
        evidence_image_urls: List[str],
        evidence_image_box_urls: List[str],
        full_text: Any,
    ) -> Dict[str, Any]:
        return {
            "params": {
                "sourceId": source_id,
                "evidence_image_urls": evidence_image_urls,
                "evidence_image_box_urls": evidence_image_box_urls,
                "full_text": full_text,
            }
        }

    # ------------------------------------------------------------------
    # Machine：主流程（流式输出 JSON 字符串）
    # ------------------------------------------------------------------
    async def _handle_machine_request(
        self,
        rpc_id: Any,
        params: Dict[str, Any],
        ws_client_id: Optional[str],
    ) -> AsyncIterator[str]:
        if rpc_id is None:
            yield json.dumps({"error": {"message": "缺少 jsonrpc.id，无法作为 task_id 进行目录管理。"}}, ensure_ascii=False)
            return

        state = (params.get("state") or "").lower()
        key = self._make_machine_task_key(rpc_id)

        # ✅ stop：立即清理 + 注销（并且 finally 再兜底清理一次）
        if state == "stop":
            meta = self._task_registry.get(key)
            if not meta or not meta.analyzer:
                yield json.dumps({"params": {"message": "未找到运行中的巡检任务（可能已停止/已清理）。"}}, ensure_ascii=False)
                return

            try:
                meta.analyzer.force_stop("machine stop command")
            except Exception as e:
                logger.warning("[Machine] force_stop 异常: %s", e)

            try:
                await asyncio.to_thread(cleanup_machine_storage_sync, rpc_id)
            except Exception as e:
                logger.warning("[CLEANUP][MACHINE] stop cleanup failed: %s", e)

            self._unregister_task(key)
            yield json.dumps({"params": {"message": "巡检任务已停止并清理完成。"}}, ensure_ascii=False)
            return

        if state != "start":
            yield json.dumps({"error": {"message": f"不支持的 state 值: {state!r}"}}, ensure_ascii=False)
            return

        # ✅ Machine：启动防爆盘 GC（只启动一次）
        await self._ensure_machine_gc_started()

        existing_meta = self._task_registry.get(key)
        if existing_meta and existing_meta.analyzer:
            yield json.dumps({"error": {"message": "该响应 ID 已有正在运行的巡检任务，请先 state=stop 停止后再重新启动。"}}, ensure_ascii=False)
            return

        if self._get_active_task_count() >= self.max_concurrent:
            yield json.dumps({"error": {"message": "当前任务繁忙，建议指数退避后再次请求。"}}, ensure_ascii=False)
            return

        backend, model_hint, backend_error = self._resolve_backend_and_model(params)
        if backend_error:
            yield json.dumps({"error": {"message": backend_error}}, ensure_ascii=False)
            return
        cv_task, cv_task_error = self._resolve_cv_task(params)
        if cv_task_error:
            yield json.dumps({"error": {"message": cv_task_error}}, ensure_ascii=False)
            return

        if backend == "cv" and cv_task and not model_hint:
            model_hint = "yolov8n-pose.pt"

        cv_cfg = build_cv_config(model_hint) if backend == "cv" else None
        cv_flags = {}
        if cv_cfg:
            cv_flags = self._apply_cv_overrides(cv_cfg, params)
            if cv_task:
                cv_cfg.task = cv_task
                if not cv_flags.get("conf"):
                    cv_cfg.conf = min(cv_cfg.conf, 0.2)
                model_name = os.path.basename(cv_cfg.model_path or "")
                if "pose" not in model_name.lower():
                    logger.warning("[Machine] cv_task=%s 建议使用 pose 模型，当前=%s", cv_task, model_name or "?")

        sources = params.get("source") or []
        if not isinstance(sources, list) or not sources:
            yield json.dumps({"error": {"message": "缺少参数 source，至少需要提供一个视频源。"}}, ensure_ascii=False)
            return

        interval = params.get("interval") or 10
        try:
            interval = int(interval)
        except Exception:
            interval = 10

        def _to_bool(v: Any, default: bool = False) -> bool:
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("1", "true", "yes", "y", "on"):
                    return True
                if s in ("0", "false", "no", "n", "off", ""):
                    return False
            return default

        always_return = _to_bool(params.get("alwaysReturn", False), default=False)

        analyzer: Optional[StreamingAnalyze] = None
        meta: Optional[TaskMeta] = None

        try:
            mode = MODEL.SECURITY_SINGLE if len(sources) == 1 else MODEL.SECURITY_POLLING
            analyzer = self._create_machine_analyzer(
                mode=mode,
                sources=sources,
                interval=interval,
                task_id=str(rpc_id),
                backend=backend,
                cv_config=cv_cfg,
            )

            meta = TaskMeta(
                key=key,
                kind="machine",
                mode=mode.name,
                started_at=time.time(),
                rpc_id=rpc_id,
                analyzer=analyzer,
                client_id=ws_client_id,  # ✅ 绑定 client_id
            )
            self._register_task(meta)

            async for ev in _iterate_analyzer_events_async(analyzer):
                ev_type = ev.get("type")

                if ev_type == "vlm_stream_done":
                    source_id = ev.get("stream_rtsp_id")

                    raw_evidence = ev.get("evidence_image_urls") or []
                    evidence_image_urls = self._absolutize_evidence_urls(list(raw_evidence))

                    # ✅ B 侧 full_text 是“原始透传”，这里做宽松解析并统一为 objects[]
                    full_text_raw = ev.get("full_text")
                    objects_flat = self._coerce_to_objects_list(full_text_raw)

                    # ✅ 对外协议：full_text 必须是数组，且数组元素为 {"objects": objects[]}
                    full_text_out: Any = [{"objects": objects_flat}]

                    # ✅ 画框输入 events（单事件即可）
                    events_for_box: List[Dict[str, Any]] = [{"objects": objects_flat}] if objects_flat else []

                    # 空 objects：alwaysReturn=false => 不返回；alwaysReturn=true => 返回原证据图（box_urls 复用 evidence_urls）
                    if not objects_flat:
                        if not always_return:
                            continue
                        evidence_image_box_urls = list(evidence_image_urls)

                    # 非空：画框并导出到 storage/evidence_images_box/<task_id>/
                    else:
                        task_dir = (_EVIDENCE_IMAGES_BOX_ROOT / str(rpc_id)).resolve()
                        box_url_prefix = f"/storage/evidence_images_box/{rpc_id}"
                        box_cfg = SimpleNamespace(
                            vlm_static_evidence_images_box_dir=str(task_dir),
                            vlm_static_evidence_images_box_url_prefix=box_url_prefix,
                        )

                        evidence_images_for_box: List[str] = []

                        # 优先使用 raw_evidence（/storage/...）映射成本地文件
                        for u in raw_evidence:
                            if not isinstance(u, str) or not u:
                                continue
                            if u.startswith(("http://", "https://")):
                                continue
                            rel = u[1:] if u.startswith("/") else u
                            if rel.startswith("storage/"):
                                p = (project_root / rel).resolve()
                                if p.exists():
                                    evidence_images_for_box.append(str(p))

                        # 兜底：使用底层 evidence_images（file:// or local path）
                        if not evidence_images_for_box:
                            fallback_imgs = ev.get("evidence_images") or []
                            for uri in fallback_imgs:
                                if not isinstance(uri, str) or not uri:
                                    continue
                                local = uri[7:] if uri.lower().startswith("file://") else uri
                                if os.path.exists(local):
                                    evidence_images_for_box.append(local)

                        seg_idx = int(ev.get("segment_index", -1))
                        if not events_for_box:
                            events_for_box = [{"objects": objects_flat}]

                        try:
                            roi_cfg = self._pick_cv_roi(cv_cfg, source_id)
                            raw_box_urls = export_evidence_images_with_boxes(
                                evidence_images=evidence_images_for_box,
                                events=events_for_box,
                                seg_idx=seg_idx,
                                vlm_config=box_cfg,  # SimpleNamespace
                                roi=roi_cfg,
                            )
                            evidence_image_box_urls = (
                                self._absolutize_evidence_urls(list(raw_box_urls)) or list(evidence_image_urls)
                            )
                        except Exception as e:
                            logger.warning("[Machine] export boxes failed: seg#%s err=%s", seg_idx, e)
                            evidence_image_box_urls = list(evidence_image_urls)

                    payload = self._build_machine_success_payload(
                        source_id,
                        evidence_image_urls,
                        evidence_image_box_urls,
                        full_text_out,
                    )
                    self._emit_event({"kind": "machine_vlm_event", "rpc_id": rpc_id, "payload": payload})
                    yield json.dumps(payload, ensure_ascii=False)

                elif ev_type == "error":
                    msg = ev.get("message") or "巡检过程中发生未知错误"
                    yield json.dumps({"error": {"message": msg}}, ensure_ascii=False)

        finally:
            # ✅ finally：兜底再清一次
            try:
                await asyncio.to_thread(cleanup_machine_storage_sync, rpc_id)
            except Exception as e:
                logger.warning("[CLEANUP][MACHINE] finally cleanup failed: %s", e)

            try:
                self._unregister_task(key)
            except Exception:
                pass

            try:
                if analyzer:
                    analyzer.force_stop("machine task finished")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # BaseTemplateAgent 接口
    # ------------------------------------------------------------------
    async def process_message_stream(
        self,
        message: str = "",
        context: Optional[Dict[str, Any]] = None,
        original_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        self.last_active = datetime.now()
        original_params = original_params or {}

        # 抽取 ws_client_id（用于 machine/chat 任务绑定）
        ws_client_id: Optional[str] = None
        try:
            ws_client_id = (original_params.get("_ws_client_id") or "").strip() or None
        except Exception:
            ws_client_id = None

        if self._is_chat_envelope(original_params):
            content, _ = self._extract_chat_message_and_system_prompt(original_params)
            self._md_append("user", content)
            full_response = ""
            async for chunk in self._handle_chat_request(original_params):
                chunk_text = str(chunk) if chunk is not None else ""
                full_response += chunk_text
                yield chunk
            if full_response:
                self._md_append("assistant", full_response)
            return

        rpc_id, params = self._extract_rpc_id_and_params(original_params)
        async for chunk in self._handle_machine_request(rpc_id, params, ws_client_id=ws_client_id):
            yield chunk

    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        chunks: List[str] = []
        async for c in self.process_message_stream(message=message, context=context, original_params=original_params):
            chunks.append(str(c))
        return "\n".join(chunks)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.agent_name,
            "supports": {
                "chat": ["OFFLINE", "SECURITY_SINGLE"],
                "machine": ["SECURITY_SINGLE", "SECURITY_POLLING"],
            },
            "max_concurrent": self.max_concurrent,
        }

    def get_patrol_info(self) -> Dict[str, Any]:
        return {
            "status": self.patrol_status,
            "current_task_id": self.current_task_id,
            "active_tasks": [
                {
                    "key": meta.key,
                    "kind": meta.kind,
                    "mode": meta.mode,
                    "started_at": meta.started_at,
                    "session_id": meta.session_id,
                    "rpc_id": meta.rpc_id,
                    "url": meta.url,
                    "local_url": meta.local_url,
                    "stop_after_sec": meta.stop_after_sec,
                    "stop_deadline_ts": meta.stop_deadline_ts,
                    "client_id": meta.client_id,
                }
                for meta in self._task_registry.values()
            ],
        }
