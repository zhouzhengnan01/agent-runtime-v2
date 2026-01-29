# -*- coding: utf-8 -*-
"""
约定：
1) 所有 ffmpeg/ffprobe 子进程都通过 ffbin()/fpbin() 获取绝对路径。
2) 直播输入（RTSP/RTSPS/RTMP/RTMPS）统一附加：
   - 低延迟/小探测缓冲：-fflags nobuffer -flags low_delay -probesize 256k -analyzeduration 0
   - 并开启“到达壁钟映射”：-use_wallclock_as_timestamps 1
   - RTSP 额外：-rtsp_transport tcp
3) 切片流程：优先 copy，OpenCV 校验失败则重编码（libx264 + yuv420p）。
4) ensure_ffmpeg() 在启动时打印并校验实际使用的二进制绝对路径；发现可疑（bm/sophon）版本直接警告或中止。

说明（壁钟映射）：
- 我们使用输入参数 -use_wallclock_as_timestamps 1，让解复用时刻以“本机壁钟到达时间”为参考；
- 在 cut_and_standardize_segment() 中记录窗口开始（调用时刻）的 epoch，并生成 t0_iso/t1_iso；
- 这是“到达时间”，会包含网络/缓冲延迟，与“摄像机 UTC 拍摄时刻”略有差异，但实现成本最低、展示体验好。

5) 子进程超时保护：
- cut_and_standardize_segment 增加 cmd_timeout_sec 参数；
- 默认直播（RTSP/RTSPS/RTMP/RTMPS）=3s（可通过 env: FFMPEG_LIVE_CMD_TIMEOUT_SEC 或 FFMPEG_RTSP_CMD_TIMEOUT_SEC 覆盖）；
- 统一用于：
    * ffprobe 流探测
    * ffmpeg 视频流拷贝切片
    * ffmpeg 视频重编码切片
    * ffmpeg 音频切片
- 超时会记录日志并返回 timeout=True 的空结果，避免卡死上层流水线。
- RTMP / RTMPS：识别、输入参数分流、ffprobe 探测超时

6) 防止平台因 ffmpeg stderr “误判报错”掐断 websocket）：
- ffmpeg 相关命令统一“捕获并过滤 stderr”，不让其直接输出到父进程 stderr；
- 对已知无害的 RTSP 提示（如 PAUSE 551 Option not supported）直接过滤；
- 成功(returncode==0)：stderr 不外抛，只在 debug 级别记录（过滤后仍有内容才记）；
- 失败(returncode!=0)：抛出 CalledProcessError，走你原来的回退/失败逻辑。
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from typing import Optional, Dict, List, Iterator, Union
from time import time
from datetime import datetime, timezone
import uuid
from pathlib import Path

from app.shared.jetlinks_video.utils.logger_utils import get_logger
from app.shared.jetlinks_video.utils.file_utils import read_env_kv

logger = get_logger(__name__)

__all__ = [
    "ffbin", "fpbin", "ensure_ffmpeg",
    "normalize_path", "is_rtsp", "is_rtmp", "is_live", "input_args_for",
    "run_ffprobe", "probe_duration_seconds",
    "get_audio_duration_seconds", "get_video_duration_seconds",
    "have_audio_track", "standardize_audio_to_wav16k_mono",
    "standardize_video_strip_audio", "compress_video_for_vlm",
    "cut_and_standardize_segment", "streaming_cut_generator",
    "detect_silence_intervals",
    "grab_frame_by_index",
]

# ============================================================
# 类型别名：切片结果 dict 的 value 不止 str|None
# 还包括 float/int/bool 等元数据字段
# 用于消除 Pylance 返回类型告警
# ============================================================
SegmentValue = Union[str, float, int, bool, None]

# ============================================================
# 在 Python 进程内部初始化 ffmpeg 环境变量
# ============================================================
BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent
ENV_PATH = os.path.join(BASE_PATH, ".env")
# 这里应当是“目录”，不能写成 ffmpeg 文件名，更不能带尾部空格
_FFMPEG_ENV_BIN_DIR = read_env_kv(ENV_PATH, "FFMPEG_ENV_BIN_DIR") or "/usr/local/ffmpeg/bin"
_FFMPEG_DEFAULT_BIN = os.path.join(_FFMPEG_ENV_BIN_DIR, "ffmpeg")
_FFPROBE_DEFAULT_BIN = os.path.join(_FFMPEG_ENV_BIN_DIR, "ffprobe")


def _bootstrap_ffmpeg_env() -> None:
    """
    在 Python 进程内部初始化 ffmpeg 环境变量：

    - 如果 _FFMPEG_ENV_BIN_DIR 存在：
        * 把该目录插入 PATH 最前面；
        * 若外部没有显式设置 FFMPEG_BIN / FFPROBE_BIN，则设置默认值；
    - 如果目录不存在（如开发机上没有这个 env），则不做任何修改，
      此时 ffbin()/fpbin() 会退回使用系统默认 ffmpeg/ffprobe。
    """
    if not os.path.isdir(_FFMPEG_ENV_BIN_DIR):
        return

    old_path = os.environ.get("PATH", "")
    if _FFMPEG_ENV_BIN_DIR not in old_path.split(":"):
        os.environ["PATH"] = f"{_FFMPEG_ENV_BIN_DIR}:{old_path}"

    os.environ.setdefault("FFMPEG_BIN", _FFMPEG_DEFAULT_BIN)
    os.environ.setdefault("FFPROBE_BIN", _FFPROBE_DEFAULT_BIN)


_bootstrap_ffmpeg_env()

# ============================================================

SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

# ============================================================
# 防“误判报错”：过滤已知无害 stderr，并且不让 stderr 直出父进程
# ============================================================

_BENIGN_STDERR_PATTERNS = [
    # RTSP 结束/seek 时 FFmpeg 可能发送 PAUSE，某些服务端不支持会回 551；
    # 这通常不影响任务成功，但会出现在 stderr 且带 error 字样
    re.compile(r"method\s+PAUSE\s+failed:\s*551\s+Option\s+not\s+supported", re.IGNORECASE),
]


def _filter_benign_stderr(stderr_text: str) -> str:
    """过滤已知无害 stderr 行（例如 RTSP PAUSE 551）。"""
    if not stderr_text:
        return ""
    keep_lines: List[str] = []
    for line in (stderr_text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if any(p.search(s) for p in _BENIGN_STDERR_PATTERNS):
            continue
        keep_lines.append(s)
    return "\n".join(keep_lines)


def _run_cmd_quiet(
    cmd: List[str],
    *,
    timeout: Optional[float] = None,
    name: str = "ffmpeg",
) -> subprocess.CompletedProcess:
    """
    运行子进程但不把 stderr/stdout 直接输出到父进程，避免平台误判“底层报错”掐 websocket。

    - 成功(returncode==0)：过滤后 stderr 不外抛；若仍有内容，仅 logger.debug 记录
    - 失败(returncode!=0)：抛 CalledProcessError（stderr 为过滤后的内容，便于你日志/回退）
    """
    p = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,   # ffmpeg 的 stdout 基本无用，直接丢弃，避免内存膨胀
        stderr=subprocess.PIPE,      # 捕获 stderr，防止直出
        text=True,
        timeout=timeout,
    )

    filtered_err = _filter_benign_stderr(p.stderr or "")

    if p.returncode != 0:
        raise subprocess.CalledProcessError(
            p.returncode,
            cmd,
            output=None,
            stderr=filtered_err or (p.stderr or ""),
        )

    if filtered_err:
        logger.debug("[%s][stderr suppressed]\n%s", name, filtered_err)

    return p


# ============================================================
# grab_frame_by_index
# ============================================================

def grab_frame_by_index(video_path: str, out_dir: str, out_name: str, index: int) -> List[str]:
    """
    近似按帧索引导出 1 张图片（兜底/应急）。
    - 使用系统/环境指定的 ffmpeg，可与 sophon-ffmpeg_0.10.0 兼容。
    - out_name 不含扩展名，函数会生成 .jpg。

    返回：成功时 [输出路径]；失败时 []。
    """
    os.makedirs(out_dir, exist_ok=True)
    uid = uuid.uuid4().hex  # 32位十六进制
    out = os.path.join(out_dir, f"{out_name}_{uid}.jpg")
    cmd = [
        ffbin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"select='eq(n\\,{max(0, index)})'",
        "-vsync", "vfr",
        out
    ]
    try:
        _run_cmd_quiet(cmd, timeout=None, name="ffmpeg.grab_frame")
        return [out] if (os.path.exists(out) and os.path.getsize(out) > 0) else []
    except Exception:
        return []


# ======== 小工具 ========

def _iso_utc(ts_epoch: float) -> str:
    """epoch 秒 -> ISO-8601（UTC，秒级，尾部 Z）"""
    try:
        return datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc) \
            .isoformat(timespec="seconds") \
            .replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _live_cmd_timeout_sec(default: float = 3.0) -> float:
    """
    直播类（RTSP/RTMP）ffprobe/ffmpeg 子进程默认超时。
    兼容旧 env：FFMPEG_RTSP_CMD_TIMEOUT_SEC；新增统一 env：FFMPEG_LIVE_CMD_TIMEOUT_SEC
    """
    v = os.getenv("FFMPEG_LIVE_CMD_TIMEOUT_SEC") or os.getenv("FFMPEG_RTSP_CMD_TIMEOUT_SEC")
    try:
        t = float(v) if v is not None else float(default)
        return t if t > 0 else float(default)
    except Exception:
        return float(default)


# ======== 视频编码（可选） ========

def _get_video_reencode_codec(default: str = "libx264") -> str:
    v = (os.getenv("VIDEO_REENCODE_CODEC") or default).strip()
    return v or default


def _get_video_reencode_preset(default: str) -> str:
    v = (os.getenv("VIDEO_REENCODE_PRESET") or default).strip()
    return v or default


def _get_video_reencode_crf(default: str) -> str:
    v = (os.getenv("VIDEO_REENCODE_CRF") or default).strip()
    return v or default


def _get_video_encoder_opt_raw() -> str:
    # 兼容你截图里的小写 key（video_encoder_opt=...）
    v = (os.getenv("VIDEO_ENCODER_OPT") or os.getenv("video_encoder_opt") or "").strip()
    if v:
        return v
    # 某些部署仅写入 .env（未注入 os.environ），这里做文件兜底
    return (
        (read_env_kv(ENV_PATH, "VIDEO_ENCODER_OPT") or read_env_kv(ENV_PATH, "video_encoder_opt") or "")
        .strip()
    )


def _parse_video_encoder_opt(raw: str) -> List[str]:
    """
    将 `.env` 的 VIDEO_ENCODER_OPT 转为 ffmpeg args。

    支持两种格式：
    1) 逗号分隔 key=value：`gop_preset=2,is_dma_buffer=0` -> `-gop_preset 2 -is_dma_buffer 0`
    2) 原生 args：以 `-` 开头，例如 `-rc vbr -b:v 2M`
    """
    s = (raw or "").strip()
    if not s:
        return []

    if s.lstrip().startswith("-"):
        try:
            return shlex.split(s)
        except Exception:
            return s.split()

    args: List[str] = []
    for part in s.split(","):
        p = (part or "").strip()
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
            k = (k or "").strip()
            v = (v or "").strip()
            if not k:
                continue
            args.extend([f"-{k}", v])
        else:
            args.append(f"-{p}")
    return args


# ======== 路径与输入参数 ========

def ffbin() -> str:
    """优先用环境变量指定的 ffmpeg；否则查 PATH。"""
    env_bin = os.getenv("FFMPEG_BIN")
    if env_bin:
        return env_bin
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def fpbin() -> str:
    """优先用环境变量指定的 ffprobe；否则查 PATH。"""
    env_bin = os.getenv("FFPROBE_BIN")
    if env_bin:
        return env_bin
    found = shutil.which("ffprobe")
    return found or "ffprobe"


def ensure_ffmpeg() -> None:
    ff = ffbin()
    fp = fpbin()
    if not shutil.which(ff):
        raise RuntimeError(f"[FFmpeg] 未找到 ffmpeg，可导出 FFMPEG_BIN 指定：{ff}")
    if not shutil.which(fp):
        raise RuntimeError(f"[FFmpeg] 未找到 ffprobe，可导出 FFPROBE_BIN 指定：{fp}")
    logger.info(f"[FFmpeg] using ffmpeg={shutil.which(ff)} ffprobe={shutil.which(fp)}")


def normalize_path(url_or_path: str) -> str:
    """将 file://xxx 转为本地路径；其他协议原样返回"""
    if (url_or_path or "").startswith("file://"):
        return url_or_path.replace("file://", "", 1)
    return url_or_path


def is_rtsp(src: str) -> bool:
    s = (src or "").lower()
    return s.startswith("rtsp://") or s.startswith("rtsps://")


def is_rtmp(src: str) -> bool:
    s = (src or "").lower()
    return s.startswith("rtmp://") or s.startswith("rtmps://")


def is_live(src: str) -> bool:
    return is_rtsp(src) or is_rtmp(src)


def _rtsp_hwaccel_args() -> List[str]:
    # RTSP hwaccel/decoder options. Defaults keep legacy behavior.
    hwaccel = (os.getenv("FFMPEG_RTSP_HWACCEL") or "").strip()
    vcodec = (os.getenv("FFMPEG_RTSP_VCODEC") or "").strip()
    args: List[str] = []
    if hwaccel:
        args += ["-hwaccel", hwaccel]
    else:
        args += ["-hwaccel", "none"]
    if vcodec and vcodec.lower() not in {"auto", "default", "none"}:
        args += ["-c:v", vcodec]
    elif not hwaccel or hwaccel.lower() == "none":
        args += ["-c:v", "h264"]
    return args


def input_args_for(src: str, *, for_probe: bool = False) -> List[str]:
    """
    为 RTSP / RTMP / 本地分别返回合适的输入端参数。
    for_probe=True 时用于 ffprobe（ffprobe 不支持 -hwaccel / -c:v / -nostdin）。
    直播输入统一加 -use_wallclock_as_timestamps 1。
    """
    rtsp = is_rtsp(src)
    rtmp = is_rtmp(src)

    if for_probe:
        if rtsp:
            return ["-rtsp_transport", "tcp", "-use_wallclock_as_timestamps", "1"]
        if rtmp:
            return ["-use_wallclock_as_timestamps", "1"]
        return []

    if rtsp:
        return [
            "-nostdin",
            "-rtsp_transport", "tcp",
            "-use_wallclock_as_timestamps", "1",
            *_rtsp_hwaccel_args(),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "256k",
            "-analyzeduration", "0",
        ]

    if rtmp:
        # RTMP/RTMPS：不要带 -rtsp_transport；也不要强行指定 -c:v h264（可能不是 h264）
        return [
            "-nostdin",
            "-use_wallclock_as_timestamps", "1",
            "-hwaccel", "none",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "256k",
            "-analyzeduration", "0",
        ]

    return ["-nostdin"]


# ======== ffprobe 基础 ========

def _sanitize_ffprobe_json(raw: str) -> str:
    """
    Sophon ffprobe 可能把日志混进 stdout，导致 JSON 解析失败。
    这里只保留“看起来像 JSON 的行”，尽量剔除噪声。
    """
    if not raw:
        return raw
    lines: List[str] = []
    started = False
    for line in raw.splitlines():
        if not started:
            if "{" in line:
                started = True
                line = line[line.find("{"):]
            else:
                continue
        s = line.strip()
        if not s:
            continue
        head = s[0]
        if head in "{}":
            lines.append(s)
            continue
        if head == "\"":
            lines.append(s)
            continue
        if head == "]":
            lines.append(s)
            continue
        if head == "[":
            tail = s[1:].lstrip()
            if not tail or tail[0] in "]}\"-0123456789ntf":
                lines.append(s)
            continue
    return "\n".join(lines) or raw

def run_ffprobe(cmd: List[str], *, timeout: Optional[float] = None) -> str:
    """
    ffprobe 执行封装（支持 timeout）。
    默认保持旧行为：timeout=None（无超时）。
    """
    if timeout is None:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
        timeout=timeout,
    )
    return p.stdout


def probe_duration_seconds(url_or_path: str) -> Optional[float]:
    """
    通用时长探测：本地文件返回秒数；直播（RTSP/RTMP）可能无 duration -> 返回 None
    """
    src = normalize_path(url_or_path)
    in_args = input_args_for(src, for_probe=True)

    # 对直播给一个小 timeout，避免卡死
    timeout = _live_cmd_timeout_sec() if is_live(src) else None

    try:
        out = run_ffprobe([
            fpbin(), "-v", "error", *in_args,
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            src
        ], timeout=timeout).strip()
        if out:
            return max(0.0, float(out))
    except subprocess.TimeoutExpired:
        logger.debug("[FFmpeg] ffprobe duration 超时(%.1fs): %s", float(timeout or 0), src)
        return None
    except Exception:
        pass

    try:
        jout = run_ffprobe([
            fpbin(), "-v", "error", *in_args,
            "-print_format", "json",
            "-show_format",
            src
        ], timeout=timeout)
        payload = _sanitize_ffprobe_json(jout)
        j = json.loads(payload or "{}")
        dur = j.get("format", {}).get("duration")
        if dur is None or dur == "N/A":
            return None
        return max(0.0, float(dur))
    except subprocess.TimeoutExpired:
        logger.debug("[FFmpeg] ffprobe show_format 超时(%.1fs): %s", float(timeout or 0), src)
        return None
    except Exception as e:
        logger.debug(f"[FFmpeg] ffprobe 获取时长失败: {e}")
        return None


# ======== 对外简单 API ========

def get_audio_duration_seconds(standardized_audio: str) -> Optional[float]:
    return probe_duration_seconds(standardized_audio)


def get_video_duration_seconds(standardized_video: str) -> Optional[float]:
    return probe_duration_seconds(standardized_video)


# ======== 音轨判定 ========

def have_audio_track(url_or_path: str) -> bool:
    src = normalize_path(url_or_path)
    in_args = input_args_for(src, for_probe=True)

    timeout = _live_cmd_timeout_sec() if is_live(src) else None

    cmd = [
        fpbin(), "-v", "error", *in_args,
        "-select_streams", "a", "-show_streams",
        "-print_format", "json", src
    ]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout,
        )
        payload = _sanitize_ffprobe_json(p.stdout or "")
        j = json.loads(payload or "{}")
        streams = j.get("streams", []) or []
        logger.info(f"[ffmpeg] 音轨探测 have_audio_track(): {len(streams) > 0}")
        return len(streams) > 0
    except subprocess.TimeoutExpired:
        logger.warning("[ffprobe] 音轨探测超时(%.1fs)，src=%s", float(timeout or 0), src)
        return False
    except Exception as e:
        logger.debug(f"[FFmpeg] ffprobe 检测音轨失败: {e}")
        return False


# ======== 标准化 ========

def standardize_audio_to_wav16k_mono(src_path: str, out_path: str) -> str:
    ensure_ffmpeg()
    src = normalize_path(src_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        ffbin(), "-y",
        *input_args_for(src),
        "-i", src, "-vn",
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        out_path
    ]
    # 这里本来就把输出吞掉了，不会直出 stderr（保留原逻辑）
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path


def standardize_video_strip_audio(src_path: str, out_path: str, reencode: bool = False) -> str:
    """
    生成无声视频给视觉侧：
    - reencode=False（默认）：视频流直接拷贝，最快
    - reencode=True：重编码为 H.264 + yuv420p，兼容性更好但耗时更长
    """
    ensure_ffmpeg()
    src = normalize_path(src_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not reencode:
        cmd = [
            ffbin(), "-y",
            *input_args_for(src),
            "-i", src, "-an", "-c:v", "copy",
            out_path
        ]
    else:
        codec = _get_video_reencode_codec()
        preset = _get_video_reencode_preset("fast")
        crf = _get_video_reencode_crf("23")
        opt_raw = _get_video_encoder_opt_raw()
        opt_args = _parse_video_encoder_opt(opt_raw)

        def _build(extra: List[str]) -> List[str]:
            c = [
                ffbin(), "-y",
                *input_args_for(src),
                "-i", src, "-an",
                "-c:v", codec,
            ]
            if codec.lower() == "libx264":
                c += ["-preset", preset, "-crf", crf]
            c += [*extra, "-pix_fmt", "yuv420p", out_path]
            return c

        try:
            subprocess.check_call(_build(opt_args), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception:
            if opt_args:
                logger.warning("[FFmpeg] standardize_video_strip_audio 重编码失败，回退不带 VIDEO_ENCODER_OPT 再试一次。")
                subprocess.check_call(_build([]), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            else:
                raise
        return out_path

    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path


# ======== 小视频压缩（给 VLM / 直传场景） ========

def compress_video_for_vlm(
    in_video: str, out_video: str,
    fps: int = 8, height: int = 480,
    crf: int = 28, preset: str = "veryfast"
) -> str:
    """
    小视频压缩规范：
    - 编码：H.264
    - 分辨率：最长边 ≤ height（等比缩放：scale=-2:height）
    - FPS <= fps
    - CRF ~ 28，preset veryfast
    - 去音轨（ASR 走单独管线）
    """
    ensure_ffmpeg()
    src = normalize_path(in_video)
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    vf = f"scale=-2:{int(height)}"
    codec = _get_video_reencode_codec()
    preset_eff = _get_video_reencode_preset(preset)
    crf_eff = _get_video_reencode_crf(str(int(crf)))
    opt_raw = _get_video_encoder_opt_raw()
    opt_args = _parse_video_encoder_opt(opt_raw)

    def _build(extra: List[str]) -> List[str]:
        c = [
            ffbin(), "-y",
            *input_args_for(src),
            "-i", src, "-an",
            "-r", str(int(fps)), "-vf", vf,
            "-c:v", codec,
        ]
        if codec.lower() == "libx264":
            c += ["-preset", preset_eff, "-crf", crf_eff]
        c += [*extra, "-pix_fmt", "yuv420p", out_video]
        return c

    try:
        subprocess.check_call(_build(opt_args), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception:
        if opt_args:
            logger.warning("[FFmpeg] compress_video_for_vlm 失败，回退不带 VIDEO_ENCODER_OPT 再试一次。")
            subprocess.check_call(_build([]), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            raise
    return out_video


# ======== 切窗并标准化（含容错 + 子进程超时） ========

def cut_and_standardize_segment(
    src_url: str,
    start_time: float,
    duration: float,
    output_dir: str,
    segment_index: int,
    have_audio: bool = True,
    *,
    cmd_timeout_sec: Optional[float] = None,
) -> Dict[str, SegmentValue]:
    """
    切出一段音/视频片段，并标准化为：
    - 若有视频：去音轨的无声 mp4（video_path）
    - 若有音频：16kHz 单声道 wav（audio_path）
    纯音频 → 仅返回 audio_path（video_path=None）
    纯视频 → 仅返回 video_path（audio_path=None）

    新增：
    - cmd_timeout_sec：子进程超时秒数
      * 不传时：直播（RTSP/RTMP）默认 3s；本地/文件 None
      * env 可覆盖：FFMPEG_LIVE_CMD_TIMEOUT_SEC / FFMPEG_RTSP_CMD_TIMEOUT_SEC

    容错：
    1) 先用“流拷贝”快速切，写完用 OpenCV 校验能否读取；
    2) 若打不开/0帧，自动回退“重编码”切片，确保可读。

    壁钟映射：
    - 记录 wallclock_epoch 并生成 t0_iso/t1_iso
    """
    import cv2 as _cv2  # 延迟导入

    def _verify_openable(p: str) -> bool:
        try:
            cap = _cv2.VideoCapture(p)
            if not cap.isOpened():
                cap.release()
                return False
            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                ok, _ = cap.read()
                cap.release()
                return bool(ok)
            cap.release()
            return True
        except Exception:
            return False

    def _safe_remove(p: Optional[str]) -> None:
        if not p:
            return
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    def _effective_timeout_sec(src: str) -> Optional[float]:
        # 显式参数优先
        if cmd_timeout_sec is not None:
            try:
                t = float(cmd_timeout_sec)
                return t if t > 0 else None
            except Exception:
                return None

        # 默认：直播给一个小超时，避免卡死
        if is_live(src):
            return _live_cmd_timeout_sec(default=3.0)

        # 本地文件默认不设超时（你也可以按需设置）
        return None

    ensure_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    wallclock_epoch_t0 = time()
    wallclock_epoch_t1 = wallclock_epoch_t0 + float(duration)

    uid = uuid.uuid4().hex
    v_out = os.path.join(output_dir, f"segment_{segment_index:04d}_{uid}_video.mp4")
    a_out = os.path.join(output_dir, f"segment_{segment_index:04d}_{uid}_audio.wav")

    src_norm = normalize_path(src_url)
    live = is_live(src_norm)
    timeout_sec = _effective_timeout_sec(src_norm)
    timed_out = False

    # 探测实际存在的音/视频流
    def _has_stream(kind: str) -> bool:
        nonlocal timed_out
        cmd = [
            fpbin(), "-v", "error",
            *input_args_for(src_norm, for_probe=True),
            "-select_streams", f"{kind}:0",
            "-show_entries", "stream=index",
            "-of", "json",
            src_norm,
        ]
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout_sec,
            )
            payload = _sanitize_ffprobe_json(p.stdout or "")
            data = json.loads(payload or "{}")
            return bool(data.get("streams"))
        except subprocess.TimeoutExpired:
            timed_out = True
            logger.warning(
                "[FFprobe] 探测 %s 流超时(%.1fs)，src=%s",
                "video" if kind == "v" else "audio",
                float(timeout_sec or 0),
                src_norm,
            )
            return False
        except Exception:
            return False

    has_video = _has_stream("v")
    has_audio_real = _has_stream("a")

    # 探测阶段即超时
    if timed_out:
        _safe_remove(v_out)
        _safe_remove(a_out)
        logger.warning(
            "[FFmpeg] seg#%s 探测阶段超时，视为本次切片失败并跳过。src=%s",
            segment_index,
            src_norm,
        )
        return {
            "video_path": None,
            "audio_path": None,
            "t0": start_time,
            "t1": start_time + duration,
            "duration": duration,
            "index": segment_index,
            "have_audio": False,
            "timeout": True,
            "timeout_sec": timeout_sec,
            "t0_iso": _iso_utc(wallclock_epoch_t0),
            "t1_iso": _iso_utc(wallclock_epoch_t1),
            "t0_epoch": wallclock_epoch_t0,
            "t1_epoch": wallclock_epoch_t1,
        }

    v_path_ret: Optional[str] = None
    a_path_ret: Optional[str] = None

    # -------- 视频 --------
    if has_video:
        # 对 RTSP/RTMP 等直播流：不要使用 -ss 做“相对时间 seek”（会导致越跑越慢/超时）。
        seek_args = [] if live else ["-ss", str(start_time)]
        cmd_v_fast = [
            ffbin(), "-y", "-hide_banner", "-loglevel", "error",
            *input_args_for(src_norm),
            *seek_args, "-i", src_norm, "-t", str(duration),
            "-map", "v:0", "-an",
            "-fflags", "+genpts",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            "-c:v", "copy",
            v_out
        ]

        try:
            _run_cmd_quiet(cmd_v_fast, timeout=timeout_sec, name="ffmpeg.cut.fastcopy")
        except subprocess.TimeoutExpired:
            timed_out = True
            _safe_remove(v_out)
            logger.warning(
                "[FFmpeg] 视频流拷贝切片超时(%.1fs)，seg#%s src=%s",
                float(timeout_sec or 0),
                segment_index,
                src_norm,
            )
        except subprocess.CalledProcessError:
            logger.warning("[FFmpeg] 流拷贝切分失败, 回落到重编码切分。")
        except Exception as e:
            logger.warning("[FFmpeg] 流拷贝切分异常: %s", e)

        if not timed_out:
            need_reencode = False
            if not os.path.exists(v_out) or os.path.getsize(v_out) <= 0:
                need_reencode = True
            elif not _verify_openable(v_out):
                need_reencode = True
                logger.warning("[FFmpeg] 流拷贝切片视频无法被OpenCV打开, 删除坏文件, 回落到重编码切分。")
                _safe_remove(v_out)

            if need_reencode:
                codec = _get_video_reencode_codec()
                preset = _get_video_reencode_preset("veryfast")
                crf = _get_video_reencode_crf("23")
                opt_raw = _get_video_encoder_opt_raw()
                opt_args = _parse_video_encoder_opt(opt_raw)

                def _build(extra: List[str]) -> List[str]:
                    c = [
                        ffbin(), "-y", "-hide_banner", "-loglevel", "error",
                        *input_args_for(src_norm),
                        "-i", src_norm,
                        *([] if live else ["-ss", str(start_time)]),
                        "-t", str(duration),
                        "-map", "v:0", "-an",
                        "-c:v", codec,
                    ]
                    if codec.lower() == "libx264":
                        c += ["-preset", preset, "-crf", crf, "-g", "16", "-keyint_min", "16", "-sc_threshold", "0"]
                    else:
                        c += ["-g", "16"]
                    c += [*extra, "-pix_fmt", "yuv420p", "-movflags", "+faststart", v_out]
                    return c

                cmd_v_slow = _build(opt_args)
                try:
                    _run_cmd_quiet(cmd_v_slow, timeout=timeout_sec, name="ffmpeg.cut.reencode")
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _safe_remove(v_out)
                    logger.warning(
                        "[FFmpeg] 视频重编码切片超时(%.1fs)，seg#%s src=%s",
                        float(timeout_sec or 0),
                        segment_index,
                        src_norm,
                    )
                except Exception as e:
                    if opt_args:
                        _safe_remove(v_out)
                        logger.warning("[FFmpeg] 视频重编码切片失败（带 VIDEO_ENCODER_OPT），回退默认参数再试一次: %s", e)
                        try:
                            _run_cmd_quiet(_build([]), timeout=timeout_sec, name="ffmpeg.cut.reencode.fallback")
                        except Exception as e2:
                            _safe_remove(v_out)
                            logger.warning("[FFmpeg] 视频重编码切片失败: %s", e2)
                    else:
                        _safe_remove(v_out)
                        logger.warning("[FFmpeg] 视频重编码切片失败: %s", e)

        if (not timed_out) and os.path.exists(v_out) and os.path.getsize(v_out) > 0:
            v_path_ret = v_out

    # -------- 音频 --------
    if has_audio_real and have_audio and not timed_out:
        # 直播流同理：不要 seek，直接取接下来 duration 秒
        seek_args = [] if live else ["-ss", str(start_time)]
        cmd_a = [
            ffbin(), "-y", "-hide_banner", "-loglevel", "error",
            *input_args_for(src_norm),
            *seek_args, "-i", src_norm, "-t", str(duration),
            "-map", "a:0",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            a_out
        ]
        try:
            _run_cmd_quiet(cmd_a, timeout=timeout_sec, name="ffmpeg.cut.audio")
            if os.path.exists(a_out) and os.path.getsize(a_out) > 0:
                a_path_ret = a_out
        except subprocess.TimeoutExpired:
            timed_out = True
            _safe_remove(a_out)
            logger.warning(
                "[FFmpeg] 音频切片超时(%.1fs)，seg#%s src=%s",
                float(timeout_sec or 0),
                segment_index,
                src_norm,
            )
        except Exception as e:
            _safe_remove(a_out)
            logger.warning("[FFmpeg] 音频切片失败: %s", e)

    logger.info(f"切出视频片段：{v_path_ret}, 切出音频片段：{a_path_ret}")

    return {
        "video_path": v_path_ret,
        "audio_path": a_path_ret,
        "t0": start_time,
        "t1": start_time + duration,
        "duration": duration,
        "index": segment_index,
        "have_audio": bool(a_path_ret),
        "timeout": bool(timed_out),
        "timeout_sec": timeout_sec,
        "t0_iso": _iso_utc(wallclock_epoch_t0),
        "t1_iso": _iso_utc(wallclock_epoch_t1),
        "t0_epoch": wallclock_epoch_t0,
        "t1_epoch": wallclock_epoch_t1,
    }


# ======== 连续切窗生成器 ========

def streaming_cut_generator(
    src_url: str,
    output_dir: str,
    slice_sec: int = 10,
    have_audio: Optional[bool] = None,
    start_offset: float = 0.0,
    max_duration: Optional[float] = None,
) -> Iterator[Dict[str, SegmentValue]]:
    """
    连续流式切窗（生成标准化后片段）：
    - 本地文件：探测总时长，到末尾停止
    - RTSP/RTMP/直播：无限循环（外部 STOP 中断）
    - 片段内的音/视频轨道是否存在，以 cut_and_standardize_segment 的“实际产出”为准
    """
    ensure_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    total_dur = probe_duration_seconds(src_url)
    if total_dur is not None and max_duration is None:
        max_duration = total_dur

    seg_idx = 0
    t0 = float(start_offset)

    logger.info(f"开始流式切窗: {src_url}, 窗口={slice_sec}s")
    while True:
        if max_duration is not None and t0 >= max_duration:
            logger.info(f"[FFmpeg] 已到达文件末尾，总时长 {max_duration:.2f}s，结束切窗。")
            break

        seg = cut_and_standardize_segment(
            src_url=src_url,
            start_time=t0,
            duration=slice_sec,
            output_dir=output_dir,
            segment_index=seg_idx,
            have_audio=True if have_audio is None else bool(have_audio),
        )

        has_v = bool(seg.get("video_path"))
        has_a = bool(seg.get("audio_path"))
        logger.debug(
            "[FFmpeg] 切片完成 seg#%04d t0=%.3f t1=%.3f 产出: video=%s audio=%s | t0_iso=%s | timeout=%s",
            seg_idx,
            float(seg.get("t0") or 0.0),
            float(seg.get("t1") or 0.0),
            "yes" if has_v else "no",
            "yes" if has_a else "no",
            seg.get("t0_iso"),
            "yes" if seg.get("timeout") else "no",
        )

        yield seg

        seg_idx += 1
        t0 += slice_sec


# ======== 静音探测 ========

def detect_silence_intervals(
    src_url: str,
    noise_db: float = -35.0,
    min_silence: float = 0.5,
    audio_stream_index: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    max_intervals: Optional[int] = None,
) -> List[Dict[str, Optional[float]]]:
    """
    基于 FFmpeg silencedetect 的静音探测。
    适用于本地文件与直播流（直播建议设置 timeout_sec 以避免无限跑）。

    返回：
    [
        {"start": 12.34, "end": 14.90, "duration": 2.56},
        ...
    ]
    若流在“静音中”就结束，最后一段的 end/duration 可能为 None。
    """

    def _normalize(p: str) -> str:
        return p.replace("file://", "", 1) if p.startswith("file://") else p

    src = _normalize(src_url)

    filter_expr = f"silencedetect=noise={noise_db}dB:d={min_silence}"
    cmd = [
        ffbin(), "-hide_banner", "-nostats", "-v", "info",
        *input_args_for(src),
        "-i", src,
    ]
    if audio_stream_index is not None:
        cmd += ["-map", f"0:a:{audio_stream_index}"]
    else:
        cmd += ["-map", "0:a:0"]
    cmd += ["-af", filter_expr, "-vn", "-f", "null", "-"]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1
    )

    intervals: List[Dict[str, Optional[float]]] = []
    cur_start: Optional[float] = None
    start_ts = time()

    try:
        assert proc.stderr is not None
        for line in iter(proc.stderr.readline, ''):
            if timeout_sec is not None and (time() - start_ts) > timeout_sec:
                break

            m1 = SILENCE_START_RE.search(line)
            if m1:
                try:
                    cur_start = float(m1.group(1))
                except Exception:
                    cur_start = None
                continue

            m2 = SILENCE_END_RE.search(line)
            if m2:
                try:
                    end = float(m2.group(1))
                    dur = float(m2.group(2))
                except Exception:
                    end, dur = None, None

                if cur_start is None and end is not None and dur is not None:
                    cur_start = max(0.0, end - dur)

                intervals.append({"start": cur_start, "end": end, "duration": dur})
                cur_start = None

                if max_intervals is not None and len(intervals) >= max_intervals:
                    break

        try:
            proc.terminate()
        except Exception:
            pass
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    finally:
        if cur_start is not None:
            intervals.append({"start": cur_start, "end": None, "duration": None})

    return intervals
