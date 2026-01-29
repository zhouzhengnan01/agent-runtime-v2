'''
Author: 13594053100@163.com
Date: 2025-10-08 08:41:48
LastEditTime: 2025-12-03 17:46:39
'''

from enum import Enum

class MODEL(Enum):
    OFFLINE = "offline"
    SECURITY_SINGLE = "security_single"
    SECURITY_POLLING = "security_polling"

class SOURCE_KIND(Enum):
    # 离线本地文件 
    AUDIO_FILE = "audio_file"
    VIDEO_FILE = "video_file"
    # 实时流
    RTSP = "rtsp"


class CLOUD_ASR_MODEL_NAME(Enum):
    # 模型运行在阿里百炼平台
    PARAFORMER_REALTIME_V2 = "paraformer-realtime-v2"

class LOCAL_ASR_MODEL_NAME(Enum):
    # 模型运行在算能TPU
    WHISPER = "Whisper"


    

    
