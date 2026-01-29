'''
Author: 13594053100@163.com
Date: 2025-11-04 09:36:22
LastEditTime: 2025-12-01 17:12:23
'''
from typing import Optional, List,Union
from pydantic import BaseModel, Field, model_validator, ConfigDict
from app.shared.jetlinks_video.utils.logger_utils import get_logger

logger = get_logger(__name__)

class RTSP(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    rtsp_id: Optional[Union[int, str]] = None # 流唯一ID, 必须由上层提供, 便于上层唯一对应流信息
    rtsp_url: Optional[str] = None
    rtsp_system_prompt: Optional[str] = ""
    rtsp_cut_number: int = Field(ge=1, le=20, default=1, description="该流在一轮轮询中要切几个窗口")

class RTSPBatchConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    polling_list: List[RTSP]
    polling_batch_interval: float = Field(
        default=60.0 * 10.0, ge=0.0, description="两轮轮询之间的间隔，单位秒"
    )

    @model_validator(mode="after")
    def _check_vlm_config(self):
        if len(self.polling_list) >= 2 and self.polling_batch_interval < 10:
            raise ValueError(f'两流及以上时, 默认为MODE.SECURITY_POLLING, 此时polling_batch_interval必须大于等于10秒。')
        return self
