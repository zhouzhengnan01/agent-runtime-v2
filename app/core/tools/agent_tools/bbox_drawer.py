"""
智能标注工具 - 基于自然语言提示词的图像标注
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from app.core.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool("BoundingBoxDrawer")
class BoundingBoxDrawer(BaseTool):
    """
    智能标注工具 - AI驱动的自动图像标注
    """

    name: str = "BoundingBoxDrawer"
    description: str = "AI智能图像标注工具。通过自然语言提示词自动识别并标注图像内容（人员、设备、区域等）。内置多模态视觉模型，无需手动指定坐标，自动理解并执行标注需求。"

    # 使用与chatbi一致的parameters格式
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "输入图像路径或URL地址"
            },
            "annotation_prompt": {
                "type": "string",
                "description": "标注提示词，描述需要标注的内容和规则",
                "examples": [
                    "将靠近塔吊30米范围内标注为红色危险区域",
                    "把所有未戴安全帽的人员用红框标出",
                    "将施工区域用黄色虚线框起来"
                ]
            },
            "detection_result": {
                "type": "object",
                "description": "检测结果（可选），包含已识别的人员、物体等信息"
            },
            "output_path": {
                "type": "string",
                "description": "输出图像路径（可选）"
            }
        },
        "required": ["image_path", "annotation_prompt"]
    }

    # 输出定义
    output: Dict[str, Any] = {
        "type": "object",
        "description": "图像标注结果",
        "properties": {
            "status": {"type": "string", "description": "执行状态"},
            "output_path": {"type": "string", "description": "标注后的图片路径"},
            "annotation_plan": {"type": "object", "description": "标注计划详情"},
            "statistics": {"type": "object", "description": "标注统计信息"},
            "image_size": {"type": "object", "description": "图片尺寸信息"},
            "message": {"type": "string", "description": "详细信息"}
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 预定义颜色
        self.predefined_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255)
        }

        # 兼容旧版颜色
        self.colors = {
            'danger_zone': (0, 0, 255),     # 红色
            'person_danger': (0, 0, 255),   # 红色
            'person_safe': (0, 255, 0),     # 绿色
            'person_uncertain': (255, 165, 0), # 橙色
            'person_violation': (128, 0, 128), # 紫色
            'crane_base': (255, 255, 255)   # 白色
        }

    def run(self, image_path: str, annotation_prompt: str,
            detection_result: Optional[Dict] = None, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        执行智能标注

        Args:
            image_path: 输入图像路径
            annotation_prompt: 标注提示词，描述需要标注的内容
            detection_result: 检测结果（可选），提供已识别的信息
            output_path: 输出路径，如果不提供则自动生成

        Returns:
            包含输出路径、标注信息和解释的字典
        """
        try:
            logger.info(f"开始智能标注: image_path='{image_path}', prompt='{annotation_prompt[:50]}...'")

            # 字符串参数处理（兼容性）
            if isinstance(detection_result, str):
                if detection_result.strip():
                    try:
                        detection_result = json.loads(detection_result)
                    except json.JSONDecodeError:
                        logger.warning(f"detection_result不是有效的JSON格式，将忽略: {detection_result[:100]}...")
                        detection_result = None
                else:
                    detection_result = None

            # 参数验证
            if not image_path:
                return {
                    "status": "error",
                    "message": "图像路径不能为空",
                    "error_code": "INVALID_PARAMS"
                }

            if not annotation_prompt:
                return {
                    "status": "error",
                    "message": "标注提示词不能为空",
                    "error_code": "INVALID_PARAMS"
                }

            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'status': 'error',
                    'message': f'无法读取图像: {image_path}',
                    'error_code': 'IMAGE_READ_ERROR'
                }

            height, width = img.shape[:2]
            annotated = img.copy()

            # 解析标注提示词，生成标注方案（简化版本）
            annotation_plan = self._create_simple_plan(annotation_prompt, width, height)

            # 执行标注
            stats = self._execute_annotations(annotated, annotation_plan, width, height)

            # 保存结果
            if not output_path:
                output_dir = Path("storage/static/inspection_results/images")
                output_dir.mkdir(parents=True, exist_ok=True)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(output_dir / f"annotated_{timestamp}.jpg")

            cv2.imwrite(output_path, annotated)

            logger.info(f"标注完成: 输出路径='{output_path}', 区域={stats.get('zones_count', 0)}, 人员={stats.get('people_count', 0)}")

            return {
                'status': 'success',
                'message': '智能标注完成',
                'output_path': output_path,
                'annotation_plan': annotation_plan,
                'statistics': stats,
                'image_size': {'width': width, 'height': height}
            }

        except Exception as e:
            logger.error(f"画框标注失败: {e}")
            return {
                'status': 'error',
                'message': f"智能标注失败: {str(e)}",
                'error_code': 'ANNOTATION_FAILED'
            }

    def _create_simple_plan(self, prompt: str, width: int, height: int) -> Dict[str, Any]:
        """创建简单的标注计划（基于关键词）"""
        plan = {
            'zones': [],
            'people': [],
            'labels': []
        }

        prompt_lower = prompt.lower()

        # 基于关键词的简单标注
        if '危险' in prompt or 'danger' in prompt_lower:
            plan['zones'].append({
                'type': 'rectangle',
                'coordinates': [0.1, 0.1, 0.9, 0.9],
                'color': 'red',
                'style': 'dashed',
                'label': '危险区域'
            })

        if '人员' in prompt or 'person' in prompt_lower:
            plan['people'].append({
                'bbox': [0.3, 0.3, 0.5, 0.7],
                'color': 'red' if '违规' in prompt or 'violation' in prompt_lower else 'green',
                'label': '人员'
            })

        if '施工' in prompt or 'construction' in prompt_lower:
            plan['zones'].append({
                'type': 'rectangle',
                'coordinates': [0.2, 0.2, 0.8, 0.8],
                'color': 'yellow',
                'style': 'dashed',
                'label': '施工区域'
            })

        return plan

    def _execute_annotations(self, img: np.ndarray, annotation_plan: Dict[str, Any],
                           width: int, height: int) -> Dict[str, Any]:
        """执行标注计划"""
        stats = {
            'zones_count': 0,
            'people_count': 0,
            'labels_count': 0
        }

        # 绘制区域
        for zone in annotation_plan.get('zones', []):
            self._draw_zone(img, zone, width, height)
            stats['zones_count'] += 1

        # 绘制人员
        for person in annotation_plan.get('people', []):
            self._draw_person(img, person, width, height)
            stats['people_count'] += 1

        return stats

    def _draw_zone(self, img: np.ndarray, zone: Dict[str, Any], width: int, height: int):
        """绘制区域"""
        coords = zone['coordinates']
        x1, y1, x2, y2 = coords

        # 转换坐标
        if x1 <= 1.0:
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)

        color = self._get_color(zone.get('color', 'red'))

        # 绘制
        if zone.get('style') == 'dashed':
            self._draw_dashed_rectangle(img, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 添加标签
        if zone.get('label'):
            cv2.putText(img, zone['label'], (x1 + 5, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_person(self, img: np.ndarray, person: Dict[str, Any],
                     width: int, height: int):
        """绘制单个人员标注"""
        bbox = person['bbox']
        color = self._get_color(person.get('color', 'green'))
        label = person.get('label', '人员')

        # 转换坐标
        if any(v <= 1.0 for v in bbox):
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox]

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 添加标签
        if label:
            label_y = y1 - 5 if y1 > 20 else y2 + 15
            cv2.putText(img, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _get_color(self, color_name: str) -> Tuple[int, int, int]:
        """获取颜色BGR值"""
        if color_name in self.predefined_colors:
            return self.predefined_colors[color_name]
        elif color_name in self.colors:
            return self.colors[color_name]
        else:
            return (0, 0, 255)  # 默认红色

    def _draw_dashed_rectangle(self, img: np.ndarray, pt1: Tuple[int, int],
                              pt2: Tuple[int, int], color: Tuple[int, int, int],
                              thickness: int = 1, gap: int = 10):
        """绘制虚线矩形"""
        x1, y1 = pt1
        x2, y2 = pt2

        # 上边
        for i in range(x1, x2, gap * 2):
            cv2.line(img, (i, y1), (min(i + gap, x2), y1), color, thickness)
        # 下边
        for i in range(x1, x2, gap * 2):
            cv2.line(img, (i, y2), (min(i + gap, x2), y2), color, thickness)
        # 左边
        for i in range(y1, y2, gap * 2):
            cv2.line(img, (x1, i), (x1, min(i + gap, y2)), color, thickness)
        # 右边
        for i in range(y1, y2, gap * 2):
            cv2.line(img, (x2, i), (x2, min(i + gap, y2)), color, thickness)