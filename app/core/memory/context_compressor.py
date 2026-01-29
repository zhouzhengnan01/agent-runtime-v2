"""
上下文压缩器 - 智能压缩长期上下文

压缩策略：
1. 摘要提取：将长对话压缩为摘要
2. 重要性筛选：保留重要信息，删除冗余
3. 层级压缩：近期详细，远期概要
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import hashlib
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """压缩级别"""
    NONE = "none"  # 不压缩
    LIGHT = "light"  # 轻度压缩（保留80%信息）
    MEDIUM = "medium"  # 中度压缩（保留50%信息）
    HEAVY = "heavy"  # 重度压缩（保留20%核心信息）
    SUMMARY = "summary"  # 摘要级别（只保留关键点）


@dataclass
class ContextSegment:
    """上下文片段"""
    content: str
    timestamp: datetime
    importance: float
    token_count: int
    compression_level: CompressionLevel
    metadata: Dict[str, Any]


class ContextCompressor:
    """
    上下文压缩器
    
    主要功能：
    1. 自动压缩超过阈值的上下文
    2. 分层压缩（时间越远压缩越多）
    3. 保留关键信息和转折点
    4. 支持增量压缩
    """
    
    def __init__(self, 
                 max_tokens: int = 8000,
                 compression_threshold: float = 0.8):
        """
        初始化压缩器
        
        Args:
            max_tokens: 最大token数限制
            compression_threshold: 触发压缩的阈值（占最大token的比例）
        """
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.trigger_tokens = int(max_tokens * compression_threshold)
        
        # 压缩配置
        self.time_windows = [
            (timedelta(hours=1), CompressionLevel.NONE),      # 1小时内：不压缩
            (timedelta(hours=6), CompressionLevel.LIGHT),     # 6小时内：轻度压缩
            (timedelta(days=1), CompressionLevel.MEDIUM),     # 1天内：中度压缩
            (timedelta(days=7), CompressionLevel.HEAVY),      # 7天内：重度压缩
            (None, CompressionLevel.SUMMARY)                  # 更早：只保留摘要
        ]
        
        # 重要性阈值
        self.importance_thresholds = {
            CompressionLevel.LIGHT: 0.3,
            CompressionLevel.MEDIUM: 0.5,
            CompressionLevel.HEAVY: 0.7,
            CompressionLevel.SUMMARY: 0.9
        }
    
    async def compress_context(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        压缩上下文
        
        Args:
            messages: 消息列表
            current_tokens: 当前token数（如果已知）
            
        Returns:
            (压缩后的消息列表, 压缩统计信息)
        """
        if not messages:
            return [], {"compressed": False}
        
        # 计算当前token数
        if current_tokens is None:
            current_tokens = self._estimate_tokens(messages)
        
        # 如果未超过阈值，不压缩
        if current_tokens < self.trigger_tokens:
            return messages, {
                "compressed": False,
                "original_tokens": current_tokens,
                "reason": "未达到压缩阈值"
            }
        
        # 分析消息重要性
        analyzed_messages = self._analyze_importance(messages)
        
        # 按时间窗口分组
        grouped_messages = self._group_by_time_window(analyzed_messages)
        
        # 对每个组应用压缩
        compressed_messages = []
        for time_window, level, group in grouped_messages:
            if level == CompressionLevel.NONE:
                compressed_messages.extend(group)
            else:
                compressed = await self._compress_group(group, level)
                compressed_messages.extend(compressed)
        
        # 计算压缩后的token数
        final_tokens = self._estimate_tokens(compressed_messages)
        
        # 如果还是超过限制，进行进一步压缩
        if final_tokens > self.max_tokens:
            compressed_messages = await self._aggressive_compress(
                compressed_messages, 
                self.max_tokens
            )
            final_tokens = self._estimate_tokens(compressed_messages)
        
        return compressed_messages, {
            "compressed": True,
            "original_tokens": current_tokens,
            "final_tokens": final_tokens,
            "compression_ratio": 1 - (final_tokens / current_tokens),
            "messages_before": len(messages),
            "messages_after": len(compressed_messages)
        }
    
    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """估算token数（简单实现，实际应使用tokenizer）"""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            else:
                total_chars += len(json.dumps(content))
        
        # 粗略估算：平均4个字符一个token
        return total_chars // 4
    
    def _analyze_importance(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        分析消息重要性
        
        重要性判断标准：
        1. 包含关键决策
        2. 任务转折点
        3. 错误和异常
        4. 用户明确要求
        5. 总结性内容
        """
        analyzed = []
        
        for i, msg in enumerate(messages):
            importance = 0.5  # 基础重要性
            content = str(msg.get("content", "")).lower()
            
            # 关键词检测
            if any(kw in content for kw in ["决定", "决策", "重要", "关键", "必须"]):
                importance += 0.2
            
            # 任务相关
            if any(kw in content for kw in ["任务", "目标", "计划", "执行"]):
                importance += 0.15
            
            # 错误和问题
            if any(kw in content for kw in ["错误", "失败", "问题", "异常"]):
                importance += 0.25
            
            # 总结性内容
            if any(kw in content for kw in ["总结", "结论", "结果", "完成"]):
                importance += 0.2
            
            # 用户消息通常更重要
            if msg.get("role") == "user":
                importance += 0.1
            
            # 第一条和最后几条消息更重要
            if i < 3 or i >= len(messages) - 3:
                importance += 0.1
            
            # 限制最大值
            importance = min(importance, 1.0)
            
            msg_copy = msg.copy()
            msg_copy["importance"] = importance
            msg_copy["original_index"] = i
            analyzed.append(msg_copy)
        
        return analyzed
    
    def _group_by_time_window(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Tuple[timedelta, CompressionLevel, List[Dict[str, Any]]]]:
        """按时间窗口分组消息"""
        now = datetime.now()
        grouped = []
        
        for window, level in self.time_windows:
            group = []
            
            for msg in messages:
                # 获取消息时间
                timestamp = msg.get("timestamp")
                if isinstance(timestamp, str):
                    msg_time = datetime.fromisoformat(timestamp)
                elif isinstance(timestamp, datetime):
                    msg_time = timestamp
                else:
                    # 如果没有时间戳，根据索引估算
                    hours_ago = (len(messages) - msg["original_index"]) * 0.5
                    msg_time = now - timedelta(hours=hours_ago)
                
                # 判断是否在当前时间窗口内
                if window is None:  # 最后一个窗口，包含所有剩余消息
                    group.append(msg)
                elif now - msg_time <= window:
                    group.append(msg)
            
            if group:
                grouped.append((window, level, group))
            
            # 从原列表中移除已分组的消息
            messages = [m for m in messages if m not in group]
        
        return grouped
    
    async def _compress_group(
        self,
        messages: List[Dict[str, Any]],
        level: CompressionLevel
    ) -> List[Dict[str, Any]]:
        """
        压缩一组消息
        
        Args:
            messages: 消息组
            level: 压缩级别
            
        Returns:
            压缩后的消息
        """
        if not messages:
            return []
        
        # 根据重要性阈值筛选
        threshold = self.importance_thresholds.get(level, 0.5)
        
        if level == CompressionLevel.SUMMARY:
            # 生成摘要
            return [await self._generate_summary(messages)]
        
        # 筛选重要消息
        important_messages = [
            msg for msg in messages 
            if msg.get("importance", 0.5) >= threshold
        ]
        
        # 如果筛选后太少，保留一些次重要的
        if len(important_messages) < max(1, len(messages) // 5):
            sorted_msgs = sorted(messages, key=lambda x: x.get("importance", 0), reverse=True)
            important_messages = sorted_msgs[:max(2, len(messages) // 4)]
        
        # 根据压缩级别处理消息内容
        compressed = []
        for msg in important_messages:
            compressed_msg = msg.copy()
            
            if level == CompressionLevel.HEAVY:
                # 重度压缩：只保留关键信息
                compressed_msg["content"] = self._extract_key_points(msg["content"])
                compressed_msg["compressed"] = True
            elif level == CompressionLevel.MEDIUM:
                # 中度压缩：缩短内容
                compressed_msg["content"] = self._shorten_content(msg["content"], 0.5)
                compressed_msg["compressed"] = True
            elif level == CompressionLevel.LIGHT:
                # 轻度压缩：去除冗余
                compressed_msg["content"] = self._remove_redundancy(msg["content"])
            
            compressed.append(compressed_msg)
        
        return compressed
    
    async def _generate_summary(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成消息组的摘要"""
        # 提取关键信息
        key_points = []
        for msg in messages:
            if msg.get("importance", 0) >= 0.6:
                point = self._extract_key_points(msg.get("content", ""))
                if point:
                    key_points.append(point)
        
        # 如果有LLM，可以调用LLM生成摘要
        # 这里使用简单的拼接
        summary_content = f"[历史摘要 - {len(messages)}条消息]\n"
        summary_content += "关键点：\n"
        for i, point in enumerate(key_points[:5], 1):  # 最多5个关键点
            summary_content += f"{i}. {point}\n"
        
        return {
            "role": "system",
            "content": summary_content,
            "is_summary": True,
            "original_count": len(messages),
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_key_points(self, content: str) -> str:
        """提取关键点（简化实现）"""
        if not content:
            return ""
        
        # 提取前100个字符或第一句话
        sentences = content.split("。")
        if sentences:
            return sentences[0][:100] + ("..." if len(sentences[0]) > 100 else "。")
        return content[:100] + ("..." if len(content) > 100 else "")
    
    def _shorten_content(self, content: str, ratio: float) -> str:
        """缩短内容"""
        if not content:
            return ""
        
        target_length = int(len(content) * ratio)
        if len(content) <= target_length:
            return content
        
        # 尝试在句子边界截断
        sentences = content.split("。")
        shortened = ""
        for sentence in sentences:
            if len(shortened) + len(sentence) <= target_length:
                shortened += sentence + "。"
            else:
                break
        
        return shortened if shortened else content[:target_length] + "..."
    
    def _remove_redundancy(self, content: str) -> str:
        """去除冗余信息"""
        if not content:
            return ""
        
        # 简单实现：去除重复的空格和换行
        content = " ".join(content.split())
        
        # 去除过多的标点
        for punct in ["。。", "，，", "！！", "？？"]:
            content = content.replace(punct, punct[0])
        
        return content
    
    async def _aggressive_compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        激进压缩：当常规压缩仍超过限制时使用
        
        策略：
        1. 只保留最重要的消息
        2. 生成整体摘要
        3. 保留最近的几条消息
        """
        # 按重要性排序
        sorted_msgs = sorted(
            messages, 
            key=lambda x: x.get("importance", 0), 
            reverse=True
        )
        
        # 保留最重要的消息
        keep_count = max(5, len(messages) // 10)  # 至少保留5条
        important = sorted_msgs[:keep_count]
        
        # 保留最近的消息
        recent = messages[-3:] if len(messages) > 3 else messages
        
        # 合并并去重
        final_messages = []
        seen_contents = set()
        
        # 先添加摘要
        summary = await self._generate_summary(messages)
        final_messages.append(summary)
        
        # 添加重要消息
        for msg in important + recent:
            content_hash = hashlib.md5(
                str(msg.get("content", "")).encode()
            ).hexdigest()
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                # 进一步压缩内容
                compressed_msg = msg.copy()
                compressed_msg["content"] = self._extract_key_points(msg["content"])
                final_messages.append(compressed_msg)
        
        return final_messages
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        return {
            "max_tokens": self.max_tokens,
            "trigger_tokens": self.trigger_tokens,
            "compression_threshold": self.compression_threshold,
            "compression_levels": [level.value for level in CompressionLevel],
            "time_windows": [
                {
                    "window": str(window) if window else "older",
                    "level": level.value
                }
                for window, level in self.time_windows
            ]
        }


# 使用LLM的高级压缩器
class LLMContextCompressor(ContextCompressor):
    """
    基于LLM的上下文压缩器
    使用大语言模型进行智能压缩和摘要生成
    """
    
    def __init__(self, llm_provider=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm_provider
    
    async def _generate_summary(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用LLM生成高质量摘要"""
        if not self.llm:
            # 如果没有LLM，使用父类的简单实现
            return await super()._generate_summary(messages)
        
        # 准备对话历史
        conversation = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"
            for msg in messages[-10:]  # 最近10条
        ])
        
        # 调用LLM生成摘要
        prompt = f"""
请将以下对话历史压缩为简洁的摘要，保留关键信息和决策点：

{conversation}

摘要要求：
1. 保留所有重要决策和结论
2. 记录关键任务和目标
3. 标注问题和解决方案
4. 字数控制在200字以内

摘要：
"""
        
        try:
            summary_text = await self.llm.generate(prompt, max_tokens=300)
            
            return {
                "role": "system",
                "content": f"[AI摘要 - {len(messages)}条历史消息]\n{summary_text}",
                "is_summary": True,
                "original_count": len(messages),
                "timestamp": datetime.now().isoformat(),
                "llm_generated": True
            }
        except Exception as e:
            logger.error(f"LLM摘要生成失败: {e}")
            return await super()._generate_summary(messages)