"""
智能上下文压缩器
借鉴Claude Code的92%信息保留率算法
"""
import asyncio
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from app.core.llm.client import LLMClient
from app.core.memory.context_compressor import ContextCompressor


logger = logging.getLogger(__name__)


class IntelligentContextCompressor:
    """
    智能上下文压缩器
    目标：在压缩到目标token数的同时，保留92%以上的关键信息
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.base_compressor = ContextCompressor()
        
        # 压缩策略权重
        self.strategy_weights = {
            "importance_scoring": 0.35,
            "semantic_clustering": 0.25,
            "redundancy_removal": 0.20,
            "temporal_relevance": 0.20
        }
        
        # 重要性评分配置
        self.importance_config = {
            "time_decay_factor": 0.95,  # 时间衰减因子
            "entity_weight": 2.0,  # 实体权重
            "tool_call_weight": 3.0,  # 工具调用权重
            "error_weight": 4.0,  # 错误信息权重
            "user_mark_weight": 5.0,  # 用户标记权重
            "decision_weight": 3.5,  # 决策点权重
        }
        
        # 语义相似度阈值
        self.similarity_threshold = 0.85
        
        # 缓存
        self._embedding_cache = {}
        self._summary_cache = {}
        
    async def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int = 2000,
        min_retention_rate: float = 0.92
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        智能压缩消息列表
        
        Args:
            messages: 原始消息列表
            target_tokens: 目标token数
            min_retention_rate: 最小信息保留率
            
        Returns:
            (压缩后的消息, 压缩统计信息)
        """
        if not messages:
            return [], {"compression_ratio": 0, "retention_rate": 1.0}
        
        original_tokens = self._estimate_tokens(messages)
        
        # 如果已经小于目标，直接返回
        if original_tokens <= target_tokens:
            return messages, {
                "compression_ratio": 0,
                "retention_rate": 1.0,
                "original_tokens": original_tokens,
                "compressed_tokens": original_tokens
            }
        
        # Step 1: 重要性评分
        scored_messages = await self._score_importance(messages)
        
        # Step 2: 语义去重
        deduplicated = await self._semantic_deduplication(scored_messages)
        
        # Step 3: 智能分组和摘要
        compressed = await self._progressive_summarization(
            deduplicated, target_tokens
        )
        
        # Step 4: 验证信息保留率
        retention_rate = await self._calculate_retention_rate(
            messages, compressed
        )
        
        # 如果保留率不足，使用更保守的策略
        if retention_rate < min_retention_rate:
            compressed = await self._conservative_compression(
                messages, target_tokens
            )
            retention_rate = await self._calculate_retention_rate(
                messages, compressed
            )
        
        compressed_tokens = self._estimate_tokens(compressed)
        
        return compressed, {
            "compression_ratio": 1 - (compressed_tokens / original_tokens),
            "retention_rate": retention_rate,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "original_count": len(messages),
            "compressed_count": len(compressed),
            "method": "intelligent" if retention_rate >= min_retention_rate else "conservative"
        }
    
    async def _score_importance(self, messages: List[Dict[str, Any]]) -> List[Tuple[float, Dict]]:
        """
        计算每条消息的重要性分数
        """
        scored = []
        current_time = datetime.now()
        
        for i, msg in enumerate(messages):
            score = 0.0
            
            # 1. 时间衰减
            if "timestamp" in msg:
                msg_time = datetime.fromisoformat(msg["timestamp"])
                time_diff = (current_time - msg_time).total_seconds() / 3600  # 小时
                time_score = self.importance_config["time_decay_factor"] ** (time_diff / 24)
                score += time_score * 10
            
            # 2. 内容特征分析
            content = msg.get("content", "")
            
            # 实体密度（简化版，实际应使用NER）
            entities = self._extract_entities(content)
            entity_score = len(entities) * self.importance_config["entity_weight"]
            score += entity_score
            
            # 工具调用
            if msg.get("tool_calls") or "tool" in content.lower():
                score += self.importance_config["tool_call_weight"]
            
            # 错误信息
            if msg.get("error") or "error" in content.lower():
                score += self.importance_config["error_weight"]
            
            # 决策点
            if any(keyword in content.lower() for keyword in ["decide", "choose", "select", "plan"]):
                score += self.importance_config["decision_weight"]
            
            # 3. 结构重要性
            role = msg.get("role", "")
            if role == "system":
                score += 5  # 系统消息更重要
            elif role == "user":
                score += 3  # 用户消息重要
            
            # 4. 位置权重（首尾消息更重要）
            position_weight = 1.0
            if i < 3 or i >= len(messages) - 3:
                position_weight = 1.5
            score *= position_weight
            
            scored.append((score, msg))
        
        # 归一化分数
        max_score = max(s[0] for s in scored) if scored else 1.0
        normalized = [(s[0] / max_score, msg) for s, msg in scored]
        
        return sorted(normalized, key=lambda x: x[0], reverse=True)
    
    async def _semantic_deduplication(
        self, 
        scored_messages: List[Tuple[float, Dict]]
    ) -> List[Dict[str, Any]]:
        """
        语义级别去重，合并相似消息
        """
        if len(scored_messages) <= 1:
            return [msg for _, msg in scored_messages]
        
        # 提取消息内容
        contents = [msg.get("content", "") for _, msg in scored_messages]
        
        # 获取向量表示
        embeddings = await self._get_embeddings(contents)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 去重和合并
        unique_messages = []
        processed_indices = set()
        
        for i, (score, msg) in enumerate(scored_messages):
            if i in processed_indices:
                continue
            
            # 找到所有相似的消息
            similar_indices = []
            for j in range(i + 1, len(scored_messages)):
                if j not in processed_indices and similarity_matrix[i][j] > self.similarity_threshold:
                    similar_indices.append(j)
            
            if similar_indices:
                # 合并相似消息
                merged = await self._merge_similar_messages(
                    msg, 
                    [scored_messages[j][1] for j in similar_indices],
                    score,
                    [scored_messages[j][0] for j in similar_indices]
                )
                unique_messages.append(merged)
                processed_indices.update(similar_indices)
            else:
                unique_messages.append(msg)
            
            processed_indices.add(i)
        
        return unique_messages
    
    async def _progressive_summarization(
        self, 
        messages: List[Dict[str, Any]], 
        target_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        渐进式摘要
        """
        current_tokens = self._estimate_tokens(messages)
        
        if current_tokens <= target_tokens:
            return messages
        
        # 按主题分组
        groups = self._group_by_topic(messages)
        compressed_groups = []
        
        # 计算每组的目标token数
        group_targets = self._distribute_tokens(groups, target_tokens)
        
        for group, group_target in zip(groups, group_targets):
            if len(group) <= 2:
                # 小组不压缩
                compressed_groups.extend(group)
            else:
                # 对大组进行摘要
                summary = await self._summarize_group(group, group_target)
                compressed_groups.append(summary)
        
        return compressed_groups
    
    async def _summarize_group(
        self, 
        messages: List[Dict[str, Any]], 
        target_tokens: int
    ) -> Dict[str, Any]:
        """
        对一组消息进行摘要
        """
        # 检查缓存
        cache_key = self._get_cache_key(messages)
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]
        
        # 提取关键信息
        key_points = []
        tool_calls = []
        decisions = []
        errors = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # 提取工具调用
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])
            
            # 提取错误
            if msg.get("error"):
                errors.append(msg["error"])
            
            # 提取决策点
            if "decision" in content.lower() or "plan" in content.lower():
                decisions.append(content[:200])  # 截取前200字符
        
        # 使用LLM生成摘要
        summary_prompt = f"""
        请将以下对话摘要为不超过{target_tokens}个token的内容，保留所有关键信息：
        
        对话内容：
        {self._format_messages_for_summary(messages[:10])}  # 限制输入长度
        
        必须保留的信息：
        - 工具调用：{tool_calls[:5]}
        - 错误信息：{errors[:3]}
        - 决策点：{decisions[:3]}
        
        要求：
        1. 保留所有重要的事实、数字、决定
        2. 保持时间顺序
        3. 不要添加推断或解释
        4. 使用简洁的语言
        """
        
        summary_content = await self.llm_client.generate(
            prompt=summary_prompt,
            max_tokens=target_tokens
        )
        
        summary_msg = {
            "role": "assistant",
            "content": summary_content,
            "metadata": {
                "is_summary": True,
                "original_count": len(messages),
                "summarized_at": datetime.now().isoformat(),
                "key_points": key_points[:5],
                "tool_calls": tool_calls[:5],
                "errors": errors[:3]
            }
        }
        
        # 缓存结果
        self._summary_cache[cache_key] = summary_msg
        
        return summary_msg
    
    async def _calculate_retention_rate(
        self, 
        original: List[Dict[str, Any]], 
        compressed: List[Dict[str, Any]]
    ) -> float:
        """
        计算信息保留率
        """
        if not original or not compressed:
            return 0.0
        
        # 提取关键信息
        original_info = self._extract_key_information(original)
        compressed_info = self._extract_key_information(compressed)
        
        # 计算保留的信息比例
        retained = 0
        total = len(original_info)
        
        for info in original_info:
            if self._information_exists(info, compressed_info):
                retained += 1
        
        retention_rate = retained / total if total > 0 else 0.0
        
        # 考虑语义相似度
        if self.llm_client:
            semantic_score = await self._calculate_semantic_similarity(
                original, compressed
            )
            # 综合得分
            retention_rate = 0.7 * retention_rate + 0.3 * semantic_score
        
        return min(retention_rate, 1.0)
    
    async def _conservative_compression(
        self, 
        messages: List[Dict[str, Any]], 
        target_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        保守压缩策略（当智能压缩未达到保留率要求时使用）
        """
        # 保留最重要的消息
        scored = await self._score_importance(messages)
        
        compressed = []
        current_tokens = 0
        
        for score, msg in scored:
            msg_tokens = self._estimate_message_tokens(msg)
            if current_tokens + msg_tokens <= target_tokens:
                compressed.append(msg)
                current_tokens += msg_tokens
            elif score > 0.8:  # 只保留非常重要的消息
                # 截断消息内容
                truncated = self._truncate_message(msg, target_tokens - current_tokens)
                if truncated:
                    compressed.append(truncated)
                    break
        
        # 确保保留首尾消息
        if messages and messages[0] not in compressed:
            compressed.insert(0, messages[0])
        if messages and messages[-1] not in compressed:
            compressed.append(messages[-1])
        
        return compressed
    
    def _extract_entities(self, text: str) -> List[str]:
        """简化的实体提取"""
        # 这里使用正则表达式简化实现
        # 实际应使用NER模型
        entities = []
        
        # 提取大写词（可能是专有名词）
        entities.extend(re.findall(r'\b[A-Z][a-z]+\b', text))
        
        # 提取数字
        entities.extend(re.findall(r'\b\d+\.?\d*\b', text))
        
        # 提取引号内容
        entities.extend(re.findall(r'"([^"]*)"', text))
        
        return entities
    
    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本向量表示"""
        # 简化实现，使用TF-IDF
        # 实际应使用更先进的embedding模型
        vectorizer = TfidfVectorizer(max_features=100)
        embeddings = vectorizer.fit_transform(texts)
        return embeddings.toarray()
    
    async def _merge_similar_messages(
        self, 
        primary: Dict[str, Any], 
        similar: List[Dict[str, Any]],
        primary_score: float,
        similar_scores: List[float]
    ) -> Dict[str, Any]:
        """合并相似消息"""
        merged = primary.copy()
        
        # 合并元数据
        merged_metadata = merged.get("metadata", {})
        merged_metadata["merged_count"] = len(similar) + 1
        merged_metadata["merged_scores"] = [primary_score] + similar_scores
        
        # 合并工具调用
        all_tool_calls = merged.get("tool_calls", [])
        for msg in similar:
            if msg.get("tool_calls"):
                all_tool_calls.extend(msg["tool_calls"])
        
        if all_tool_calls:
            merged["tool_calls"] = all_tool_calls
        
        merged["metadata"] = merged_metadata
        
        return merged
    
    def _group_by_topic(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """按主题分组消息"""
        # 简化实现：按时间窗口分组
        groups = []
        current_group = []
        
        for msg in messages:
            if not current_group:
                current_group.append(msg)
            elif len(current_group) < 5:  # 每组最多5条
                current_group.append(msg)
            else:
                groups.append(current_group)
                current_group = [msg]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _distribute_tokens(
        self, 
        groups: List[List[Dict[str, Any]]], 
        target_tokens: int
    ) -> List[int]:
        """分配每组的目标token数"""
        if not groups:
            return []
        
        # 按组的重要性分配
        group_scores = []
        for group in groups:
            # 简化：使用组大小作为重要性
            group_scores.append(len(group))
        
        total_score = sum(group_scores)
        targets = []
        
        for score in group_scores:
            group_target = int((score / total_score) * target_tokens)
            targets.append(max(group_target, 50))  # 最少50 tokens
        
        return targets
    
    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """估算token数"""
        total = 0
        for msg in messages:
            total += self._estimate_message_tokens(msg)
        return total
    
    def _estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """估算单条消息的token数"""
        # 简化实现：假设平均每4个字符一个token
        content = str(message.get("content", ""))
        return len(content) // 4 + 10  # 加10作为元数据开销
    
    def _truncate_message(self, message: Dict[str, Any], max_tokens: int) -> Optional[Dict[str, Any]]:
        """截断消息到指定token数"""
        if max_tokens < 20:
            return None
        
        truncated = message.copy()
        content = truncated.get("content", "")
        
        # 估算需要保留的字符数
        max_chars = (max_tokens - 10) * 4
        
        if len(content) > max_chars:
            truncated["content"] = content[:max_chars] + "..."
            truncated["metadata"] = truncated.get("metadata", {})
            truncated["metadata"]["truncated"] = True
        
        return truncated
    
    def _get_cache_key(self, messages: List[Dict[str, Any]]) -> str:
        """生成缓存键"""
        content = "".join(str(msg.get("content", "")) for msg in messages)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """格式化消息用于摘要"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # 限制长度
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    def _extract_key_information(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取关键信息"""
        key_info = []
        
        for msg in messages:
            # 提取实体
            entities = self._extract_entities(msg.get("content", ""))
            
            # 提取工具调用
            tool_calls = msg.get("tool_calls", [])
            
            # 提取错误
            errors = [msg.get("error")] if msg.get("error") else []
            
            key_info.append({
                "entities": entities,
                "tool_calls": tool_calls,
                "errors": errors
            })
        
        return key_info
    
    def _information_exists(
        self, 
        info: Dict[str, Any], 
        info_list: List[Dict[str, Any]]
    ) -> bool:
        """检查信息是否存在于列表中"""
        for item in info_list:
            # 检查实体
            if set(info["entities"]) & set(item["entities"]):
                return True
            
            # 检查工具调用
            if info["tool_calls"] and item["tool_calls"]:
                if any(tc in item["tool_calls"] for tc in info["tool_calls"]):
                    return True
            
            # 检查错误
            if info["errors"] and item["errors"]:
                if any(err in item["errors"] for err in info["errors"]):
                    return True
        
        return False
    
    async def _calculate_semantic_similarity(
        self, 
        original: List[Dict[str, Any]], 
        compressed: List[Dict[str, Any]]
    ) -> float:
        """计算语义相似度"""
        # 简化实现
        original_text = " ".join(msg.get("content", "") for msg in original)
        compressed_text = " ".join(msg.get("content", "") for msg in compressed)
        
        if not original_text or not compressed_text:
            return 0.0
        
        # 使用TF-IDF计算相似度
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([original_text, compressed_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return float(similarity)