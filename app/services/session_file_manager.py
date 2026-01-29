"""
会话文件管理器

提供高级的会话级别文件管理服务，整合存储架构组件，支持：
- 会话文件的自动组织和管理
- 多媒体内容处理和转换
- 对话历史的持久化
- 文件检索和分析
- 会话级别的文件清理
"""

import os
import json
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from app.core.storage.session_storage import SessionStorageManager
from app.core.storage.file_organizer import FileOrganizer
from app.core.storage.markdown_templates import MarkdownTemplates

logger = logging.getLogger(__name__)


class SessionFileManager:
    """会话文件管理器 - 高级文件管理服务"""

    def __init__(self, base_storage_path: str = "storage/sessions"):
        """
        初始化会话文件管理器

        Args:
            base_storage_path: 基础存储路径
        """
        self.storage_manager = SessionStorageManager(base_storage_path)
        self.file_organizer = FileOrganizer()
        self.markdown_templates = MarkdownTemplates()

        logger.info(f"✅ 会话文件管理器已初始化: {base_storage_path}")

    async def save_uploaded_file(
        self,
        agent_id: str,
        session_id: str,
        file_path: Union[str, Path],
        file_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        保存上传的文件到会话存储

        Args:
            agent_id: 智能体ID
            session_id: 会话ID
            file_path: 上传的文件路径
            file_metadata: 文件元数据

        Returns:
            文件保存结果
        """
        try:
            # 创建会话目录
            session_path = self.storage_manager.create_session_directory(agent_id, session_id)

            # 使用文件组织器整理文件
            organize_result = self.file_organizer.organize_file(
                source_path=file_path,
                target_directory=session_path / "files",
                timestamp_prefix=datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            # 补充会话相关的元数据
            if file_metadata:
                organize_result['metadata'].update(file_metadata)

            organize_result['metadata'].update({
                "agent_id": agent_id,
                "session_id": session_id,
                "saved_at": datetime.now().isoformat()
            })

            # 更新元数据文件
            metadata_file = Path(organize_result['metadata_path'])
            self.file_organizer._save_metadata(metadata_file, organize_result['metadata'])

            logger.info(f"📁 会话文件已保存: {agent_id}/{session_id}")

            return {
                "success": True,
                "file_id": organize_result['timestamp_dir'],
                "file_path": organize_result['target_path'],
                "metadata": organize_result['metadata'],
                "session_path": str(session_path)
            }

        except Exception as e:
            logger.error(f"❌ 会话文件保存失败: {agent_id}/{session_id}, 错误: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def save_conversation_turn(
        self,
        agent_id: str,
        session_id: str,
        conversation_data: Dict[str, Any]
    ) -> str:
        """
        保存单轮对话记录

        Args:
            agent_id: 智能体ID
            session_id: 会话ID
            conversation_data: 对话数据

        Returns:
            保存的文件路径
        """
        try:
            # 保存对话记录
            file_path = self.storage_manager.save_conversation(
                agent_id=agent_id,
                session_id=session_id,
                conversation_data=conversation_data
            )

            logger.info(f"💬 对话记录已保存: {agent_id}/{session_id}")
            return file_path

        except Exception as e:
            logger.error(f"❌ 对话记录保存失败: {agent_id}/{session_id}, 错误: {e}")
            raise

    async def generate_conversation_markdown(
        self,
        agent_id: str,
        session_id: str,
        date_filter: Optional[str] = None
    ) -> Optional[str]:
        """
        生成对话历史的Markdown文档

        Args:
            agent_id: 智能体ID
            session_id: 会话ID
            date_filter: 日期过滤器 (YYYYMMDD格式)，None表示所有日期

        Returns:
            Markdown文档路径，失败时返回None
        """
        try:
            session_path = self.storage_manager.get_session_path(agent_id, session_id)
            conversations_dir = session_path / "conversations" / "daily"

            if not conversations_dir.exists():
                logger.warning(f"⚠️ 对话目录不存在: {conversations_dir}")
                return None

            # 收集对话记录
            all_conversations = []
            pattern = f"{date_filter}.json" if date_filter else "*.json"

            for conv_file in conversations_dir.glob(pattern):
                try:
                    conv_content = self.storage_manager.load_file(
                        agent_id, session_id, f"conversations/daily/{conv_file.name}"
                    )
                    if conv_content:
                        conversations = json.loads(conv_content)
                        all_conversations.extend(conversations)
                except Exception as e:
                    logger.warning(f"⚠️ 对话文件读取失败: {conv_file}, 错误: {e}")

            if not all_conversations:
                logger.info(f"📝 没有找到对话记录: {agent_id}/{session_id}")
                return None

            # 按时间戳排序
            all_conversations.sort(key=lambda x: x.get('timestamp', ''))

            # 获取会话信息
            session_summary = self.storage_manager.get_session_summary(agent_id, session_id)

            # 生成Markdown
            markdown_content = self.markdown_templates.generate_conversation_markdown(
                conversations=all_conversations,
                session_info={
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "created_at": session_summary.get('created_at'),
                    "conversation_count": len(all_conversations)
                }
            )

            # 保存Markdown文件
            markdown_filename = f"conversations/conversation_{date_filter if date_filter else 'all'}.md"
            markdown_path = self.storage_manager.save_file(
                agent_id, session_id, markdown_filename, markdown_content
            )

            logger.info(f"📄 对话Markdown已生成: {markdown_path}")
            return markdown_path

        except Exception as e:
            logger.error(f"❌ 对话Markdown生成失败: {agent_id}/{session_id}, 错误: {e}")
            return None

    async def search_session_files(
        self,
        agent_id: str,
        session_id: str,
        category: Optional[str] = None,
        file_type: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        搜索会话中的文件

        Args:
            agent_id: 智能体ID
            session_id: 会话ID
            category: 文件类别过滤 (images, videos, audio, documents)
            file_type: 文件类型过滤
            date_range: 日期范围过滤 (start_date, end_date)
            limit: 最大返回数量

        Returns:
            匹配的文件列表
        """
        try:
            session_path = self.storage_manager.get_session_path(agent_id, session_id)
            files_dir = session_path / "files"

            if not files_dir.exists():
                return []

            # 使用文件组织器搜索
            results = []

            # 按类别搜索
            if category:
                category_files = self.file_organizer.find_files_by_category(
                    search_directory=files_dir,
                    category=category,
                    limit=limit
                )
                results.extend(category_files)

            # 按时间范围搜索
            elif date_range:
                time_files = self.file_organizer.find_files_by_timerange(
                    search_directory=files_dir,
                    start_time=date_range[0],
                    end_time=date_range[1]
                )
                results.extend(time_files)

            # 搜索所有文件
            else:
                for metadata_file in files_dir.rglob("metadata.json"):
                    try:
                        metadata = self.file_organizer._load_metadata(metadata_file)
                        if metadata:
                            results.append({
                                "metadata": metadata,
                                "metadata_path": str(metadata_file),
                                "organized_at": metadata.get('organized_at'),
                                "timestamp_dir": metadata.get('timestamp_dir')
                            })
                    except Exception as e:
                        logger.warning(f"⚠️ 元数据读取失败: {metadata_file}, 错误: {e}")

            # 按文件类型过滤
            if file_type:
                results = [r for r in results if r.get('metadata', {}).get('file_type') == file_type]

            # 按时间排序并限制数量
            results.sort(key=lambda x: x.get('organized_at', ''), reverse=True)
            results = results[:limit]

            logger.info(f"🔍 会话文件搜索完成: {agent_id}/{session_id}, 找到 {len(results)} 个文件")
            return results

        except Exception as e:
            logger.error(f"❌ 会话文件搜索失败: {agent_id}/{session_id}, 错误: {e}")
            return []

    async def generate_session_summary_markdown(
        self,
        agent_id: str,
        session_id: str
    ) -> Optional[str]:
        """
        生成会话总结的Markdown文档

        Args:
            agent_id: 智能体ID
            session_id: 会话ID

        Returns:
            Markdown文档路径，失败时返回None
        """
        try:
            # 获取会话摘要
            session_summary = self.storage_manager.get_session_summary(agent_id, session_id)
            if not session_summary.get('exists'):
                return None

            # 收集统计信息
            conversations_dir = self.storage_manager.get_session_path(agent_id, session_id) / "conversations" / "daily"

            statistics = {
                "total_messages": 0,
                "tool_calls": 0,
                "files_processed": 0
            }

            main_topics = []
            tools_used = {}

            # 分析对话文件
            for conv_file in conversations_dir.glob("*.json"):
                try:
                    conv_content = self.storage_manager.load_file(
                        agent_id, session_id, f"conversations/daily/{conv_file.name}"
                    )
                    if conv_content:
                        conversations = json.loads(conv_content)

                        for conv in conversations:
                            data = conv.get('data', {})
                            statistics["total_messages"] += 1

                            if 'tool_calls' in data:
                                for tool_call in data['tool_calls']:
                                    tool_name = tool_call.get('tool_name', 'Unknown')
                                    tools_used[tool_name] = tools_used.get(tool_name, 0) + 1
                                    statistics["tool_calls"] += 1

                except Exception as e:
                    logger.warning(f"⚠️ 对话文件分析失败: {conv_file}, 错误: {e}")

            # 统计文件数量
            file_results = await self.search_session_files(agent_id, session_id)
            statistics["files_processed"] = len(file_results)

            # 构建总结数据
            summary_data = {
                "session_id": session_id,
                "agent_id": agent_id,
                "start_time": session_summary.get('created_at'),
                "end_time": session_summary.get('last_updated'),
                "statistics": statistics,
                "main_topics": main_topics,
                "tools_used": tools_used
            }

            # 生成关键交互（简化版本）
            key_interactions = [
                {
                    "timestamp": session_summary.get('created_at', ''),
                    "description": "会话开始",
                    "importance": "normal"
                }
            ]

            # 生成Markdown
            markdown_content = self.markdown_templates.generate_session_summary_markdown(
                session_summary=summary_data,
                key_interactions=key_interactions
            )

            # 保存Markdown文件
            markdown_path = self.storage_manager.save_file(
                agent_id, session_id, "conversations/summary.md", markdown_content
            )

            logger.info(f"📊 会话总结Markdown已生成: {markdown_path}")
            return markdown_path

        except Exception as e:
            logger.error(f"❌ 会话总结Markdown生成失败: {agent_id}/{session_id}, 错误: {e}")
            return None

    async def cleanup_old_sessions(
        self,
        days_threshold: int = 30,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        清理过期的会话数据

        Args:
            days_threshold: 保留天数阈值
            dry_run: 是否为演习模式（不实际删除）

        Returns:
            清理结果统计
        """
        try:
            base_path = self.storage_manager.base_path
            cutoff_date = datetime.now() - timedelta(days=days_threshold)

            cleanup_stats = {
                "scanned_sessions": 0,
                "expired_sessions": 0,
                "cleaned_sessions": 0,
                "freed_space_bytes": 0,
                "errors": []
            }

            # 扫描所有会话
            for agent_dir in base_path.iterdir():
                if not agent_dir.is_dir():
                    continue

                for session_dir in agent_dir.iterdir():
                    if not session_dir.is_dir():
                        continue

                    cleanup_stats["scanned_sessions"] += 1

                    # 检查会话创建时间
                    index_file = session_dir / "conversations" / "index.json"
                    if index_file.exists():
                        try:
                            with open(index_file, 'r', encoding='utf-8') as f:
                                index_data = json.load(f)

                            created_at = index_data.get('created_at')
                            if created_at:
                                session_date = datetime.fromisoformat(created_at)

                                if session_date < cutoff_date:
                                    cleanup_stats["expired_sessions"] += 1

                                    # 计算目录大小
                                    dir_size = sum(
                                        f.stat().st_size
                                        for f in session_dir.rglob('*')
                                        if f.is_file()
                                    )
                                    cleanup_stats["freed_space_bytes"] += dir_size

                                    if not dry_run:
                                        shutil.rmtree(session_dir)
                                        cleanup_stats["cleaned_sessions"] += 1
                                        logger.info(f"🗑️ 已清理过期会话: {agent_dir.name}/{session_dir.name}")

                        except Exception as e:
                            error_msg = f"处理会话失败: {agent_dir.name}/{session_dir.name}, 错误: {e}"
                            cleanup_stats["errors"].append(error_msg)
                            logger.warning(f"⚠️ {error_msg}")

            if dry_run:
                logger.info(f"🔍 会话清理演习完成，发现 {cleanup_stats['expired_sessions']} 个过期会话")
            else:
                logger.info(f"🗑️ 会话清理完成，清理了 {cleanup_stats['cleaned_sessions']} 个过期会话")

            return cleanup_stats

        except Exception as e:
            logger.error(f"❌ 会话清理失败: {e}")
            return {"error": str(e)}

    async def get_session_analytics(
        self,
        agent_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        获取会话分析数据

        Args:
            agent_id: 智能体ID
            session_id: 会话ID

        Returns:
            会话分析数据
        """
        try:
            # 基础会话信息
            session_summary = self.storage_manager.get_session_summary(agent_id, session_id)

            # 文件存储统计
            file_summary = self.file_organizer.get_storage_summary(
                self.storage_manager.get_session_path(agent_id, session_id) / "files"
            )

            # 搜索各类文件
            image_files = await self.search_session_files(agent_id, session_id, category="images")
            video_files = await self.search_session_files(agent_id, session_id, category="videos")
            document_files = await self.search_session_files(agent_id, session_id, category="documents")

            analytics = {
                "session_info": session_summary,
                "file_storage": file_summary,
                "file_breakdown": {
                    "images": len(image_files),
                    "videos": len(video_files),
                    "documents": len(document_files),
                    "total_files": session_summary.get('file_count', 0)
                },
                "storage_efficiency": {
                    "total_size_mb": file_summary.get('total_size_bytes', 0) / 1024 / 1024,
                    "avg_file_size_kb": (
                        file_summary.get('total_size_bytes', 0) / 1024 / max(session_summary.get('file_count', 1), 1)
                    )
                }
            }

            logger.info(f"📊 会话分析数据已生成: {agent_id}/{session_id}")
            return analytics

        except Exception as e:
            logger.error(f"❌ 会话分析失败: {agent_id}/{session_id}, 错误: {e}")
            return {"error": str(e)}


# 全局会话文件管理器实例
session_file_manager = SessionFileManager()