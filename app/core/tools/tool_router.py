"""
工具路由器 - 根据工具特性选择执行方式

功能：
1. 快速工具（<5秒）：同步执行，立即返回结果
2. 长时工具（≥5秒）：后台执行，流式返回部分结果
3. 支持并发执行多个后台任务

使用示例：
    router = ToolRouter(websocket_handler)
    result = await router.execute_tool(tool, params)
"""

from typing import Dict, Any, Optional, AsyncIterator
import asyncio
import uuid
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """后台任务信息"""
    task_id: str
    tool_name: str
    status: str  # submitted, running, completed, failed
    task: asyncio.Task
    created_at: datetime
    final_result: Any = None  # 任务的最终结果
    error: str = None          # 任务失败时的错误信息


class ToolRouter:
    """工具路由器 - 智能选择执行方式"""

    def __init__(self, websocket_handler=None):
        """
        Args:
            websocket_handler: WebSocket处理器，用于推送流式消息
        """
        self.websocket = websocket_handler
        self.background_tasks: Dict[str, TaskInfo] = {}
        self.threshold = 5.0  # 5秒以上视为长时任务

        # 待确认的外部工具执行（用于 WebSocket 工具确认流程）
        self.pending_confirmations: Dict[str, Dict[str, Any]] = {}

    async def execute_tool(self, tool, params: Dict[str, Any]) -> Any:
        """
        智能路由工具执行

        Args:
            tool: 工具实例
            params: 工具参数

        Returns:
            工具执行结果或任务ID
        """
        # 获取工具属性（如果不存在则使用默认值）
        estimated_time = getattr(tool, 'estimated_time', 1.0)
        support_streaming = getattr(tool, 'support_streaming', False)

        logger.info(f"🔀 工具路由: {tool.name} (预计{estimated_time}秒, 流式={support_streaming})")

        # 🆕 详细日志：记录工具调用详情
        if settings.ENABLE_DETAILED_LOGGING and settings.SHOW_TOOL_CALLS:
            self._log_tool_call_details(tool, params, "开始执行")

        # 判断：长时间 + 支持流式 → 后台执行
        if estimated_time >= self.threshold and support_streaming:
            return await self._execute_background(tool, params)
        else:
            # 短时任务，直接同步执行
            logger.info(f"⚡ 同步执行工具: {tool.name}")
            return await self._execute_sync(tool, params)

    async def _execute_sync(self, tool, params: Dict[str, Any]) -> Any:
        """同步执行工具"""
        try:
            # 记录工具开始执行
            logger.info(f"🚀 开始执行工具: {tool.name}")

            # 如果工具有异步call方法，使用它
            if hasattr(tool, 'acall'):
                result = await tool.acall(params)
            else:
                # 否则在线程池中执行同步call
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool.call(params))

            # 记录工具执行成功
            self._log_tool_result(tool, params, result, "执行成功")

            # 🆕 保存工具调用历史到Redis
            await self._save_tool_call_history(tool, params, result)

            return result

        except Exception as e:
            logger.error(f"❌ 工具执行失败: {tool.name}, 错误: {e}")

            # 🆕 详细日志：记录工具执行失败
            if settings.ENABLE_DETAILED_LOGGING:
                self._log_tool_result(tool, params, None, "执行失败", str(e))

            raise

    async def _execute_background(self, tool, params: Dict[str, Any]) -> Dict[str, str]:
        """后台流式执行工具"""
        task_id = str(uuid.uuid4())

        logger.info(f"📤 提交后台任务: {tool.name} (ID: {task_id})")

        # 通知前端：任务已提交
        if self.websocket:
            await self.websocket.send_message({
                "type": "task.submitted",
                "task_id": task_id,
                "tool_name": tool.name,
                "estimated_time": getattr(tool, 'estimated_time', 60)
            })

        # 创建后台任务（非阻塞）
        task = asyncio.create_task(
            self._stream_tool_results(task_id, tool, params)
        )

        self.background_tasks[task_id] = TaskInfo(
            task_id=task_id,
            tool_name=tool.name,
            status="running",
            task=task,
            created_at=datetime.now()
        )

        # 返回任务ID（不等待任务完成）
        return {
            "task_id": task_id,
            "status": "submitted",
            "message": f"后台任务已提交: {tool.name}"
        }

    async def _stream_tool_results(self, task_id: str, tool, params: Dict[str, Any]):
        """流式执行工具并推送结果"""
        final_result = None  # 保存最终结果

        try:
            logger.info(f"🔄 开始流式执行: {tool.name} (ID: {task_id})")

            # 调用工具的流式接口
            if hasattr(tool, 'call_stream'):
                async for chunk in tool.call_stream(params):
                    # 推送部分结果
                    if self.websocket:
                        await self.websocket.send_message({
                            "type": "tool.stream",
                            "task_id": task_id,
                            "tool_name": tool.name,
                            "chunk": chunk
                        })

                    logger.debug(f"📊 流式输出: {task_id} - {chunk.get('type', 'unknown')}")

                    # 保存最终结果
                    if chunk.get("type") == "final_result":
                        final_result = chunk.get("data")
            else:
                # 工具不支持流式，回退到同步执行
                logger.warning(f"⚠️ 工具 {tool.name} 不支持流式执行，回退到同步模式")
                result = await self._execute_sync(tool, params)
                final_result = result  # 保存最终结果

                # 发送最终结果
                if self.websocket:
                    await self.websocket.send_message({
                        "type": "tool.stream",
                        "task_id": task_id,
                        "tool_name": tool.name,
                        "chunk": {
                            "type": "final_result",
                            "data": result
                        }
                    })

            # 标记完成并保存结果
            if task_id in self.background_tasks:
                self.background_tasks[task_id].status = "completed"
                self.background_tasks[task_id].final_result = final_result

            logger.info(f"✅ 后台任务完成: {tool.name} (ID: {task_id})")

            # 发送完成消息
            if self.websocket:
                await self.websocket.send_message({
                    "type": "task.completed",
                    "task_id": task_id,
                    "tool_name": tool.name
                })

        except Exception as e:
            logger.error(f"❌ 后台任务失败: {tool.name} (ID: {task_id}), 错误: {e}")

            # 标记失败并保存错误
            if task_id in self.background_tasks:
                self.background_tasks[task_id].status = "failed"
                self.background_tasks[task_id].error = str(e)

            # 通知前端
            if self.websocket:
                await self.websocket.send_message({
                    "type": "task.failed",
                    "task_id": task_id,
                    "tool_name": tool.name,
                    "error": str(e)
                })

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """查询任务状态"""
        task_info = self.background_tasks.get(task_id)
        if not task_info:
            return None

        return {
            "task_id": task_info.task_id,
            "tool_name": task_info.tool_name,
            "status": task_info.status,
            "created_at": task_info.created_at.isoformat()
        }

    def get_running_tasks(self) -> list:
        """获取所有运行中的任务"""
        return [
            self.get_task_status(task_id)
            for task_id, task_info in self.background_tasks.items()
            if task_info.status == "running"
        ]

    async def cancel_task(self, task_id: str) -> bool:
        """取消后台任务"""
        task_info = self.background_tasks.get(task_id)
        if not task_info:
            return False

        if task_info.status == "running":
            task_info.task.cancel()
            task_info.status = "cancelled"
            logger.info(f"🛑 取消任务: {task_id}")
            return True

        return False

    def cleanup_finished_tasks(self, max_age_seconds: int = 3600):
        """清理已完成的任务（默认保留1小时）"""
        now = datetime.now()
        to_remove = []

        for task_id, task_info in self.background_tasks.items():
            if task_info.status in ["completed", "failed", "cancelled"]:
                age = (now - task_info.created_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self.background_tasks[task_id]

        if to_remove:
            logger.info(f"🧹 清理了 {len(to_remove)} 个过期任务")

    async def wait_task(self, task_id: str, timeout: float = 300) -> Any:
        """
        等待后台任务完成并返回结果

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒），默认5分钟

        Returns:
            任务的最终结果

        Raises:
            ValueError: 任务不存在
            RuntimeError: 任务失败或超时
        """
        task_info = self.background_tasks.get(task_id)
        if not task_info:
            raise ValueError(f"任务不存在: {task_id}")

        try:
            # 等待asyncio.Task完成（带超时）
            logger.info(f"⏳ 等待任务完成: {task_id} (超时: {timeout}秒)")
            await asyncio.wait_for(task_info.task, timeout=timeout)

            # 检查任务状态并返回结果
            if task_info.status == "completed":
                logger.info(f"✅ 任务完成，返回结果: {task_id}")
                return task_info.final_result
            elif task_info.status == "failed":
                error = task_info.error or 'Unknown error'
                logger.error(f"❌ 任务失败: {task_id}, 错误: {error}")
                raise RuntimeError(f"任务执行失败: {error}")
            else:
                logger.warning(f"⚠️ 任务状态异常: {task_id}, status={task_info.status}")
                raise RuntimeError(f"任务状态异常: {task_info.status}")

        except asyncio.TimeoutError:
            logger.error(f"⏰ 任务超时: {task_id} (超过 {timeout} 秒)")
            # 取消任务
            await self.cancel_task(task_id)
            raise RuntimeError(f"任务执行超时（{timeout}秒）")

    # ═══════════════════════════════════════════════════════════
    # 工具确认流程（区分内部/外部工具）
    # ═══════════════════════════════════════════════════════════

    async def execute_tool_with_confirmation(
        self,
        tool,
        params: Dict[str, Any],
        websocket_handler=None
    ) -> Any:
        """
        统一的工具执行入口（自动区分内部/外部工具）

        【核心逻辑】
        1. 使用 internal_tool_registry 判断工具类型
        2. 内部工具：直接执行（通过 execute_tool）
        3. 外部工具：发送确认消息 → 等待 WebSocket 执行 → 返回结果

        【工具类型判断】
        - 内部工具：在数据库 tools 表中，source='internal'
        - 外部工具：来自 JetLinks 平台，通过 WebSocket 传入

        Args:
            tool: 工具实例
            params: 工具参数
            websocket_handler: WebSocket处理器（外部工具需要）

        Returns:
            工具执行结果

        示例：
        ```python
        # 在 cognitive_agent.py 中调用
        result = await self.tool_router.execute_tool_with_confirmation(
            tool=tool,
            params=tool_params,
            websocket_handler=websocket_handler
        )
        ```
        """
        from app.core.tools import internal_tool_registry

        tool_name = getattr(tool, 'name', '')

        # 判断是否为内部工具
        if internal_tool_registry.is_internal_tool(tool_name):
            # ═══════════════════════════════════════════════════════
            # 内部工具：直接执行
            # ═══════════════════════════════════════════════════════
            logger.info(f"🔧 内部工具，直接执行: {tool_name}")
            return await self.execute_tool(tool, params)
        else:
            # ═══════════════════════════════════════════════════════
            # 外部工具：需要 WebSocket 确认流程
            # ═══════════════════════════════════════════════════════
            logger.info(f"🌐 外部工具，等待确认: {tool_name}")
            return await self._execute_external_tool(tool, params, websocket_handler)

    async def _execute_external_tool(
        self,
        tool,
        params: Dict[str, Any],
        websocket_handler
    ) -> Any:
        """
        执行外部工具（通过 WebSocket 确认流程）

        【流程】
        1. 生成唯一的 confirmation_id
        2. 创建 Future 用于等待结果
        3. 构建 tools.confirm 消息
        4. 通过 WebSocket 发送确认消息
        5. await Future（暂停等待）
        6. WebSocket 收到 tools.execute 后调用 complete_confirmation()
        7. Future 被设置结果，恢复执行
        8. 返回工具结果

        Args:
            tool: 工具实例
            params: 工具参数
            websocket_handler: WebSocket处理器

        Returns:
            工具执行结果

        Raises:
            TimeoutError: 工具执行超时（默认300秒）
            ValueError: WebSocket处理器未提供
        """
        if not websocket_handler:
            raise ValueError("外部工具需要 WebSocket 处理器，但未提供")

        confirmation_id = str(uuid.uuid4())

        # 创建等待 Future
        result_future = asyncio.Future()
        self.pending_confirmations[confirmation_id] = {
            "tool": tool,
            "params": params,
            "future": result_future,
            "created_at": time.time()
        }

        logger.info(
            f"📋 创建工具确认 | "
            f"ID: {confirmation_id} | "
            f"Tool: {tool.name}"
        )

        # 构建确认消息（符合 JAIP 协议）
        confirmation = {
            "jsonrpc": "2.0",
            "id": confirmation_id,
            "method": "tools.confirm",
            "params": {
                "name": f"需要调用工具：{tool.name}",
                "description": "操作待确认",
                "call": {
                    "toolId": tool.name,
                    "toolName": tool.name,
                    "arguments": params
                }
            }
        }

        # 通过 yield 返回确认消息（cognitive_agent 会 yield 给 WebSocket）
        # 注意：这里不能直接发送，需要返回给 cognitive_agent
        # cognitive_agent 会通过 yield confirmation 发送给 WebSocket
        # 所以这个方法需要改造...

        # 实际上，这个方法不应该直接发送，而是返回给 cognitive_agent
        # 让 cognitive_agent yield 出去
        # 所以我们需要另一个设计...

        logger.warning("⚠️ 外部工具执行流程需要重新设计")
        raise NotImplementedError("外部工具确认流程需要在 cognitive_agent 层面实现")

    def complete_confirmation(self, confirmation_id: str, result: Any):
        """
        完成工具确认（由 WebSocket 调用）

        【调用时机】
        WebSocket 收到 tools.execute 消息后：
        1. 执行外部工具（通过 SessionTool）
        2. 获取工具执行结果
        3. 调用此方法：tool_router.complete_confirmation(execution_id, result)
        4. 触发 Future，让 cognitive_agent 恢复执行

        Args:
            confirmation_id: 确认消息ID（对应 tools.confirm 的 id）
            result: 工具执行结果

        示例：
        ```python
        # 在 websocket.py 中
        if method == "tools.execute":
            execution_id = params.get("id")
            result = await execute_tool_directly(...)
            
            # 通知 tool_router 完成
            tool_router.complete_confirmation(execution_id, result)
        ```
        """
        if confirmation_id in self.pending_confirmations:
            future = self.pending_confirmations[confirmation_id]["future"]

            if not future.done():
                future.set_result(result)
                logger.info(f"✅ 工具确认完成: {confirmation_id}")
            else:
                logger.warning(f"⚠️ Future 已完成，忽略重复结果: {confirmation_id}")

            # 清理
            del self.pending_confirmations[confirmation_id]
        else:
            logger.warning(f"⚠️ 未找到待确认的工具: {confirmation_id}")

    async def wait_for_confirmation(self, confirmation_id: str, timeout: float = 300) -> Any:
        """
        等待外部工具确认完成（由 cognitive_agent 调用）

        【调用流程】
        1. cognitive_agent yield 确认消息（包含 confirmation_id）
        2. cognitive_agent 调用此方法等待结果
        3. WebSocket 收到 tools.execute 后执行工具
        4. WebSocket 调用 complete_confirmation() 设置结果
        5. 此方法返回结果给 cognitive_agent

        Args:
            confirmation_id: 确认消息ID（对应 tools.confirm 的 id）
            timeout: 超时时间（秒），默认300秒

        Returns:
            工具执行结果

        Raises:
            ValueError: 未找到待确认的工具
            asyncio.TimeoutError: 超时未收到确认

        示例：
        ```python
        # 在 cognitive_agent 中
        confirmation_id = f"tool_confirm_{int(time.time() * 1000)}"

        # 创建 Future 并存储
        future = asyncio.Future()
        tool_router.pending_confirmations[confirmation_id] = {
            "future": future,
            "tool_call": tool_call,
            "created_at": time.time()
        }

        # yield 确认消息
        yield {
            "type": "tools.confirm",
            "id": confirmation_id,
            "tool": tool_call
        }

        # 等待结果
        result = await tool_router.wait_for_confirmation(confirmation_id)
        ```
        """
        if confirmation_id not in self.pending_confirmations:
            raise ValueError(f"未找到待确认的工具: {confirmation_id}")

        future = self.pending_confirmations[confirmation_id]["future"]

        try:
            logger.info(f"⏳ 等待外部工具确认: {confirmation_id} (超时: {timeout}秒)")
            result = await asyncio.wait_for(future, timeout=timeout)
            logger.info(f"✅ 外部工具确认完成: {confirmation_id}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"❌ 外部工具确认超时: {confirmation_id}")
            # 清理超时的确认
            if confirmation_id in self.pending_confirmations:
                del self.pending_confirmations[confirmation_id]
            raise

    # ═══════════════════════════════════════════════════════════
    # 🆕 详细日志记录函数
    # ═══════════════════════════════════════════════════════════

    def _log_tool_call_details(self, tool, params: Dict[str, Any], action: str):
        """记录工具调用详情"""
        if not settings.LOG_TOOL_INPUTS:
            return

        tool_name = getattr(tool, 'name', 'Unknown')
        tool_description = getattr(tool, 'description', '')

        # 截断长文本
        max_length = settings.MAX_LOG_MESSAGE_LENGTH
        truncated_params = str(params)
        if len(truncated_params) > max_length:
            truncated_params = truncated_params[:max_length] + "...(truncated)"

        logger.info(f"🛠️ [工具调用] {action} | 工具: {tool_name}")
        logger.info(f"   📝 描述: {tool_description[:100]}...")
        logger.info(f"   📥 参数: {truncated_params}")

    def _log_tool_result(self, tool, params: Dict[str, Any], result: Any, status: str, error: str = None):
        """记录工具执行结果"""
        if not settings.LOG_TOOL_OUTPUTS:
            return

        tool_name = getattr(tool, 'name', 'Unknown')

        # 截断结果
        result_str = str(result) if result else "None"
        if len(result_str) > settings.MAX_LOG_MESSAGE_LENGTH:
            result_str = result_str[:settings.MAX_LOG_MESSAGE_LENGTH] + "...(truncated)"

        logger.info(f"✅ [工具结果] {tool_name} | 状态: {status}")
        if error:
            logger.error(f"   ❌ 错误: {error}")
        elif result:
            logger.info(f"   📤 结果: {result_str}")
        else:
            logger.info(f"   📤 结果: 空结果")

    def _log_execution_flow(self, message: str, level: str = "info"):
        """记录执行流程"""
        if not settings.SHOW_EXECUTION_FLOW:
            return

        if level == "info":
            logger.info(f"🔄 [执行流程] {message}")
        elif level == "debug":
            logger.debug(f"🔄 [执行流程] {message}")
        elif level == "warning":
            logger.warning(f"⚠️ [执行流程] {message}")
        elif level == "error":
            logger.error(f"❌ [执行流程] {message}")

    async def _save_tool_call_history(self, tool, params: Dict[str, Any], result: Any):
        """保存工具调用历史到Redis"""
        try:
            # 获取session和agent信息（从上下文或全局）
            # 这里需要通过其他方式获取，因为tool_router没有直接的context
            # 暂时使用一个全局存储的方式
            import json
            from datetime import datetime

            tool_call_record = {
                "tool_name": getattr(tool, 'name', 'unknown'),
                "arguments": params,
                "result": str(result)[:500] if result else "",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }

            # 尝试保存到Redis（如果可用）
            try:
                from app.core.memory.redis_chat_memory import redis_memory_manager
                # 这里需要session_id，暂时跳过保存
                # 可以通过外部调用传入context来完善
                logger.debug(f"🔧 [工具历史] 准备保存: {tool_call_record['tool_name']}")
            except Exception as redis_error:
                logger.debug(f"🔧 [工具历史] Redis保存跳过: {redis_error}")

        except Exception as e:
            # 保存历史失败不影响主流程
            logger.debug(f"🔧 [工具历史] 保存失败: {e}")
