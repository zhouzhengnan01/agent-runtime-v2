"""
JAIP 协议处理器

职责：
1. 会话管理（初始化、消息处理）
2. 工具调用流程管理（确认、执行、结果处理）
3. JAIP 协议消息格式封装
4. Agent 交互
"""

import logging
import json
import os
import time
import uuid
import random
from typing import Dict, Any, Optional, AsyncIterator, Tuple

from sqlalchemy.orm import Session

from app.core.agents.template_agent_factory import TemplateAgentFactory
from app.core.tools.session_tool_manager import session_tool_manager
from app.services.agent_service import AgentService
from app.shared.jetlinks_video.utils.logger_utils import get_logger

logger = get_logger(__file__)


class JAIPHandler:
    """JAIP 协议处理器"""

    def __init__(self, websocket_manager=None, client_id=None):
        """初始化handler

        Args:
            websocket_manager: WebSocket连接管理器（用于发送消息给前端）
            client_id: 客户端ID（用于标识WebSocket连接）
        """
        self.agent_factory = TemplateAgentFactory()
        self.agent = None
        self.system_prompt = "你是一个智能助手，使用提供的工具和信息来回答用户的问题。"

        # WebSocket 相关
        self.ws_manager = websocket_manager
        self.client_id = client_id
        self.session_id = str(uuid.uuid4())

        # 工具调用流程管理
        self._tool_contexts = {}  # {execution_id: {toolName, arguments, user_query, toolType}}
        self._pending_tools = {}  # {execution_id: {toolName, arguments, user_query, toolType, result_id}}
        self._pending_tool_results = {}  # {random_tool_id: execution_id}

        # 生成器管理（支持暂停/恢复）
        self._active_generators = {}  # {session_id: generator_object}

        # 流式 chunk 合并（降低 WebSocket 小包频率，避免高频输出导致断链）
        # key: (session_id, response_id)
        self._chunk_buffers: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 内部：安全发送（若断链，立刻清理并抛异常中断上游循环）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    async def _safe_send(self, payload: Dict[str, Any]) -> None:
        ok = True
        try:
            ok = await self.ws_manager.send_message(self.client_id, payload)
        except Exception:
            ok = False

        if not ok:
            # send_message 内部会触发 disconnect hook（stop 下游 + 清理 storage）
            # 这里再兜底一次，保证 handler 自己也停掉生成器
            try:
                await self.cleanup_async(reason="send failed / ws disconnected")
            except Exception:
                pass
            raise ConnectionError("WebSocket disconnected")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 会话管理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def handle_initialize(
        self,
        params: Dict[str, Any],
        request_id: str,
        agent_id: str,
        session_id: str,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """处理 session.initialize 请求"""
        own_session = False
        if db is None:
            from app.db.session import SessionLocal

            db = SessionLocal()
            own_session = True

        protocol_version = params.get("protocolVersion", "2.0")
        new_session_id = session_id  # 使用传入的session_id
        available_tools = params.get("availableTools", [])
        user_context = params.get("userContext", {})
        init_params = params.get("parameters", {})

        logger.info(f"🔍 [session.initialize] init_params: {init_params}")

        try:
            # 从数据库加载 Agent 配置
            agent_service = AgentService(db)
            agent_config = agent_service.get_agent(agent_id)

            # DB 可能在容器启动时尚未就绪，导致启动阶段的 create_all/seed 失败；
            # 这里做一次“按需自举”，尽量保证默认智能体可用（幂等）。
            if not agent_config:
                logger.warning(f"⚠️ Agent {agent_id} not found, trying DB bootstrap/seed once...")
                try:
                    from app.db.session import engine, Base

                    Base.metadata.create_all(bind=engine)
                except Exception as e:
                    logger.warning(f"⚠️ DB create_all failed during initialize: {e}")

                try:
                    from app.db.seed_defaults import seed_default_templates_and_agents

                    seed_default_templates_and_agents()
                except Exception as e:
                    logger.warning(f"⚠️ Default seed failed during initialize: {e}")

                try:
                    db.rollback()
                except Exception:
                    pass

                agent_config = agent_service.get_agent(agent_id)

            if not agent_config:
                raise ValueError(f"Agent {agent_id} not found")

            config = agent_config.config if agent_config and agent_config.config else {}
            system_prompt = (config.get("system_prompt") or config.get("prompt") or "").strip()
            template_id = (config.get("template") or config.get("type") or agent_config.type or "tool_calling")
            model = config.get("model", "qwen-turbo")
            temperature = config.get("temperature", 0.3)
            max_tokens = config.get("max_tokens", 2048)

            result = await self.initialize_session(
                session_id=new_session_id,
                agent_config={
                    "name": agent_config.name,
                    "type": template_id,
                    "model": model,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "id": agent_id,
                    "init_parameters": init_params
                },
                available_tools=available_tools,
                client_session_id=new_session_id
            )

            self.session_id = new_session_id
            return {
                "sessionId": new_session_id,
                "agentId": agent_id,
                "protocolVersion": protocol_version,
                **result
            }
        finally:
            if own_session:
                try:
                    db.close()
                except Exception:
                    pass

    async def initialize_session(
        self,
        session_id: str,
        agent_config: Dict[str, Any],
        available_tools: list = None,
        client_session_id: str = None,
    ) -> Dict[str, Any]:
        """初始化会话，使用工厂创建Agent（内部方法）"""
        actual_session_id = client_session_id or session_id
        tool_instances = []

        # 注册工具（来自JetLinks平台）
        if available_tools:
            session_tool_manager.register_session_tools(
                actual_session_id,
                available_tools,
                ws_handler=self
            )
            external_tools = session_tool_manager.get_session_tools(actual_session_id)
            tool_instances.extend(external_tools.values())
            logger.info(f"  ✅ [外部工具] 注册完成，共 {len(external_tools)} 个")

        # 加载内部工具（从TOOL_REGISTRY）
        from app.core.tools.base import TOOL_REGISTRY
        internal_tool_names = agent_config.get("tools", [])
        if isinstance(internal_tool_names, list) and len(internal_tool_names) > 0:
            for tool_name in internal_tool_names:
                tool_class = TOOL_REGISTRY.get(tool_name)
                if tool_class:
                    tool_instance = tool_class()
                    tool_instances.append(tool_instance)
                    logger.info(f"  ✅ [内部工具] 加载: {tool_name}")
                else:
                    logger.warning(f"  ⚠️ [内部工具] 未找到: {tool_name} (可用工具: {list(TOOL_REGISTRY.keys())})")
            logger.info(f"  ✅ [内部工具] 加载完成，共 {len([t for t in internal_tool_names if t in TOOL_REGISTRY])} 个")

        template = agent_config.get("type") or agent_config.get("template", "tool_calling")

        config = {
            "name": agent_config.get("name", "LangChain智能体"),
            "template": template,
            "model": agent_config.get("model", "qwen-turbo"),
            "system_prompt": agent_config.get("system_prompt", "你是一个智能助手"),
            "temperature": agent_config.get("temperature", 0.3),
            "max_tokens": agent_config.get("max_tokens", 2000),
            "tools": tool_instances,
            "handler": self,
            "enable_memory": True,
            "session_id": actual_session_id,
            "agent_id": agent_config.get("id"),
            "init_parameters": agent_config.get("init_parameters", {}),
            "enable_compression": False,
            "enable_multimodal": False
        }
        self.system_prompt = agent_config.get("system_prompt", "你是一个智能助手")

        try:
            agent = self.agent_factory.create_agent(config)
            self.agent = agent
            return {
                "status": "initialized",
                "agent_info": {
                    "name": config["name"],
                    "template": template,
                    "model": config["model"],
                    "tool_count": len(tool_instances)
                }
            }
        except Exception as e:
            logger.error(f"❌ Agent创建失败: {e}", exc_info=True)
            if available_tools:
                session_tool_manager.clear_session_tools(actual_session_id)
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 机器视觉任务：ComputerVisionTask
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    async def handle_computer_vision_task(
        self,
        params: Dict[str, Any],
        request_id: Any,
        agent_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        处理机器视觉 JSON-RPC 请求（ComputerVisionTask）

        - 将 JSON-RPC 包装后的 original_params 透传给 VideoPatrolAgent
        - 按用户要求的格式，将流式结果封装为 JSON-RPC 响应发送
        """
        if not self.agent:
            await self._send_cv_error(request_id, "Agent未初始化，请先调用 session.initialize")
            return {"handled": False, "error": "agent_not_initialized", "suppress_ack": True}

        original_params = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "ComputerVisionTask",
            "params": params,
            "_ws_client_id": self.client_id,  # ✅ 透传给 VideoPatrolAgent
        }

        try:
            async for chunk in self.process_message_stream(
                session_id=session_id,
                message="",
                context={},
                params=original_params
            ):
                await self._dispatch_computer_vision_chunk(chunk, request_id)
        except ConnectionError:
            # ws 断开：已触发 stop/cleanup，静默结束
            return {"handled": True, "suppress_ack": True}
        except Exception as e:
            logger.error(f"❌ [ComputerVisionTask] 处理异常: {e}", exc_info=True)
            try:
                await self._send_cv_error(request_id, f"ComputerVisionTask 处理失败: {e}")
            except Exception:
                pass
            return {"handled": False, "error": str(e), "suppress_ack": True}

        return {"handled": True, "suppress_ack": True}

    async def _dispatch_computer_vision_chunk(self, chunk: Any, request_id: Any) -> None:
        """处理 VideoPatrolAgent 返回的流式 chunk，并发送符合协议的 JSON-RPC 响应"""
        data: Optional[Dict[str, Any]] = None

        if isinstance(chunk, str):
            try:
                data = json.loads(chunk)
            except Exception:
                logger.warning(f"⚠️ [ComputerVisionTask] 无法解析chunk为JSON: {chunk[:200]}")
                return
        elif isinstance(chunk, dict):
            data = chunk
        else:
            logger.debug(f"[ComputerVisionTask] 忽略未知类型chunk: {type(chunk)}")
            return

        if not isinstance(data, dict):
            logger.debug("[ComputerVisionTask] 解析后的数据不是dict，忽略")
            return

        # 错误分支
        if "error" in data:
            err = data.get("error") or {}
            message = err.get("message") if isinstance(err, dict) else str(err)
            await self._send_cv_error(request_id, message or "任务执行失败")
            return

        payload = data.get("params") or {}
        if not isinstance(payload, dict):
            logger.debug(f"[ComputerVisionTask] payload 非 dict，忽略: {payload}")
            return

        # 停止任务的确认：result true
        if "message" in payload and not any(k in payload for k in ("sourceId", "evidence_image_urls", "evidence_image_box_urls", "full_text")):
            await self._safe_send({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": True
            })
            return

        # 正常事件结果
        result_body = {
            "sourceId": payload.get("sourceId"),
            "evidence_image_urls": payload.get("evidence_image_urls") or [],
            "evidence_image_box_urls": payload.get("evidence_image_box_urls") or [],
            "full_text": payload.get("full_text") or []
        }

        await self._safe_send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result_body
        })

    async def _send_cv_error(self, request_id: Any, message: str) -> None:
        """发送机器视觉任务错误"""
        await self._safe_send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"message": message}
        })

    async def handle_message(
        self,
        params: Dict[str, Any],
        request_id: str,
        session_id: str
    ):
        """处理 session.message 请求（流式输出）"""
        logger.info(f"[handle_message] 接收到消息: {params}")

        message, message_type, message_context, _ = self._extract_message_content(params)
        msg_session_id = session_id

        # 兼容 session.message 不同入参格式：确保 original_params 同时包含 sessionId/content/context
        params_for_agent = dict(params) if isinstance(params, dict) else {}
        params_for_agent.setdefault("sessionId", msg_session_id)

        if "content" not in params_for_agent:
            msg_obj = params_for_agent.get("message")
            if isinstance(msg_obj, dict) and isinstance(msg_obj.get("content"), str):
                params_for_agent["content"] = msg_obj.get("content")

        if "content" not in params_for_agent:
            params_for_agent["content"] = message

        if "context" not in params_for_agent or params_for_agent.get("context") is None:
            params_for_agent["context"] = message_context or {}

        # ✅ 透传 client_id 给 VideoPatrolAgent（用于断链 stop/cleanup）
        params_for_agent["_ws_client_id"] = self.client_id

        if not self._validate_message(message, message_context):
            logger.debug("消息验证失败，跳过处理")
            return

        response_id = f"resp_{uuid.uuid4().hex[:8]}"
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        sent_chunks = 0
        response_start_sent = False

        try:
            async for item in self.process_message_stream(
                session_id=msg_session_id,
                message=message,
                context=message_context,
                params=params_for_agent
            ):
                # 处理工具确认消息
                if isinstance(item, dict):
                    # 在发送工具/控制消息前，先把缓冲的文本 chunk 冲刷出去，保证顺序
                    if response_start_sent:
                        sent_chunks += await self.flush_chunks(msg_session_id, response_id)

                    if item.get("method") == "tools.confirm":
                        execution_id = item.get("id")
                        call_params = item.get("params", {})
                        call_info = call_params.get("call", {})
                        tool_type = call_params.get("toolType", "external")

                        # 保存工具调用上下文
                        self._tool_contexts[execution_id] = {
                            "toolName": call_info.get("toolName", ""),
                            "arguments": call_info.get("arguments", {}),
                            "user_query": message,
                            "toolType": tool_type
                        }

                        logger.info(f"🔧 [工具调用] AI 请求工具: {execution_id}")
                        logger.info(f"🔧 [工具调用] 工具名称: {call_info.get('toolName')}")
                        logger.info(f"🔧 [工具调用] 工具类型: {tool_type}")

                        self._pending_tools[execution_id] = {
                            "toolName": call_info.get("toolName", ""),
                            "arguments": call_info.get("arguments", {}),
                            "user_query": message,
                            "toolType": tool_type,
                            "result_id": None,
                        }
                        logger.info(f"📋 已注册工具等待: {execution_id} ({tool_type})")

                    # 转发给前端（若断链，会抛 ConnectionError）
                    await self._safe_send(item)

                    # ✅ 外部工具发送后，立即退出循环，等待前端返回
                    if item.get("method") == "tools.confirm" and tool_type == "external":
                        logger.info(f"⏸️  [外部工具] 已发送确认，等待前端返回: {execution_id}")
                        break

                    if item.get("method") == "tools.confirm" and tool_type == "internal":
                        logger.info(f"📤 [内部工具] 已发送确认通知（不暂停）: {execution_id}")

                    continue

                # 处理文本内容
                text_content = str(item) if item else ""
                if not text_content:
                    continue

                if not response_start_sent:
                    await self.send_response_start(msg_session_id, response_id, message_id)
                    response_start_sent = True

                sent_chunks += await self.send_chunk(msg_session_id, response_id, text_content, sent_chunks)

            if response_start_sent:
                sent_chunks += await self.flush_chunks(msg_session_id, response_id)
            await self.send_response_end(msg_session_id, response_id, sent_chunks, start_time)

        except ConnectionError:
            # ws 断开：已 stop/cleanup，静默结束
            return
        except Exception as e:
            logger.error(f"消息处理失败: {e}", exc_info=True)
            try:
                if response_start_sent:
                    sent_chunks += await self.flush_chunks(msg_session_id, response_id)
                await self.send_response_end(
                    msg_session_id, response_id, sent_chunks, start_time,
                    status="error", error=str(e)
                )
            except Exception:
                pass

    async def process_message_stream(
        self,
        session_id: str,
        message: str,
        context: Dict[str, Any] = None,
        resume_data: Any = None,
        params: Any = None
    ) -> AsyncIterator[str]:
        """流式处理用户消息（支持生成器暂停/恢复）"""
        logger.info(f"   self.agent: {self.agent}")
        logger.info(f"   type(self.agent): {type(self.agent)}")
        logger.info(f"   message: {message[:100]}...")

        if context is None:
            context = {}

        context["session_id"] = session_id
        context["system_prompt"] = self.system_prompt
        context["websocket_handler"] = self

        if session_id in self._active_generators and resume_data is not None:
            logger.info(f"🔄 [生成器] 恢复会话: {session_id}")
            generator = self._active_generators[session_id]

            try:
                async for chunk in self._resume_generator(generator, resume_data):
                    yield chunk
                del self._active_generators[session_id]
            except StopAsyncIteration:
                logger.info(f"✅ [生成器] 会话完成: {session_id}")
                del self._active_generators[session_id]

        else:
            logger.info(f"🆕 [生成器] 创建新会话: {session_id}")
            logger.info(f"🔍 [创建生成器前] self.agent: {self.agent}")
            if self.agent is None:
                logger.error("❌ self.agent 是 None，无法创建生成器！")
                yield {"error": "Agent未初始化", "message": "请先调用 session.initialize"}
                return

            logger.info(f"✅ [创建生成器] Agent检查通过，开始调用 process_message_stream")
            generator = self.agent.process_message_stream(message=message, context=context, original_params=params)

            self._active_generators[session_id] = generator

            async for chunk in generator:
                yield chunk

            if session_id in self._active_generators:
                del self._active_generators[session_id]

    async def _resume_generator(self, generator, data):
        """恢复生成器并继续迭代"""
        try:
            next_item = await generator.asend(data)
            yield next_item
            async for item in generator:
                yield item
        except StopAsyncIteration as e:
            if hasattr(e, 'value') and e.value:
                yield e.value

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 工具调用流程管理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def handle_tool_confirmation(self, execution_id: str, confirmation_result: Dict):
        """处理前端对 tools.confirm 的响应"""
        if_wait = confirmation_result.get("wait", False)

        if not if_wait:
            tool_info = self._tool_contexts.get(execution_id)
            if not tool_info:
                logger.error(f"未找到工具上下文: {execution_id}")
                return

            random_tool_id = str(random.randint(1, 999999))
            self._pending_tool_results[random_tool_id] = execution_id
            if execution_id in self._pending_tools:
                self._pending_tools[execution_id]["result_id"] = random_tool_id

            await self._safe_send({
                "jsonrpc": "2.0",
                "id": random_tool_id,
                "method": "tools.execute",
                "params": {
                    "toolName": tool_info["toolName"],
                    "arguments": tool_info["arguments"],
                    "executionId": execution_id
                }
            })

    async def handle_tool_result(self, msg_id: str, result_data: Any, agent_id: str = None):
        """处理工具执行结果"""
        logger.info(f"🔧 [工具结果] 收到工具执行结果 | msg_id={msg_id}")
        logger.info(f"🔧 [工具结果] result_data类型: {type(result_data)}")
        logger.info(f"🔧 [工具结果] result_data内容: {str(result_data)[:500]}")
        logger.info(f"🔧 [工具结果] 当前_pending_tool_results: {list(self._pending_tool_results.keys())}")

        execution_id = None
        if str(msg_id) in self._pending_tool_results:
            execution_id = self._pending_tool_results.get(str(msg_id))
        elif str(msg_id) in self._pending_tools:
            execution_id = str(msg_id)

        if not execution_id:
            logger.error(f"❌ [工具结果] 找不到对应的执行ID，放弃处理 | msg_id={msg_id}")
            return

        # ✅ 恢复生成器（新架构）
        session_id = None
        logger.info(f"🔍 [工具结果] 查找活跃生成器，当前数量: {len(self._active_generators)}")
        logger.info(f"🔍 [工具结果] 活跃session_ids: {list(self._active_generators.keys())}")

        for sid, gen_info in self._active_generators.items():
            session_id = self.session_id
            break

        if session_id and session_id in self._active_generators:
            logger.info(f"🔄 [工具结果] 恢复生成器: {session_id}")

            tool_result = result_data

            if isinstance(tool_result, dict) and "error" in tool_result:
                error_info = tool_result["error"]
                error_message = error_info.get("message", "外部工具执行失败")
                logger.error(f"❌ [外部工具错误] {error_message}")
                tool_result = {
                    "success": False,
                    "error": error_message,
                    "error_code": error_info.get("code"),
                    "is_external_tool_error": True
                }
            else:
                while isinstance(tool_result, dict) and "result" in tool_result and len(tool_result) == 1:
                    tool_result = tool_result["result"]

            logger.info(f"🔄 [恢复生成器] 提取到工具结果: {type(tool_result)}")

            if self.agent and hasattr(self.agent, 'cognitive_engine'):
                self.agent.cognitive_engine.tool_res = tool_result
                logger.info(f"🔧 [修复] 已直接设置 agent.cognitive_engine.tool_res")

            logger.info(f"🔄 [恢复生成器] 注入工具结果并恢复")

            try:
                await self._resume_generator_after_tool_result(session_id, tool_result)
            except Exception as e:
                logger.error(f"❌ [工具结果] 恢复生成器失败: {e}", exc_info=True)
        else:
            logger.warning(f"⚠️ [工具结果] 没有找到活跃的生成器，使用旧的 tool_router 机制")

            if self.agent and hasattr(self.agent, 'cognitive_engine'):
                tool_router = self.agent.cognitive_engine.tool_router

                if isinstance(result_data, dict) and "success" in result_data and "result" in result_data:
                    tool_res = result_data
                else:
                    tool_res = {
                        "success": "error" not in result_data,
                        "result": result_data
                    }

                logger.info(f"✅ [工具结果] 准备通知 tool_router | execution_id={execution_id}")
                tool_router.complete_confirmation(execution_id, tool_res)
                logger.info(f"✅ [工具结果] 外部工具完成通知已发送: {execution_id}")

        if str(msg_id) in self._pending_tool_results:
            del self._pending_tool_results[str(msg_id)]
        if execution_id in self._pending_tools:
            del self._pending_tools[execution_id]
        logger.info(f"🧹 [工具结果] 已清理 pending 映射 | msg_id={msg_id} | execution_id={execution_id}")

    async def handle_tool_execute(self, params: Dict[str, Any], agent_id: str = None, db: Session = None):
        """处理 tools.execute 请求"""
        execution_id = params.get("executionId")
        tool_name = params.get("toolName")
        arguments = params.get("arguments", {})

        logger.info(f"🔧 [tools.execute] 收到工具执行请求")
        logger.info(f"🔧 [tools.execute] execution_id={execution_id}")
        logger.info(f"🔧 [tools.execute] tool_name={tool_name}")
        logger.info(f"🔧 [tools.execute] arguments={str(arguments)[:200]}")

        tool_res = await self._execute_tool_directly(tool_name, arguments, agent_id, db)
        logger.info(f"🔧 [tools.execute] 工具执行完成 | 结果: {str(tool_res)[:200]}")

        if tool_res.get("is_external"):
            logger.info(f"🌐 [外部工具] 转发给前端执行: {tool_name}")
            await self._safe_send({
                "jsonrpc": "2.0",
                "id": execution_id,
                "method": "tools.execute",
                "params": params
            })
        else:
            logger.info(f"🔧 [tools.execute] 内部工具执行完成，恢复生成器 | execution_id={execution_id}")

            result_data = {
                "id": execution_id,
                "result": tool_res
            }

            await self.handle_tool_result(execution_id, result_data, agent_id)
            logger.info(f"✅ [tools.execute] 内部工具执行完成，生成器已恢复: {execution_id}")

            if not tool_res.get("success"):
                logger.error(f"❌ [内部工具] 执行失败: {tool_res.get('error', 'Unknown error')}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # JAIP 响应发送（封装 JAIP 格式）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def send_response_start(self, session_id: str, response_id: str, message_id: str):
        """发送 session.response_start"""
        await self._safe_send({
            "jsonrpc": "2.0",
            "method": "session.response_start",
            "params": {
                "sessionId": session_id,
                "responseId": response_id,
                "contentType": "stream",
                "messageId": message_id
            }
        })

    async def send_chunk(self, session_id: str, response_id: str, content: str, sequence: int) -> int:
        """发送 session.response_chunk（带轻量合并）"""
        if not content:
            return 0

        def _get_int_env(name: str, default: int) -> int:
            try:
                v = (os.getenv(name) or "").strip()
                return int(v) if v else default
            except Exception:
                return default

        def _get_float_env(name: str, default: float) -> float:
            try:
                v = (os.getenv(name) or "").strip()
                return float(v) if v else default
            except Exception:
                return default

        flush_chars = _get_int_env("JAIP_STREAM_FLUSH_CHARS", 256)
        flush_interval_s = _get_float_env("JAIP_STREAM_FLUSH_INTERVAL_S", 0.1)
        if flush_chars <= 0:
            flush_chars = 256
        if flush_interval_s <= 0:
            flush_interval_s = 0.1

        key = (session_id, response_id)
        buf = self._chunk_buffers.get(key)
        if not buf:
            buf = {
                "parts": [],
                "len": 0,
                "last_flush_ts": time.time(),
                "seq": 0,
            }
            self._chunk_buffers[key] = buf

        buf["parts"].append(content)
        buf["len"] += len(content)

        now = time.time()
        # Always flush the first chunk so UI gets immediate feedback.
        if buf.get("seq", 0) == 0 and buf["len"] > 0:
            await self._flush_chunk_buffer(session_id, response_id, now=now)
            return 1

        if buf["len"] < flush_chars and (now - buf["last_flush_ts"]) < flush_interval_s:
            return 0

        await self._flush_chunk_buffer(session_id, response_id, now=now)
        return 1

    async def flush_chunks(self, session_id: str, response_id: str) -> int:
        """强制刷新某个 response 的缓冲 chunk。"""
        key = (session_id, response_id)
        buf = self._chunk_buffers.get(key)
        if not buf or not buf.get("parts"):
            self._chunk_buffers.pop(key, None)
            return 0

        await self._flush_chunk_buffer(session_id, response_id, now=time.time())
        return 1

    async def _flush_chunk_buffer(self, session_id: str, response_id: str, *, now: float) -> None:
        key = (session_id, response_id)
        buf = self._chunk_buffers.get(key)
        if not buf:
            return

        parts = buf.get("parts") or []
        if not parts:
            self._chunk_buffers.pop(key, None)
            return

        content = "".join(parts)
        seq = int(buf.get("seq") or 0)
        buf["seq"] = seq + 1
        buf["parts"] = []
        buf["len"] = 0
        buf["last_flush_ts"] = now

        await self._safe_send({
            "jsonrpc": "2.0",
            "method": "session.response_chunk",
            "params": {
                "sessionId": session_id,
                "responseId": response_id,
                "chunk": {
                    "type": "text",
                    "content": content,
                    "sequence": seq,
                },
            },
        })

    async def send_response_end(
        self,
        session_id: str,
        response_id: str,
        total_chunks: int,
        start_time: float,
        status: str = "completed",
        error: Optional[str] = None
    ):
        """发送 session.response_end"""
        processing_time = (time.time() - start_time) * 1000

        metadata = {"processingTime": processing_time}
        if error:
            metadata["error"] = error

        try:
            await self._safe_send({
                "jsonrpc": "2.0",
                "method": "session.response_end",
                "params": {
                    "sessionId": session_id,
                    "responseId": response_id,
                    "status": status,
                    "totalChunks": total_chunks,
                    "metadata": metadata
                }
            })
        finally:
            self._chunk_buffers.pop((session_id, response_id), None)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 内部辅助方法
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_message_content(self, params: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any], str]:
        """提取消息内容和上下文"""
        message_content = params.get("content", "")
        if not message_content:
            message_param = params.get("message", "")
            if isinstance(message_param, dict):
                message_content = message_param.get("content", "")
            else:
                message_content = message_param

        message_type = params.get("messageType", "text")
        message_context = params.get("context", {})
        msg_session_id = params.get("sessionId", self.session_id)

        return message_content, message_type, message_context, msg_session_id

    def _validate_message(self, message_content: str, message_context: Dict[str, Any]) -> bool:
        """验证消息是否有效"""
        if not isinstance(message_content, str):
            message_content = str(message_content or "")

        ctx = message_context or {}
        has_media_ctx = bool(
            ctx.get("file_id") or ctx.get("video_path") or
            ctx.get("files") or ctx.get("url")
        )

        return bool(message_content.strip() or has_media_ctx)

    async def _execute_tool_directly(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_id: str = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """直接执行工具"""
        try:
            from app.core.tools.base import TOOL_REGISTRY
            from app.core.tools.internal_tool_executor import internal_tool_executor
            from app.models.tool import Tool

            real_tool_name = tool_name.replace("_ai_service_#", "")

            logger.info(f"🔧 [工具执行] 工具名称: {tool_name} -> {real_tool_name}")

            is_internal_tool = False
            own_session = False
            tool_db = db
            if tool_db is None:
                from app.db.session import SessionLocal

                tool_db = SessionLocal()
                own_session = True

            try:
                if tool_db:
                    try:
                        tool_record = tool_db.query(Tool).filter(
                            Tool.tool_id == real_tool_name,
                            Tool.is_active == True
                        ).first()

                        if tool_record:
                            is_internal_tool = (tool_record.source == "internal")
                            logger.info(
                                f"📋 [工具执行] 数据库查询: tool_id={real_tool_name}, source={tool_record.source}"
                            )
                        else:
                            logger.info(f"📋 [工具执行] 数据库未找到工具: {real_tool_name}，判定为外部工具")
                    except Exception as db_err:
                        logger.warning(f"⚠️ [工具执行] 数据库查询失败: {db_err}")
            finally:
                if own_session and tool_db:
                    try:
                        tool_db.close()
                    except Exception:
                        pass

            if not is_internal_tool and real_tool_name in TOOL_REGISTRY:
                is_internal_tool = True
                logger.info(f"📋 [工具执行] TOOL_REGISTRY 判定为内部工具: {real_tool_name}")

            if is_internal_tool:
                logger.info(f"✅ [内部工具] 执行工具: {real_tool_name}")

                result = await internal_tool_executor.execute(
                    tool_name=real_tool_name,
                    arguments=arguments,
                    timeout=60.0
                )

                return {"success": True, "result": result}

            else:
                logger.info(f"🌐 [外部工具] 工具需要前端执行: {real_tool_name}")
                return {"success": False, "is_external": True, "tool_name": real_tool_name}

        except Exception as e:
            logger.error(f"❌ [工具执行] 执行失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _resume_generator_after_tool_result(
        self,
        session_id: str,
        tool_result: Any
    ):
        """工具结果返回后，恢复生成器继续执行"""
        if session_id not in self._active_generators:
            logger.error(f"❌ [恢复生成器] session不存在: {session_id}")
            return

        generator = self._active_generators[session_id]

        logger.info(f"🔄 [恢复生成器] 注入工具结果并恢复")

        try:
            next_item = await generator.asend(tool_result)

            logger.info(f"🔄 [恢复生成器] 收到下一个item: {type(next_item)}")

            if isinstance(next_item, dict) and next_item.get("method") == "tools.confirm":
                execution_id = next_item.get("id")
                call_params = next_item.get("params", {})
                call_info = call_params.get("call", {})
                tool_type = call_params.get("toolType", "external")

                logger.info(f"🔧 [恢复生成器] 下一个工具: {call_info.get('toolName')} ({tool_type})")

                self._tool_contexts[execution_id] = {
                    "toolName": call_info.get("toolName", ""),
                    "arguments": call_info.get("arguments", {}),
                    "toolType": tool_type
                }
                self._pending_tools[execution_id] = {
                    "toolName": call_info.get("toolName", ""),
                    "arguments": call_info.get("arguments", {}),
                    "toolType": tool_type,
                    "result_id": None,
                }

                if tool_type == "external":
                    await self._safe_send(next_item)
                    logger.info(f"📤 [恢复生成器] 已发送下一个工具确认: {execution_id}")
                else:
                    tool_name = call_info.get("toolName")
                    arguments = call_info.get("arguments", {})
                    exec_timeout = call_params.get("timeout", 300)

                    logger.info(f"🔧 [恢复生成器] 执行内部工具: {tool_name}")
                    internal_result = await self._execute_internal_tool(tool_name, arguments, exec_timeout)

                    await self._resume_generator_after_tool_result(session_id, internal_result)

            else:
                logger.info(f"📝 [恢复生成器] 收到文本响应")

                response_id = f"resp_{uuid.uuid4().hex[:8]}"
                message_id = f"msg_{uuid.uuid4().hex[:8]}"
                chunk_index = 0

                await self.send_response_start(session_id, response_id, message_id)
                await self.send_chunk(session_id, response_id, str(next_item), chunk_index)
                chunk_index += 1

                logger.info(f"🔄 [恢复生成器] 继续迭代获取后续响应内容")
                async for item in generator:
                    await self.send_chunk(session_id, response_id, str(item), chunk_index)
                    chunk_index += 1

                await self.send_response_end(session_id, response_id, chunk_index, time.time())
                logger.info(f"✅ [恢复生成器] 完整响应发送完毕，共 {chunk_index} 个chunk")

                if session_id in self._active_generators:
                    del self._active_generators[session_id]
                    logger.info(f"✅ [恢复生成器] 所有工具执行完毕，生成器已清理")

        except StopAsyncIteration as e:
            logger.info(f"✅ [恢复生成器] 生成器正常结束")

            if hasattr(e, 'value') and e.value:
                response_id = f"resp_{uuid.uuid4().hex[:8]}"
                message_id = f"msg_{uuid.uuid4().hex[:8]}"

                await self.send_response_start(session_id, response_id, message_id)
                await self.send_chunk(session_id, response_id, str(e.value), 0)
                await self.send_response_end(session_id, response_id, 1, time.time())

            if session_id in self._active_generators:
                del self._active_generators[session_id]

        except ConnectionError:
            return
        except Exception as e:
            logger.error(f"❌ [恢复生成器] 异常: {e}", exc_info=True)

            try:
                error_message = "工具执行过程中出现问题，请检查工具参数或稍后重试"
                if isinstance(tool_result, dict) and "error" in tool_result:
                    error_message = f"工具执行失败: {tool_result['error']}"

                response_id = f"resp_error_{uuid.uuid4().hex[:8]}"
                message_id = f"msg_error_{uuid.uuid4().hex[:8]}"

                await self.send_response_start(session_id, response_id, message_id)
                await self.send_chunk(session_id, response_id, error_message, 0)
                await self.send_response_end(session_id, response_id, 1, time.time(), status="error", error=str(e))
            except Exception:
                pass

            if session_id in self._active_generators:
                del self._active_generators[session_id]

    async def _execute_internal_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: int = 300
    ) -> Any:
        """执行内部工具（后台执行）"""
        logger.info(f"🔧 [内部工具执行] 开始: {tool_name}, 超时={timeout}秒")

        try:
            if not self.agent or not hasattr(self.agent, 'cognitive_engine'):
                logger.error("❌ [内部工具执行] Agent或cognitive_engine未初始化")
                return {"error": "系统未就绪"}

            tool_router = self.agent.cognitive_engine.tool_router

            from app.core.tools import TOOL_REGISTRY
            tool_obj = TOOL_REGISTRY.get(tool_name)
            if not tool_obj:
                logger.error(f"❌ [内部工具执行] 工具未注册: {tool_name}")
                return {"error": f"工具不存在: {tool_name}"}

            import asyncio
            result = await asyncio.wait_for(
                tool_router.execute_tool(tool_obj, arguments),
                timeout=timeout
            )

            if isinstance(result, dict) and 'task_id' in result:
                task_id = result['task_id']
                logger.info(f"⏳ [内部工具执行] 等待后台任务: {task_id}")
                result = await tool_router.wait_task(task_id, timeout=timeout)

            logger.info(f"✅ [内部工具执行] 完成: {tool_name}")
            return result

        except asyncio.TimeoutError:
            logger.error(f"❌ [内部工具执行] 超时: {tool_name}")
            return {"error": f"工具执行超时({timeout}秒)"}
        except Exception as e:
            logger.error(f"❌ [内部工具执行] 失败: {tool_name}, error: {e}", exc_info=True)
            return {"error": str(e)}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 清理（新增 async：断链时立即 stop 下游 + 清生成器）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    async def cleanup_async(self, reason: str = "cleanup") -> None:
        self._chunk_buffers.clear()

        # 1) 关闭生成器，阻止继续产出/继续发送
        for sid, gen in list(self._active_generators.items()):
            try:
                await gen.aclose()
            except Exception:
                pass
        self._active_generators.clear()

        # 2) 通知 agent 停止所有该 client 的下游任务（VideoPatrolAgent 实现了 stop_all_tasks_by_client_id）
        try:
            if self.agent and hasattr(self.agent, "stop_all_tasks_by_client_id"):
                self.agent.stop_all_tasks_by_client_id(self.client_id, reason=reason)
        except Exception as e:
            logger.warning("[JAIPHandler] stop_all_tasks_by_client_id failed: %s", e)

        # 3) 清理工具会话
        if self.session_id:
            try:
                session_tool_manager.clear_session_tools(self.session_id)
            except Exception:
                pass

        self._tool_contexts.clear()
        self._pending_tools.clear()
        self._pending_tool_results.clear()

    def cleanup(self):
        """清理资源（同步版本，保留兼容）"""
        if self.session_id:
            try:
                session_tool_manager.clear_session_tools(self.session_id)
            except Exception as e:
                logger.error(f"清理会话工具失败: {e}")

        self._tool_contexts.clear()
        self._pending_tools.clear()
        self._pending_tool_results.clear()

    # ==================== WebSocket 管理方法（保留原有） ====================

    def register_websocket(self, session_id: str, websocket):
        """
        注册 WebSocket 连接（兼容 websocket_auto.py）
        """
        try:
            if hasattr(self, 'ws_manager') and self.ws_manager:
                if not hasattr(self.ws_manager, 'direct_websockets'):
                    self.ws_manager.direct_websockets = {}
                self.ws_manager.direct_websockets[session_id] = websocket
                logger.info(f"✅ [JAIPHandler] WebSocket 已注册到 ws_manager | session_id: {session_id}")
            else:
                if not hasattr(self, '_websockets'):
                    self._websockets = {}
                self._websockets[session_id] = websocket
                logger.info(f"✅ [JAIPHandler] WebSocket 已直接注册 | session_id: {session_id}")

        except Exception as e:
            logger.error(f"❌ [JAIPHandler] WebSocket 注册失败 | session_id: {session_id} | error: {e}")

    def unregister_websocket(self, session_id: str):
        """
        取消注册 WebSocket 连接
        """
        try:
            if hasattr(self, 'ws_manager') and self.ws_manager and hasattr(self.ws_manager, 'direct_websockets'):
                if session_id in self.ws_manager.direct_websockets:
                    del self.ws_manager.direct_websockets[session_id]
                    logger.info(f"🔌 [JAIPHandler] WebSocket 已从 ws_manager 取消注册 | session_id: {session_id}")

            if hasattr(self, '_websockets') and session_id in self._websockets:
                del self._websockets[session_id]
                logger.info(f"🔌 [JAIPHandler] WebSocket 已直接取消注册 | session_id: {session_id}")

        except Exception as e:
            logger.error(f"❌ [JAIPHandler] WebSocket 取消注册失败 | session_id: {session_id} | error: {e}")
