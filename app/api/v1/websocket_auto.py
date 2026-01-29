"""
WebSocket接口 - JAIP协议实时通信
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import json
import logging
import random
import time
from typing import Dict, Any, Optional, Tuple
import uuid

from app.db.session import get_db
from app.core.jaip.handler import JAIPHandler
from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter()


def _normalize_rpc_method(method: Optional[str]) -> Optional[str]:
    """兼容历史方法名（避免前端仍在用旧协议导致会话无法初始化）。"""
    if not method:
        return method
    m = method.strip()
    aliases = {
        # 旧版初始化
        "initialize": "session.initialize",
        "session.init": "session.initialize",
        "init": "session.initialize",
        "session.create": "session.initialize",
        # 旧版消息
        "chat": "session.message",
        "message": "session.message",
        "session.chat": "session.message",
        "user.message": "session.message",
        "user.chat": "session.message",
        # 旧版心跳
        "ping": "session.ping",
        "session.heartbeat": "session.ping",
        "heartbeat": "session.ping",
    }
    return aliases.get(m, m)


def _normalize_initialize_params(params: Any) -> Dict[str, Any]:
    """把旧版 initialize 入参字段名映射到 session.initialize 兼容字段。"""
    if not isinstance(params, dict):
        return {}
    normalized = dict(params)

    # session_id → sessionId
    if "sessionId" not in normalized and "session_id" in normalized:
        normalized["sessionId"] = normalized.get("session_id")

    # user_id/userId → userContext.userId
    if "userContext" not in normalized:
        user_id = normalized.get("user_id") or normalized.get("userId")
        if user_id is not None:
            normalized["userContext"] = {"userId": user_id}

    return normalized


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """接受新的WebSocket连接"""
        await websocket.accept()

        # 设置TCP_NODELAY以减少延迟
        try:
            import socket
            client = websocket.client
            if hasattr(client, 'transport') and hasattr(client.transport, 'get_extra_info'):
                sock = client.transport.get_extra_info('socket')
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        self.active_connections[client_id] = websocket
        self.session_data[client_id] = {
            "session_id": str(uuid.uuid4()),
            "connected_at": time.time()
        }
        logger.info(f"WebSocket客户端已连接: {client_id}")
    
    def disconnect(self, client_id: str):
        """断开WebSocket连接"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.session_data:
            del self.session_data[client_id]
        logger.info(f"WebSocket客户端已断开: {client_id}")

    async def send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """发送消息给特定客户端

        Returns:
            bool: 发送成功返回 True，失败返回 False
        """
        if client_id not in self.active_connections:
            logger.warning(f"❌ [Auto-发送消息] 客户端���存在: {client_id}")
            return False

        websocket = self.active_connections[client_id]
        try:
            # 修复URL截断问题：确保JSON序列化不会截断长字符串
            import json
            json_str = json.dumps(message, ensure_ascii=False, separators=(',', ':'))

            # 记录调试信息
            chunk_content = message.get('params', {}).get('chunk', {}).get('content', '')
            if len(chunk_content) > 100:
                logger.debug(f"📡 [Auto-发送消息] 内容长度: {len(chunk_content)}, 预览: {chunk_content[:100]}...")

            await websocket.send_text(json_str)
            return True
        except Exception as e:
            logger.error(f"❌ [Auto-发送消息] 发送失败(连接可能已关闭): {client_id}, error: {e}")
            self.disconnect(client_id)
            return False
    
    def get_session_data(self, client_id: str) -> Dict[str, Any]:
        """获取会话数据"""
        return self.session_data.get(client_id, {})


# 创建连接管理器实例
manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════
# 辅助方法：会话初始化
# ═══════════════════════════════════════════════════════════════
async def _handle_session_initialize(
    params: Dict[str, Any],
    request_id: str,
    agent_id: str,
    session_id: str,
    session_data: Dict[str, Any],
    jaip_handler: 'JAIPHandler',
    client_id: str,
    send_response: bool = True,
) -> str:
    """处理 session.initialize 方法（内部创建数据库会话）"""
    logger.debug("初始化参数: %s", json.dumps(params, ensure_ascii=False)[:500])
    protocol_version = params.get("protocolVersion", "2.0")
    new_session_id = params.get("sessionId", session_id)
    available_tools = params.get("availableTools", [])
    user_context = params.get("userContext", {})
    init_params = params.get("parameters", {})

    # 创建数据库会话（用完立即关闭）
    from app.db.session import SessionLocal
    db = SessionLocal()

    # 在外部定义变量，避免作用域问题
    agent_name = agent_id  # 默认值

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
                from app.db.seed_defaults import (
                    seed_default_tools,
                    seed_default_templates_and_agents,
                )

                seed_default_tools()
                seed_default_templates_and_agents()
            except Exception as e:
                logger.warning(f"⚠️ Default seed failed during initialize: {e}")

            try:
                db.rollback()
            except Exception:
                pass

            agent_config = agent_service.get_agent(agent_id)

        if not agent_config:
            if send_response:
                await manager.send_message(client_id, {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": "error.agent_not_found"}
                })
            return new_session_id

        logger.debug("agent配置: %s", agent_config.config)
        agent_name = agent_config.name  # 保存到外部变量
        config = agent_config.config if agent_config.config else {}
        system_prompt = config.get("prompt", "")
        model = config.get("model", "qwen-turbo")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 2048)

        # 初始化会话
        init_result = await jaip_handler.initialize_session(
            session_id=session_id,
            agent_config={
                "name": agent_config.name,
                "type": agent_config.type,
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

        session_data["session_id"] = new_session_id
        session_data["user_context"] = user_context
        session_data['system_prompt'] = system_prompt

        logger.info(f"会话初始化成功: {new_session_id}")

        # 发送初始化响应
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": protocol_version,
                "sessionId": new_session_id,
                "status": "initialized",
                "serverCapabilities": {
                    "streaming": True,
                    "toolExecution": True,
                    "multiModal": False
                },
                "agentInfo": {
                    "id": agent_id,
                    "name": init_result["agent_info"]["name"],
                    "model": init_result["agent_info"]["model"],
                    "toolCount": init_result["agent_info"]["tool_count"]
                }
            }
        }

        if send_response:
            await manager.send_message(client_id, response)
        return new_session_id

    except Exception as e:
        logger.error(f"会话初始化失败: {e}", exc_info=True)
        if send_response:
            await manager.send_message(client_id, {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32002, "message": f"初始化失败: {str(e)}"}
            })
        return session_id
    finally:
        # 立即关闭数据库会话（只读操作，提交空事务避免ROLLBACK日志）
        try:
            db.commit()  # 提交空事务
        except:
            pass
        db.close()


# ═══════════════════════════════════════════════════════════════
# 辅助方法：消息处理
# ═══════════════════════════════════════════════════════════════
def _extract_message_content(params: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any], str]:
    """提取消息内容和上下文"""
    # 兼容两种格式: {"content": "..."} 或 {"message": {"content": "..."}}
    message_content = params.get("content", "")
    if not message_content:
        message_param = params.get("message", "")
        if isinstance(message_param, dict):
            message_content = message_param.get("content", "")
        else:
            message_content = message_param

    message_type = params.get("messageType", "text")
    message_context = params.get("context", {})
    msg_session_id = params.get("sessionId", "")

    return message_content, message_type, message_context, msg_session_id


def _validate_message(message_content: str, message_context: Dict[str, Any]) -> bool:
    """验证消息是否有效"""
    if not isinstance(message_content, str):
        message_content = str(message_content or "")

    # 空内容且无媒体上下文时无效
    ctx = message_context or {}
    has_media_ctx = bool(
        ctx.get("file_id") or ctx.get("video_path") or
        ctx.get("files") or ctx.get("url")
    )

    return bool(message_content.strip() or has_media_ctx)


# ═══════════════════════════════════════════════════════════════
# 辅助方法：响应流发送
# ═══════════════════════════════════════════════════════════════
async def _send_schema_based_response(
    client_id: str,
    request_id: str,
    session_id: str,
    content: Optional[Dict[str, Any]] = None,
    schema_name: Optional[str] = None,
    ai_response: Optional[str] = None,
    tool_result: Optional[Dict[str, Any]] = None,
    user_query: Optional[str] = None
):
    """
    发送 schema.based.response 固定结构的非流式消息

    Args:
        client_id: 客户端ID
        request_id: 请求ID（与请求时的id保持一致）
        session_id: 会话ID
        content: 固定结构的JSON内容（如果提供则直接使用）
        schema_name: Schema名称（用于自动生成content）
        ai_response: AI的文本回复（用于自动生成content）
        tool_result: 工具执行结果（用于自动生成content）
        user_query: 用户原始问题/要求（用于辅助JSON提取）

    消息格式:
    {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "schema.based.response",
        "params": {
            "sessionId": session_id,
            "content": content
        }
    }

    自动生成content的规则:
    1. 如果提供了 content 参数，直接使用
    2. 如果提供了 ai_response，生成文本回复格式
    3. 如果提供了 tool_result，生成工具结果格式
    4. 否则生成默认格式
    """
    # 生成 content
    if ai_response:
        # AI 文本回复格式 - 使用 JSON 提取器提取结构化数据
        try:
            from app.core.llm import extract_json

            # 定义提取的 JSON Schema
            extraction_schema = {
                "type": "object",
                "properties": {
                    "isCorrect": {
                        "type": "boolean",
                        "description": "根据是否满足用户需求 返回True 或者False"
                    },
                    "conclusion": {
                        "type": "string",
                        "description": "简要结论描述"
                    }
                },
                "required": ["isCorrect", "conclusion"]
            }

            # 构建提取指令（包含用户要求）
            extraction_instruction = "从AI回复中提取结构化结论"
            if user_query:
                extraction_instruction = f"""请根据用户的问题和AI的回复，提取结构化结论：

用户问题：{user_query}

AI回复：{ai_response}

请判断：
1. isCorrect: 根据AI回复的内容，判断实际结果（不是判断AI回复是否正确）
   - 例子1：用户问"图片有没有人"，AI回复"图片中没有人" → isCorrect应该返回false（因为实际没有人）
   - 例子2：用户问"图片有没有人"，AI回复"图片中有一个人" → isCorrect应该返回true（因为实际有人）
   - 例子3：用户问"设备是否正常"，AI回复"设备运行正常" → isCorrect应该返回true（因为实际正常）
2. conclusion: 提取简要结论（1-2句话概括实际情况，而非AI的回复质量"""

            # 从 AI 回复中提取结构化数据
            extracted_data = await extract_json(
                text=ai_response if not user_query else extraction_instruction,
                schema=extraction_schema,
                instruction=None if user_query else "从AI回复中提取安全检查结论，判断是否存在问题",
                temperature=0.1
            )
            logger.debug("提取结果: %s", extracted_data)

            # 生成 content
            content = {
                "isCorrect": extracted_data.get("isCorrect", False),
                "conclusion": extracted_data.get("conclusion", ai_response[:100]),
                "timestamp": int(time.time() * 1000)
            }

            logger.info(f"✅ [JSON提取] AI回复 → 结构化数据: isCorrect={content['isCorrect']}")

        except Exception as e:
                logger.warning(f"⚠️ [JSON提取] 提取失败，使用默认格式: {e}")
                # 提取失败，使用默认格式
                content = {
                    "isCorrect": False,
                    "conclusion": ai_response,
                    "schema": schema_name or "unknown",
                    "timestamp": int(time.time() * 1000)
                }


    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "schema.based.response",
        "params": {
            "sessionId": session_id,
            "content": content,
            "schema":"correctnessCheck",
            "type":"simple"
        }
    }
    logger.debug("发送 schema.based.response 内容: %s", json.dumps(response, ensure_ascii=False)[:500])
    await manager.send_message(client_id, response)
    logger.info(f"✅ 发送 schema.based.response: session={session_id}, id={request_id}, schema={schema_name or 'custom'}")


async def _send_response_start(
    client_id: str,
    msg_session_id: str,
    response_id: str,
    message_id: str
):
    """发送 session.response_start 消息"""
    await manager.send_message(client_id, {
        "jsonrpc": "2.0",
        "method": "session.response_start",
        "params": {
            "sessionId": msg_session_id,
            "responseId": response_id,
            "contentType": "stream",
            "messageId": message_id
        }
    })


async def _send_text_chunk(
    client_id: str,
    msg_session_id: str,
    response_id: str,
    content: str,
    chunk_index: int
):
    """发送 session.response_chunk 消息"""
    await manager.send_message(client_id, {
        "jsonrpc": "2.0",
        "method": "session.response_chunk",
        "params": {
            "sessionId": msg_session_id,
            "responseId": response_id,
            "chunk": {
                "type": "text",
                "content": content,
                "sequence": chunk_index
            }
        }
    })


async def _send_response_end(
    client_id: str,
    msg_session_id: str,
    response_id: str,
    chunk_index: int,
    start_time: float,
    status: str = "completed",
    error: Optional[str] = None
):
    """发送 session.response_end 消息"""
    processing_time = (time.time() - start_time) * 1000

    metadata = {"processingTime": processing_time}
    if error:
        metadata["error"] = error

    await manager.send_message(client_id, {
        "jsonrpc": "2.0",
        "method": "session.response_end",
        "params": {
            "sessionId": msg_session_id,
            "responseId": response_id,
            "status": status,
            "totalChunks": chunk_index,
            "metadata": metadata
        }
    })
# ═══════════════════════════════════════════════════════════════
# 辅助方法：处理流式消息
# ═══════════════════════════════════════════════════════════════
async def _handle_session_message(
    params: Dict[str, Any],
    request_id: str,
    session_id: str,
    agent_id: str,
    jaip_handler: 'JAIPHandler',
    client_id: str,
    tool_call_context: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """
    处理 session.message 方法（流式输出）

    Args:
        tool_call_context: 用于保存工具调用上下文的字典 {execution_id: {toolName, arguments}}

    Returns:
        工具调用的 execution_id（如果有），用于后续匹配 JetLinks 返回结果
    """
    # 提取和验证消息
    message, _, message_context, msg_session_id = _extract_message_content(params)

    if not _validate_message(message, message_context):
        logger.debug("消息验证失败，跳过处理")
        return None

    # logger.info(f"收到用户消息: {message_content[:100]}...")

    # 准备响应
    response_id = f"resp_{uuid.uuid4().hex[:8]}"
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    response_start_sent = False
    chunk_index = 0

    # 追踪工具调用 ID
    tool_execution_id: Optional[str] = None
    logger.debug("提示词参数 | context=%s | message=%s", message_context, (message[:200] + "...") if isinstance(message, str) and len(message) > 200 else message)
    # context
    try:
        async for item in jaip_handler.process_message_stream(
            session_id=msg_session_id,
            message=message,
            context=message_context
        ):
            # 处理工具调用消息
            if isinstance(item, dict):
                # 记录工具调用的 execution_id 和相关信息
                if "id" in item and item.get("method") == "tools.confirm":
                    tool_execution_id = item.get("id")

                    # 保存工具调用上下文（工具名称、参数、用户问题）
                    call_params = item.get("params", {})
                    call_info = call_params.get("call", {})
                    tool_type = call_params.get("toolType", "external")  # 工具类型

                    tool_call_context[tool_execution_id] = {
                        "toolName": call_info.get("toolName", ""),
                        "arguments": call_info.get("arguments", {}),
                        "userQuery": message,  # 保存用户原始问题
                        "toolType": tool_type
                    }

                    logger.info(f"🔧 [Auto-工具调用] AI 请求工具: {tool_execution_id}")
                    logger.info(f"🔧 [Auto-工具调用] 工具名称: {call_info.get('toolName')}")
                    logger.info(f"🔧 [Auto-工具调用] 工具类型: {tool_type}")
                    logger.info(f"🔧 [Auto-工具调用] 工具参数: {json.dumps(call_info.get('arguments', {}), ensure_ascii=False)[:200]}")

                    # 判断工具类型
                    if tool_type == "internal":
                        # 🔧 内部工具：后端直接执行，不发送给前端
                        logger.info(f"🔧 [Auto-内部工具] 后端执行: {call_info.get('toolName')}")

                        # 启动后台任务
                        tool_name = call_info.get("toolName")
                        arguments = call_info.get("arguments", {})
                        exec_timeout = call_params.get("timeout", 300)

                        # 执行内部工具
                        tool_result = await _execute_tool_directly(
                            session_id=session_id,
                            tool_name=tool_name,
                            arguments=arguments,
                            jaip_handler=jaip_handler,
                            client_id=client_id,
                            request_id=tool_execution_id,
                            agent_id=agent_id,
                            auto_send_response=False,  # 不自动发送响应
                            user_query=message
                        )

                        logger.info(f"✅ [Auto-内部工具] 执行完成: {tool_name}")

                        # 立即恢复生成器（不需要等待前端）
                        # 通过重新调用 process_message_stream 并传入工具结果
                        # 这会触发生成器的恢复
                        continue

                    else:
                        # 🌐 外部工具：发送给前端
                        item["params"]["auto"] = True  # 自动确认工具调用

                        send_success = await manager.send_message(client_id, item)
                        if not send_success:
                            logger.error(f"❌ [Auto-工具调用] 发送 tools.confirm 失败: {tool_execution_id}")
                            # 连接已断开，无需继续处理
                            return None

                        logger.info(f"✅ [Auto-工具调用] tools.confirm 已发送: {tool_execution_id}")
                        continue
            else:

                # 处理文本内容
                text_content = str(item) if item else ""
                if not text_content:
                    continue

                # 发送 response_start（仅首次）
                if not response_start_sent:
                    await _send_response_start(client_id, msg_session_id, response_id, message_id)
                    response_start_sent = True

                # 发送文本块
                await _send_text_chunk(client_id, msg_session_id, response_id, text_content, chunk_index)
                chunk_index += 1

                # 发送 response_end
                # if response_start_sent:
            await _send_response_end(client_id, msg_session_id, response_id, chunk_index, start_time)

        return tool_execution_id

    except Exception as e:
        logger.error(f"消息处理失败: {e}", exc_info=True)
        await _send_response_end(
            client_id, msg_session_id, response_id, chunk_index, start_time,
            status="error", error=str(e)
        )
        return None




async def _execute_tool_directly(
    session_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    jaip_handler: 'JAIPHandler',
    client_id: str,
    request_id: str,
    agent_id: str = None,
    auto_send_response: bool = True,
    timeout: float = 120.0,
    user_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    直接执行工具（绕过确认队列机制）

    Args:
        session_id: 会话ID
        tool_name: 工具名称
        arguments: 工具参数
        jaip_handler: JAIP处理器实例
        client_id: 客户端ID
        request_id: 请求ID（用于发送响应）
        agent_id: 智能体ID（用于保存工具历史）
        auto_send_response: 是否自动发送 schema.based.response（默认True）
        timeout: 超时时间（秒），默认120秒

    Returns:
        工具执行结果
        {
            "success": True/False,
            "result": "工具返回结果" (成功时),
            "error": "错误信息" (失败时)
        }
    """
    try:
        from app.core.tools import TOOL_REGISTRY
        from app.core.tools.internal_tool_executor import internal_tool_executor

        # 去掉前缀获取真实工具名
        real_tool_name = tool_name.replace("_ai_service_#", "")

        logger.info(f"🔧 [工具执行] 工具名称: {tool_name} -> {real_tool_name}")

        # 判断是否为内部工具（检查是否在 TOOL_REGISTRY 中）
        if real_tool_name in TOOL_REGISTRY:
            logger.info(f"✅ [内部工具] 执行工具: {real_tool_name}")

            # 使用 internal_tool_executor 执行（已修复递归bug）
            result = await internal_tool_executor.execute(
                tool_name=real_tool_name,
                arguments=arguments,
                timeout=timeout
            )
            logger.debug("工具结果: %s", result)

            # 自动发送 schema.based.response
            if auto_send_response:
                # 将工具结果转换为文本，传递给 ai_response 参数
                result_text = str(result) if result else ""
                await _send_schema_based_response(
                    client_id=client_id,
                    request_id=request_id,
                    session_id=session_id,
                    schema_name=f"tool.{real_tool_name}",
                    ai_response=result_text,  # 传递工具结果文本
                    user_query=user_query  # 传递用户问题
                )
                logger.info(f"📤 [工具执行] 已自动发送 schema.based.response: {real_tool_name}")

            return {"success": True, "result": result}
        else:
            # 外部工具 - 返回失败（需要转发到 JetLinks）
            logger.info(f"🌐 [外部工具] 工具不在内部注册表: {real_tool_name}")
            error_result = {"success": False, "error": "外部工具需要转发执行"}

            # 不自动发送响应（外部工具由 JetLinks 处理）
            # if auto_send_response:
            #     await _send_schema_based_response(
            #         client_id=client_id,
            #         request_id=request_id,
            #         session_id=session_id,
            #         schema_name=f"tool.{real_tool_name}",
            #         tool_result=error_result
            #     )

            return error_result

    except Exception as e:
        logger.error(f"❌ [工具执行] 工具执行失败: {e}", exc_info=True)
        error_result = {"success": False, "error": str(e)}

        # 发送错误响应
        if auto_send_response:
            error_text = f"工具执行失败: {str(e)}"
            await _send_schema_based_response(
                client_id=client_id,
                request_id=request_id,
                session_id=session_id,
                schema_name=f"tool.{tool_name}",
                ai_response=error_text,  # 传递错误信息文本
                user_query=user_query
            )

        return error_result

# ═══════════════════════════════════════════════════════════════
# 辅助方法：工具结果转AI回复
# ═══════════════════════════════════════════════════════════════
async def tool_result_to_ai_response(
    tool_result: Dict[str, Any],
    session_id: str,
    client_id: str,
    jaip_handler: 'JAIPHandler',
    execute_id: str,
    tool_call_context: Dict[str, Dict[str, Any]],
    agent_id: str
) -> None:
    """
    将工具执行结果通过AI生成自然语言回复并发送给客户端

    Args:
        tool_result: 工具执行结果 (包含 result 或 error 字段)
        session_id: 会话ID
        client_id: 客户端ID
        jaip_handler: JAIP处理器
        execute_id: 执行ID
        tool_call_context: 工具调用上下文
        agent_id: 智能体ID
    """
    response_id = f"resp_{uuid.uuid4().hex[:8]}"
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    chunk_index = 0
    response_started = False

    # 构造提示消息（使用 tool_result 而不是 data）
    tool_result_message = f"""<tool> 工具回复消息:{str(tool_result)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 重要提醒
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 此次为工具返回结果，不需要再调用工具
2. 如果回复为空，则告知用户工具未返回任何值，请检查参数是否正确

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 数据展示规范（严格执行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **数据真实性（最重要）**
   - 所有数据必须来自工具实际返回值
   - 禁止编造、猜测或使用示例数据
   - 工具返回什么就展示什么，不要修改数值

2. **表格格式**
   当数据适合用表格展示时：
   - 使用清晰的表头
   - 数值保留2位小数
   - 时间格式：yyyy-MM-dd HH:mm:ss

3. **图片/视频展示（重要）**
   ⚠️ 如果工具返回结果中包含图片URL（image_url字段），必须使用以下格式展示图片：

   :::image
   \`\`\`json
   图片URL地址
   \`\`\`
   :::

   如果包含视频URL（video_url字段），使用以下格式展示视频：

   :::video
   \`\`\`json
   视频URL地址
   \`\`\`
   :::

   ✅ 正确示例：
   :::image
   \`\`\`json
   https://example.com/image.jpg
   \`\`\`
   :::

4. **图表可视化（强制要求）**
   ⚠️ 当数据包含 ≥3 个时间点时，**必须同时展示表格和图表**，缺一不可！

   图表格式（严格按此格式）：

   :::echarts
   ```json
   {{{{
     "title": {{{{
       "text": "图表标题",
       "left": "center"
     }}}},
     "tooltip": {{{{
       "trigger": "axis"
     }}}},
     "xAxis": {{{{
       "type": "category",
       "data": ["时间1", "时间2", "时间3"]
     }}}},
     "yAxis": {{{{
       "type": "value",
       "name": "单位"
     }}}},
     "series": [{{{{
       "name": "数据名称",
       "type": "line",
       "smooth": true,
       "data": [数值1, 数值2, 数值3]
     }}}}]
   }}}}
   ```
   :::

   图表类型选择：
   - 折线图（line）：温度变化、趋势分析（默认首选）
   - 柱状图（bar）：多设备对比、统计数据
   - 饼图（pie）：占比分布

   ❌ 禁止行为：只展示表格不展示图表
   ✅ 正确做法：表格 + 图表双重展示

5. **知识库检索结果**
   如果数据来自知识库检索，必须在最后注明来源文档：

   ---
   📚 参考文档：
   - 文档1：《xxx》
   - 文档2：《xxx》

请严格按照上述规范生成回复，基于工具返回的真实数据，不要编造数据。"""

    try:
        async for item in jaip_handler.process_message_stream(
            session_id=session_id,
            message=tool_result_message,
            context={}
        ):
            # 发送 response_start（仅首次）
            if not response_started:
                await _send_response_start(client_id, session_id, response_id, message_id)
                response_started = True

            # 发送文本块
            await _send_text_chunk(client_id, session_id, response_id, item, chunk_index)
            chunk_index += 1

        # 发送 response_end
        await _send_response_end(client_id, session_id, response_id, chunk_index, start_time)

        # 保存工具调用历史到 Redis
        if execute_id in tool_call_context:
            tool_info = tool_call_context[execute_id]
            try:
                from app.core.tools.parameter_filler import parameter_filler
                parameter_filler.save_tool_call_history(
                    session_id=session_id,
                    tool_name=tool_info.get("toolName", ""),
                    arguments=tool_info.get("arguments", {}),
                    result=tool_result.get("result", {}),
                    agent_id=agent_id
                )
                logger.info(f"💾 已保存工具调用历史: {tool_info.get('toolName')}")
            except Exception as save_err:
                logger.warning(f"⚠️ 保存工具历史失败: {save_err}")

    except Exception as e:
        logger.error(f"处理工具结果失败: {e}", exc_info=True)
        if response_started:
            await _send_response_end(
                client_id, session_id, response_id, chunk_index, start_time,
                status="error", error=str(e)
            )


# ═══════════════════════════════════════════════════════════════
# WebSocket 端点
# ═══════════════════════════════════════════════════════════════
@router.websocket("/auto/{agent_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    agent_id: str
):
    """
    WebSocket端点 - JAIP协议实时通信

    路径: /api/v1/jaip/session/{agent_id}

    注意: 不使用 Depends(get_db)，避免长连接持有数据库会话导致连接池耗尽
    """
    client_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"

    # 初始化 JAIP 处理器
    try:
        jaip_handler = JAIPHandler()
    except Exception as e:
        logger.error(f"初始化 JAIP 处理器失败: {e}")
        await websocket.close(code=1011, reason=str(e))
        return

    try:
        # 建立连接
        await manager.connect(websocket, client_id)
        session_data = manager.get_session_data(client_id)
        session_id = session_data["session_id"]

        jaip_handler.register_websocket(session_id, websocket)

        # 发送欢迎消息
        await manager.send_message(client_id, {
            "type": "connected",
            "message": f"已连接到智能体 {agent_id}",
            "session_id": session_id,
            "agent_id": agent_id
        })

        # 状态管理：追踪工具调用 ID（用于匹配 JetLinks 返回结果）
        
        tool_id: Optional[str] = None      # 转发给 JetLinks 的随机 ID（字符串类型）
        tool_call_context: Dict[str, Dict[str, Any]] = {}  # {execution_id: {toolName, arguments}}
        # 单次历史会话记忆及提示词记忆
        global execute_id
        
        context = {}

        # 兼容客户端乱序：某些情况下会先发 session.message/schema.based.request 再发 session.initialize
        initialized = False
        auto_initialize_attempted = False
        pending_rpc_messages: list[tuple[str, Dict[str, Any], Any]] = []
        pending_rpc_messages_limit = 20

        async def _flush_pending_messages():
            global execute_id
            nonlocal pending_rpc_messages, initialized, session_id
            if not initialized or not pending_rpc_messages:
                return

            logger.warning(
                f"🔄 [WebSocket-Auto] 初始化后处理缓存消息: {len(pending_rpc_messages)} 条 | client_id={client_id}"
            )
            buffered = pending_rpc_messages
            pending_rpc_messages = []

            for buffered_method, buffered_params, buffered_request_id in buffered:
                if buffered_method == "session.message":
                    exec_id = await _handle_session_message(
                        buffered_params, buffered_request_id, session_id,
                        agent_id, jaip_handler, client_id, tool_call_context
                    )
                    if exec_id:
                        # 如果 AI 请求了工具调用，记录 execution_id
                        execute_id = exec_id
                elif buffered_method == "schema.based.request":
                    schema_name = buffered_params.get("schema")
                    response_type = buffered_params.get("type", "simple")
                    content = buffered_params.get("content", "")
                    msg_session_id = buffered_params.get("sessionId", session_id)

                    logger.info(
                        f"[Schema Request][Buffered] {schema_name}/{response_type}: "
                        f"{content[:100] if isinstance(content, str) else content}"
                    )

                    message_params = {
                        "message": {"content": content if isinstance(content, str) else str(content)},
                        "sessionId": msg_session_id,
                        "context": buffered_params
                    }

                    exec_id = await _handle_session_message(
                        message_params, buffered_request_id, msg_session_id,
                        agent_id, jaip_handler, client_id, tool_call_context
                    )
                    if exec_id:
                        execute_id = exec_id

        # 消息循环
        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"收到消息: {json.dumps(data, ensure_ascii=False)[:200]}")
                # 添加INFO级别日志，确保能看到
                logger.info(f"📨 [WebSocket-Auto] 收到消息 | keys={list(data.keys())} | id={data.get('id','N/A')} | method={data.get('method','N/A')}")
            except WebSocketDisconnect:
                break
            except Exception as recv_err:
                if "1000" in str(recv_err):
                    break
                logger.error(f"接收消息失败: {recv_err}")
                break

            if ("method" not in data):
                msg_id = data.get("id", "-1")
                logger.info(f"📥 [Auto-无method消息] msg_id={msg_id} | 期望tool_id={tool_id}")
                logger.info(f"📥 [Auto-无method消息] 消息字段: {list(data.keys())}")
                logger.info(f"📥 [Auto-无method消息] 消息内容: {json.dumps(data, ensure_ascii=False)[:500]}")

                # 情况1：收到工具执行结果（匹配转发时的随机 ID）
                if str(msg_id) == str(tool_id):
                    logger.info(f"✅ [Auto-工具结果] ID匹配成功! msg_id={msg_id} == tool_id={tool_id}")
                    logger.info(f"✅ [Auto-工具结果] 开始处理工具返回结果")
                    response_id = f"resp_{uuid.uuid4().hex[:8]}"
                    message_id = f"msg_{uuid.uuid4().hex[:8]}"
                    start_time = time.time()
                    chunk_index = 0
                    response_started = False

                    # 构造提示消息（固定输出规范）
                    tool_result_message = f"""<tool> 工具回复消息:{str(data)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 重要提醒
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 此次为工具返回结果，不需要再调用工具
2. 如果回复为空，则告知用户工具未返回任何值，请检查参数是否正确

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 数据展示规范（严格执行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **数据真实性（最重要）**
   - 所有数据必须来自工具实际返回值
   - 禁止编造、猜测或使用示例数据
   - 工具返回什么就展示什么，不要修改数值

2. **表格格式**
   当数据适合用表格展示时：
   - 使用清晰的表头
   - 数值保留2位小数
   - 时间格式：yyyy-MM-dd HH:mm:ss

3. **图表可视化（强制要求）**
   ⚠️ 当数据包含 ≥3 个时间点时，**必须同时展示表格和图表**，缺一不可！

   图表格式（严格按此格式）：

   :::echarts
   ```json
   {{
     "title": {{
       "text": "图表标题",
       "left": "center"
     }},
     "tooltip": {{
       "trigger": "axis"
     }},
     "xAxis": {{
       "type": "category",
       "data": ["时间1", "时间2", "时间3"]
     }},
     "yAxis": {{
       "type": "value",
       "name": "单位"
     }},
     "series": [{{
       "name": "数据名称",
       "type": "line",
       "smooth": true,
       "data": [数值1, 数值2, 数值3]
     }}]
   }}
   ```
   :::

   图表类型选择：
   - 折线图（line）：温度变化、趋势分析（默认首选）
   - 柱状图（bar）：多设备对比、统计数据
   - 饼图（pie）：占比分布

   ❌ 禁止行为：只展示表格不展示图表
   ✅ 正确做法：表格 + 图表双重展示

4. **知识库检索结果**
   如果数据来自知识库检索，必须在最后注明来源文档：

   ---
   📚 参考文档：
   - 文档1：《xxx》
   - 文档2：《xxx》

请严格按照上述规范生成回复，基于工具返回的真实数据，不要编造数据。"""
                    

                    try:
                        async for item in jaip_handler.process_message_stream(
                            session_id=session_id,
                            message=tool_result_message,
                            context={}
                        ):
                            # 发送 response_start（仅首次）
                            if not response_started:
                                await _send_response_start(client_id, session_id, response_id, message_id)
                                response_started = True

                            # 发送文本块
                            await _send_text_chunk(client_id, session_id, response_id, item, chunk_index)
                            chunk_index += 1

                        # 发送 response_end（仅当有响应时）
                        # if response_started:
                        await _send_response_end(client_id, session_id, response_id, chunk_index, start_time)

                        # 保存工具调用历史到 Redis
                        if execute_id in tool_call_context:
                            tool_info = tool_call_context[execute_id]
                            try:
                                from app.core.tools.tool_selector import tool_selector
                                tool_selector.save_tool_call_history(
                                    session_id=session_id,
                                    tool_name=tool_info.get("toolName", ""),
                                    arguments=tool_info.get("arguments", {}),
                                    result=data.get("result", {}),
                                    agent_id=agent_id
                                )
                                logger.info(f"💾 已保存工具调用历史: {tool_info.get('toolName')}")
                            except Exception as save_err:
                                logger.warning(f"⚠️ 保存工具历史失败: {save_err}")

                        # 清理已处理的工具调用
                        tool_id = None
                        if execute_id in tool_call_context:
                            del tool_call_context[execute_id]

                    except Exception as e:
                        logger.error(f"处理工具结果失败: {e}", exc_info=True)
                        if response_started:
                            await _send_response_end(
                                client_id, session_id, response_id, chunk_index, start_time,
                                status="error", error=str(e)
                            )
                    continue

                # 情况2：自动工具触发时，JetLinks 平台响应"准备执行工具"（匹配 AI 的 execution_id）
                elif msg_id == execute_id:
                    # logger.info(f"自动执行工具参数获取，应该是发送工具执行的参数呢")
                    # 判断是否需要用户确认
                    ifwait = False
                    try:
                     if data["result"]["wait"] == True:
                         ifwait = True
                    except Exception as e:
                         pass

                    if not ifwait:
                        # 从保存的上下文中获取工具信息
                        tool_info = tool_call_context.get(execute_id)
                        if not tool_info:
                            logger.error(f"未找到工具调用上下文: {execute_id}")
                            continue

                        tool_name = tool_info.get("toolName")
                        arguments = tool_info.get("arguments", {})

                        if not tool_name:
                            logger.error(f"工具名称为空: {execute_id}")
                            continue

                        # 生成随机ID（转为字符串保持类型一致）
                        random_tool_id = str(random.randint(1, 999999))
                        tool_id = random_tool_id

                        # 先尝试执行内部工具（超时120秒，图片分析可能较慢）
                        user_query = tool_info.get("userQuery", "")  # 获取用户问题
                        tool_res = await _execute_tool_directly(
                            session_id=session_id,
                            tool_name=tool_name,
                            arguments=arguments,
                            jaip_handler=jaip_handler,
                            client_id=client_id,
                            request_id=random_tool_id,
                            agent_id=agent_id,
                            auto_send_response=True,  # 自动发送响应
                            user_query=user_query  # 传递用户问题
                        )
                        # 判断是否为内部工具且执行成功
                        is_internal_success = tool_res.get("success") and isinstance(tool_res.get("result"), dict)

                        # 如果result内部还有success字段，需要检查内层成功状态
                        # if is_internal_success and "success" in tool_res.get("result", {}):
                        #     is_internal_success = tool_res["result"].get("success", False)

                        if is_internal_success:
                            pass
                            # # 内部工具执行成功，转AI处理
                            # logger.info(f"✅ [内部工具] 执行成功，转AI处理: {tool_name}")
                            # await tool_result_to_ai_response(
                            #     tool_result=tool_res,
                            #     session_id=session_id,
                            #     client_id=client_id,
                            #     jaip_handler=jaip_handler,
                            #     execute_id=execute_id,
                            #     tool_call_context=tool_call_context,
                            #     agent_id=agent_id
                            # )
                        else:
                            # 外部工具或内部工具执行失败，转发到 JetLinks
                            if tool_res.get("success") == False:
                                logger.info(f"🌐 [Auto-外部工具] 转发到 JetLinks: {tool_name}")
                            else:
                                logger.warning(f"⚠️ [Auto-内部工具失败] 转发到 JetLinks 重试: {tool_name}")
                                logger.warning(f"⚠️ [Auto-内部工具失败] 原因: {tool_res.get('result', {}).get('error', 'unknown')}")

                            res = {
                                "jsonrpc": "2.0",
                                "id": random_tool_id,
                                "method": "tools.execute",
                                "params": {
                                    "toolName": tool_name.replace("_ai_service_#", ""),
                                    "arguments": arguments,
                                    "executionId": execute_id
                                }
                            }
                            logger.info(f"📤 [Auto-转发工具] 发送到前端: id={random_tool_id}, tool={tool_name}")
                            logger.info(f"📤 [Auto-转发工具] execution_id={execute_id}")
                            send_success = await manager.send_message(client_id, res)
                            if not send_success:
                                logger.error(f"❌ [Auto-转发工具] 发送失败，连接已断开: {random_tool_id}")
                            else:
                                logger.info(f"✅ [Auto-转发工具] 已发送，等待前端返回结果...")

                        continue

            # ═══════════════════════════════════════════════════════════════
            # 处理 JSON-RPC 格式消息
            # ═══════════════════════════════════════════════════════════════
            if ("jsonrpc" in data or "jaipId" in data) and ("method" in data):
                raw_method = data.get("method")
                method = _normalize_rpc_method(raw_method)
                params = data.get("params", {}) or {}
                request_id = data.get("id")

                if raw_method != method:
                    logger.info(f"处理方法: {raw_method} -> {method}, ID: {request_id}")
                else:
                    logger.info(f"处理方法: {method}, ID: {request_id}")

                # 会话初始化
                if method == "session.initialize":
                    params = _normalize_initialize_params(params)
                    session_id = await _handle_session_initialize(
                        params, request_id, agent_id, session_id,
                        session_data, jaip_handler, client_id
                    )
                    initialized = True
                    await _flush_pending_messages()

                # 处理用户消息
                elif method == "session.message":
                    if not initialized:
                        # 某些客户端不会显式发送 session.initialize（或初始化调用失败后重试只发 message）：
                        # 这里做一次“按需自动初始化”，尽量减少前端误报“未准备”。
                        if not auto_initialize_attempted:
                            auto_initialize_attempted = True
                            try:
                                session_id = await _handle_session_initialize(
                                    params=_normalize_initialize_params({}),
                                    request_id=f"auto_init_{uuid.uuid4().hex[:8]}",
                                    agent_id=agent_id,
                                    session_id=session_id,
                                    session_data=session_data,
                                    jaip_handler=jaip_handler,
                                    client_id=client_id,
                                    send_response=False,
                                )
                                if getattr(jaip_handler, "agent", None) is not None:
                                    initialized = True
                                    await _flush_pending_messages()
                            except Exception as e:
                                logger.warning("⚠️ [WebSocket-Auto] auto session.initialize failed: %s", e)

                    if not initialized:
                        if len(pending_rpc_messages) >= pending_rpc_messages_limit:
                            dropped_method, _, dropped_request_id = pending_rpc_messages.pop(0)
                            if dropped_request_id is not None:
                                await manager.send_message(client_id, {
                                    "jsonrpc": "2.0",
                                    "id": dropped_request_id,
                                    "error": {
                                        "code": -32002,
                                        "message": f"AI会话尚未准备就绪（已丢弃过早的 {dropped_method} 请求，请先完成 session.initialize/initialize 后重试）"
                                    }
                                })

                        # 提前 ACK，避免部分客户端因“未收到响应”误判为未准备
                        buffered_request_id = request_id
                        if request_id is not None:
                            await manager.send_message(client_id, {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": {"accepted": True, "buffered": True, "waitingForInitialize": True},
                            })
                            buffered_request_id = None

                        pending_rpc_messages.append((method, params, buffered_request_id))
                        logger.warning(
                            f"⏳ [WebSocket-Auto] 收到 session.message 但尚未初始化，已缓存 | "
                            f"client_id={client_id} | pending={len(pending_rpc_messages)}"
                        )
                        continue
                    exec_id = await _handle_session_message(
                        params, request_id, session_id,
                        agent_id, jaip_handler, client_id, tool_call_context
                    )
                    # 如果 AI 请求了工具调用，记录 execution_id
                    if exec_id:
                        execute_id = exec_id

                # 处理 schema.based.request（基于 schema 的结构化请求）
                elif method == "schema.based.request":
                    if not initialized:
                        if not auto_initialize_attempted:
                            auto_initialize_attempted = True
                            try:
                                session_id = await _handle_session_initialize(
                                    params=_normalize_initialize_params({}),
                                    request_id=f"auto_init_{uuid.uuid4().hex[:8]}",
                                    agent_id=agent_id,
                                    session_id=session_id,
                                    session_data=session_data,
                                    jaip_handler=jaip_handler,
                                    client_id=client_id,
                                    send_response=False,
                                )
                                if getattr(jaip_handler, "agent", None) is not None:
                                    initialized = True
                                    await _flush_pending_messages()
                            except Exception as e:
                                logger.warning("⚠️ [WebSocket-Auto] auto session.initialize failed: %s", e)

                    if not initialized:
                        if len(pending_rpc_messages) >= pending_rpc_messages_limit:
                            dropped_method, _, dropped_request_id = pending_rpc_messages.pop(0)
                            if dropped_request_id is not None:
                                await manager.send_message(client_id, {
                                    "jsonrpc": "2.0",
                                    "id": dropped_request_id,
                                    "error": {
                                        "code": -32002,
                                        "message": f"AI会话尚未准备就绪（已丢弃过早的 {dropped_method} 请求，请先完成 session.initialize/initialize 后重试）"
                                    }
                                })

                        # 提前 ACK，避免部分客户端因“未收到响应”误判为未准备
                        buffered_request_id = request_id
                        if request_id is not None:
                            await manager.send_message(client_id, {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": {"accepted": True, "buffered": True, "waitingForInitialize": True},
                            })
                            buffered_request_id = None

                        pending_rpc_messages.append((method, params, buffered_request_id))
                        logger.warning(
                            f"⏳ [WebSocket-Auto] 收到 schema.based.request 但尚未初始化，已缓存 | "
                            f"client_id={client_id} | pending={len(pending_rpc_messages)}"
                        )
                        continue
                    schema_name = params.get("schema")
                    response_type = params.get("type", "simple")
                    content = params.get("content", "")
                    msg_session_id = params.get("sessionId", session_id)

                    logger.info(f"[Schema Request] {schema_name}/{response_type}: {content[:100] if isinstance(content, str) else content}")

                    # 示例：根据 schema 类型选择不同的响应方式
                    #
                    # 方式1：非流式结构化响应（适用于查询类、表单类请求）
                    if schema_name == "device.status":
                        result_content = {
                            "deviceId": "device_001",
                            "status": "online",
                            "temperature": 25.5
                        }
                        await _send_schema_based_response(
                            client_id=client_id,
                            request_id=request_id,
                            session_id=msg_session_id,
                            content=result_content
                        )
                        continue
                    
                    # 方式2：流式AI响应（默认方式，适用于对话类请求）
                    message_params = {
                        "message": {"content": content if isinstance(content, str) else str(content)},
                        "sessionId": msg_session_id,
                        "context": params
                    }

                    exec_id = await _handle_session_message(
                        message_params, request_id, msg_session_id,
                        agent_id, jaip_handler, client_id, tool_call_context
                    )
                    if exec_id:
                        execute_id = exec_id

                # 工具执行
                elif method == "tools.execute":
                    random_tool_id = random.randint(1, 999999)

                    tool_id= random_tool_id

                    # 从 tool_call_context 获取用户问题
                    execution_id = params.get("executionId", "")
                    user_query_for_tool = ""
                    if execution_id and execution_id in tool_call_context:
                        user_query_for_tool = tool_call_context[execution_id].get("userQuery", "")

                    tool_res = await _execute_tool_directly(
                        session_id=session_id,
                        tool_name=params.get("toolName", ""),
                        arguments=params.get("arguments", {}),
                        jaip_handler=jaip_handler,
                        client_id=client_id,
                        request_id=request_id,
                        agent_id=agent_id,
                        auto_send_response=True,  # 自动发送 schema.based.response
                        user_query=user_query_for_tool  # 传递用户问题
                    )
                    logger.debug("工具执行结果: %s", tool_res)
                    if not tool_res.get("success"):
                        res ={
                        "jsonrpc": "2.0", 
                        "id": random_tool_id,  
                        "method": "tools.execute", 
                        "params":{
                            "toolName": params.get("toolName", "weather").replace("_ai_service_#", ""), 
                            "arguments": params.get("arguments", {"city": "重庆", "date": "2023-10-05"}),
                            "executionId": params.get("executionId", "exec_10c1d6b4"),
                            }
                            }
                        await manager.send_message(client_id, res)
                    else:
                        await tool_result_to_ai_response(
                            tool_result=tool_res,
                            session_id=session_id,
                            client_id=client_id,
                            jaip_handler=jaip_handler,
                            execute_id=execute_id,
                            tool_call_context=tool_call_context,
                            agent_id=agent_id
                        )
 

                # 心跳
                elif method == "session.ping":
                    await manager.send_message(client_id, {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": int(time.time() * 1000)
                    })

                else:
                    await manager.send_message(client_id, {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"method not found: {raw_method}"},
                    })

    except WebSocketDisconnect:
        logger.info(f"WebSocket 客户端断开: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}", exc_info=True)
    finally:
        manager.disconnect(client_id)
        try:
            if 'session_id' in locals() and session_id:
                jaip_handler.unregister_websocket(session_id)
        except Exception:
            pass



@router.get("/auto-connections")
async def get_active_connections():
    """获取当前活动的WebSocket连接"""
    return {
        "total": len(manager.active_connections),
        "connections": list(manager.active_connections.keys()),
        "sessions": {
            client_id: data
            for client_id, data in manager.session_data.items()
        }
    }
    
