"""
WebSocket接口 - JAIP协议实时通信（精简路由层）
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
import time
import re
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
import uuid
import asyncio

from app.core.jaip.handler import JAIPHandler
from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter()
MIN_INTERVAL_SECONDS = 10

_BACKEND_NAME_RE = re.compile(r"^\s*(?P<backend>cv|yolo|vlm)(?::(?P<model>[^\s]+))?", re.IGNORECASE)


def _normalize_backend_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v in {"cv", "yolo"}:
        return "cv"
    if v in {"vlm", "llm"}:
        return "vlm"
    return None


def _infer_backend_hint(params: Dict[str, Any]) -> Optional[str]:
    backend = _normalize_backend_name(
        params.get("backend") or params.get("engine") or params.get("detector")
    )
    if backend:
        return backend

    name = (params.get("name") or "").strip()
    if not name:
        cfg = params.get("configuration") or {}
        if isinstance(cfg, dict):
            name = (cfg.get("name") or "").strip()
    if name:
        m = _BACKEND_NAME_RE.match(name)
        if m:
            return _normalize_backend_name(m.group("backend"))

    sources = params.get("source") or []
    if isinstance(sources, list):
        for src in sources:
            if not isinstance(src, dict):
                continue
            b = _normalize_backend_name(src.get("backend") or src.get("engine") or src.get("detector"))
            if b:
                return b
            n = (src.get("name") or "").strip()
            if n:
                m = _BACKEND_NAME_RE.match(n)
                if m:
                    return _normalize_backend_name(m.group("backend"))
    return None

DisconnectHook = Union[Callable[[str], None], Callable[[str], Awaitable[None]]]

def _normalize_rpc_method(method: Optional[str]) -> Optional[str]:
    """兼容历史方法名（避免前端仍在用旧协议导致“未准备/无响应”）。"""
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

    def _value_type_to_schema(value_type: Any) -> Dict[str, Any]:
        """Convert JetLinks ValueType-like dict to a JSON-Schema-ish object (best-effort)."""
        if not isinstance(value_type, dict):
            return {"type": "string"}

        raw_type = value_type.get("type") or value_type.get("id") or "string"
        t = str(raw_type).strip().lower()
        if t in {"int"}:
            t = "integer"

        if t in {"string", "number", "integer", "boolean"}:
            return {"type": t}

        if t == "array":
            elem = value_type.get("elementType") or value_type.get("items") or {}
            return {"type": "array", "items": _value_type_to_schema(elem)}

        if t == "object":
            schema: Dict[str, Any] = {"type": "object", "properties": {}}
            props = value_type.get("properties")
            if isinstance(props, list):
                for prop in props:
                    if not isinstance(prop, dict):
                        continue
                    pid = prop.get("id")
                    if not pid:
                        continue
                    child_schema = _value_type_to_schema(prop.get("valueType") or prop.get("value_type"))
                    desc = prop.get("description") or prop.get("name")
                    if desc and isinstance(child_schema, dict):
                        child_schema = {**child_schema, "description": str(desc)}
                    schema["properties"][pid] = child_schema
            return schema

        # unknown / complex: fall back to string so the LLM can still fill something.
        return {"type": "string"}

    def _proxy_inputs_to_schema(inputs: Any) -> Dict[str, Any]:
        """Convert proxyCommand.inputs (list) to JSON Schema parameters."""
        schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        if not isinstance(inputs, list):
            return schema
        for item in inputs:
            if not isinstance(item, dict):
                continue
            pid = item.get("id")
            if not pid:
                continue
            prop_schema = _value_type_to_schema(item.get("valueType"))
            desc = item.get("description") or item.get("name")
            if desc and isinstance(prop_schema, dict):
                prop_schema = {**prop_schema, "description": str(desc)}
            schema["properties"][pid] = prop_schema
            if item.get("required") is True:
                schema["required"].append(pid)
        # keep required stable & unique
        if schema["required"]:
            seen = set()
            schema["required"] = [x for x in schema["required"] if x and not (x in seen or seen.add(x))]
        return schema

    def _flatten_tool_groups(tool_groups: Any) -> tuple[list[Dict[str, Any]], Dict[str, str]]:
        """Flatten params.tools/agentConfig.tools (proxyCommands format) to availableTools."""
        flattened: list[Dict[str, Any]] = []
        aliases: Dict[str, str] = {}
        real_id_counts: Dict[str, int] = {}

        if not isinstance(tool_groups, list):
            return flattened, aliases

        # First pass: collect counts of realCommandId for safe bare-alias mapping.
        for group in tool_groups:
            if not isinstance(group, dict):
                continue
            group_id = (group.get("id") or "").strip()
            proxy_cmds = group.get("proxyCommands") or group.get("proxy_commands") or []
            if not isinstance(proxy_cmds, list):
                continue
            for cmd in proxy_cmds:
                if not isinstance(cmd, dict):
                    continue
                real_id = (cmd.get("realCommandId") or cmd.get("real_command_id") or "").strip()
                if real_id:
                    real_id_counts[real_id] = real_id_counts.get(real_id, 0) + 1

        for group in tool_groups:
            if not isinstance(group, dict):
                continue
            group_id = (group.get("id") or "").strip()
            proxy_cmds = group.get("proxyCommands") or group.get("proxy_commands") or []
            if not isinstance(proxy_cmds, list):
                continue

            for cmd in proxy_cmds:
                if not isinstance(cmd, dict):
                    continue
                command_id = (cmd.get("commandId") or cmd.get("command_id") or "").strip()
                real_command_id = (cmd.get("realCommandId") or cmd.get("real_command_id") or "").strip()
                # Prefer stable executable id: groupId#commandId
                if not group_id or not command_id:
                    continue
                tool_id = f"{group_id}#{command_id}"

                parameters = _proxy_inputs_to_schema(cmd.get("inputs"))
                # Provide simplified inputs for compatibility (some frontends use inputs as whitelist).
                simplified_inputs = []
                if isinstance(cmd.get("inputs"), list):
                    for it in cmd.get("inputs") or []:
                        if not isinstance(it, dict):
                            continue
                        iid = it.get("id")
                        if not iid:
                            continue
                        vt = it.get("valueType") or {}
                        simplified_inputs.append({
                            "id": iid,
                            "name": it.get("name") or iid,
                            "type": (vt.get("type") or vt.get("id") or "string"),
                            "description": it.get("description") or it.get("name") or iid,
                            "required": bool(it.get("required") is True),
                        })

                tool_def: Dict[str, Any] = {
                    "id": tool_id,
                    "name": cmd.get("name") or real_command_id or command_id,
                    "description": cmd.get("description") or "",
                    "parameters": parameters,
                    "inputs": simplified_inputs,
                    "output": cmd.get("output"),
                    # Keep original identifiers for aliasing/debugging.
                    "toolGroupId": group_id,
                    "commandId": command_id,
                    "realCommandId": real_command_id,
                }
                flattened.append(tool_def)

                # Alias: groupId#RealCommandId -> groupId#commandId (helps prompts that mention realCommandId).
                if real_command_id:
                    aliases[f"{group_id}#{real_command_id}"] = tool_id
                    # Bare alias is only safe when unique across all tool groups in this init.
                    if real_id_counts.get(real_command_id) == 1 and real_command_id not in aliases:
                        aliases[real_command_id] = tool_id

        # Deduplicate by id (keep first occurrence).
        seen_ids = set()
        deduped: list[Dict[str, Any]] = []
        for item in flattened:
            tid = item.get("id")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            deduped.append(item)

        return deduped, aliases

    # tools/proxyCommands → availableTools (兼容 Java 中转的初始化结构)
    tool_groups = None
    if isinstance(normalized.get("tools"), list):
        tool_groups = normalized.get("tools")
    else:
        agent_cfg = normalized.get("agentConfig") if isinstance(normalized.get("agentConfig"), dict) else {}
        tool_groups = agent_cfg.get("tools") if isinstance(agent_cfg.get("tools"), list) else None

    flattened, aliases = _flatten_tool_groups(tool_groups)

    # Merge flattened tools into availableTools (do not override existing definitions).
    existing_tools = normalized.get("availableTools")
    merged_tools: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    if isinstance(existing_tools, list):
        for t in existing_tools:
            if not isinstance(t, dict):
                continue
            tid = t.get("id") or t.get("name")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(str(tid))
            merged_tools.append(t)
    for t in flattened:
        tid = t.get("id") or t.get("name")
        if not tid or str(tid) in seen_ids:
            continue
        seen_ids.add(str(tid))
        merged_tools.append(t)
    if merged_tools and merged_tools != existing_tools:
        normalized["availableTools"] = merged_tools

    if aliases:
        params_obj = normalized.get("parameters") if isinstance(normalized.get("parameters"), dict) else {}
        params_obj.setdefault("_tool_aliases", {})
        if isinstance(params_obj.get("_tool_aliases"), dict):
            params_obj["_tool_aliases"].update(aliases)
        normalized["parameters"] = params_obj

    return normalized


def _summarize_initialize_payload(params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {"type": str(type(params))}
    summary: Dict[str, Any] = {"keys": list(params.keys())[:50]}

    available_tools = params.get("availableTools")
    if isinstance(available_tools, list):
        summary["availableTools_len"] = len(available_tools)
        summary["availableTools_sample"] = [
            (t.get("id") or t.get("name")) if isinstance(t, dict) else str(t)
            for t in available_tools[:8]
        ]

    tool_groups = params.get("tools")
    if isinstance(tool_groups, list):
        groups = []
        for g in tool_groups[:8]:
            if not isinstance(g, dict):
                continue
            gid = g.get("id")
            proxy_cmds = g.get("proxyCommands") or g.get("proxy_commands") or []
            groups.append({"id": gid, "proxyCommands": len(proxy_cmds) if isinstance(proxy_cmds, list) else None})
        summary["tools_len"] = len(tool_groups)
        summary["tools_groups_sample"] = groups

    agent_cfg = params.get("agentConfig")
    if isinstance(agent_cfg, dict):
        summary["agentConfig_keys"] = list(agent_cfg.keys())[:30]
        cfg_tools = agent_cfg.get("tools")
        if isinstance(cfg_tools, list):
            summary["agentConfig_tools_len"] = len(cfg_tools)

    return summary


def _build_jsonrpc_error(
    request_id: Any,
    message: str,
    code: int = -32602,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """构造 JSON-RPC 错误响应"""
    error_body: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }
    if data:
        error_body["error"]["data"] = data
    return error_body


async def _handle_computer_vision_task(
    params: Dict[str, Any],
    request_id: Any,
    agent_id: str,
    session_id: str,
    jaip_handler: JAIPHandler,
    client_id: str
) -> None:
    """ComputerVisionTask 入口，负责轻量校验与转发"""
    if not isinstance(params, dict):
        await manager.send_message(client_id, _build_jsonrpc_error(request_id, "params 必须为对象"))
        return

    state = params.get("state")
    if state not in {"start", "stop"}:
        await manager.send_message(client_id, _build_jsonrpc_error(request_id, "state 仅支持 start/stop"))
        return

    sources = params.get("source") or []
    if not isinstance(sources, list) or not sources:
        await manager.send_message(client_id, _build_jsonrpc_error(request_id, "source 必须为非空数组"))
        return

    invalid_sources: List[Dict[str, Any]] = []
    for idx, src in enumerate(sources):
        if not isinstance(src, dict):
            invalid_sources.append({"index": idx, "reason": "source 必须为对象"})
            continue
        src_id = src.get("id")
        has_stream = bool((src.get("rtsp") or "").strip()) or bool((src.get("rtmp") or "").strip())
        if not src_id:
            invalid_sources.append({"index": idx, "reason": "缺少 id"})
        if not has_stream:
            invalid_sources.append({"index": idx, "reason": "需提供 rtsp 或 rtmp"})
    if invalid_sources:
        await manager.send_message(
            client_id,
            _build_jsonrpc_error(
                request_id,
                "source 参数不合法",
                data={"invalidSources": invalid_sources}
            )
        )
        return

    if state == "start":
        interval_val = params.get("interval", MIN_INTERVAL_SECONDS)
        try:
            interval_val = int(interval_val)
        except (TypeError, ValueError):
            await manager.send_message(client_id, _build_jsonrpc_error(request_id, "interval 必须为整数"))
            return
        if interval_val < MIN_INTERVAL_SECONDS:
            await manager.send_message(
                client_id,
                _build_jsonrpc_error(request_id, f"interval 最小 {MIN_INTERVAL_SECONDS} 秒")
            )
            return
        params["interval"] = interval_val

        backend_hint = _infer_backend_hint(params)

        use_content = bool(params.get("useContent", False))
        if use_content:
            content = (params.get("content") or "").strip()
            if not content:
                await manager.send_message(client_id, _build_jsonrpc_error(request_id, "useContent=true 时需提供 content"))
                return
        else:
            if backend_hint == "cv":
                pass
            else:
                missing_prompt_idx = [
                    idx for idx, src in enumerate(sources)
                    if not (src.get("prompt") or "").strip()
                ]
                if missing_prompt_idx:
                    await manager.send_message(
                        client_id,
                        _build_jsonrpc_error(
                            request_id,
                            "prompt 必填（或使用 useContent+content）",
                            data={"missingPromptIndexes": missing_prompt_idx}
                        )
                    )
                    return

    if not hasattr(jaip_handler, "handle_computer_vision_task"):
        await manager.send_message(
            client_id,
            _build_jsonrpc_error(request_id, "ComputerVisionTask 暂未实现", code=-32601)
        )
        return

    try:
        result = await jaip_handler.handle_computer_vision_task(
            params=params,
            request_id=request_id,
            agent_id=agent_id,
            session_id=session_id
        )
        if isinstance(result, dict) and result.get("suppress_ack"):
            return
        if result is None:
            result = {"accepted": True, "state": state}
        await manager.send_message(client_id, {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        })
    except Exception as e:
        logger.error(f"❌ [ComputerVisionTask] 处理失败: {e}", exc_info=True)
        await manager.send_message(
            client_id,
            _build_jsonrpc_error(request_id, f"ComputerVisionTask 处理失败: {e}", code=-32000)
        )


class ConnectionManager:
    """WebSocket连接管理器（支持断开 hook：断链时立即 stop 下游任务并清理文件）"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}

        # 断开连接 hook：client_id -> callable(client_id)
        self._disconnect_hooks: Dict[str, DisconnectHook] = {}
        # 已经判定死亡的 client，避免重复触发/刷屏
        self._dead_clients: set[str] = set()

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
        # 新连接，确保不在 dead 集合
        if client_id in self._dead_clients:
            self._dead_clients.discard(client_id)

        logger.info(f"✅ [ConnectionManager] 连接已建立: {client_id}")
        logger.info(f"📊 [ConnectionManager] 当前活动连接数: {len(self.active_connections)}")

    def register_disconnect_hook(self, client_id: str, hook: DisconnectHook) -> None:
        """注册断开 hook（只会触发一次）"""
        self._disconnect_hooks[client_id] = hook

    def _fire_disconnect_hook_once(self, client_id: str) -> None:
        if client_id in self._dead_clients:
            return
        self._dead_clients.add(client_id)

        hook = self._disconnect_hooks.get(client_id)
        if not hook:
            return

        try:
            ret = hook(client_id)
            if asyncio.iscoroutine(ret):
                asyncio.create_task(ret)
        except Exception as e:
            logger.warning("[ConnectionManager] disconnect hook error: %s", e)

    def disconnect(self, client_id: str):
        """断开WebSocket连接（并触发 hook）"""
        was_connected = client_id in self.active_connections

        # 先触发 hook（只一次）
        self._fire_disconnect_hook_once(client_id)

        # 再移除连接
        self.active_connections.pop(client_id, None)
        self.session_data.pop(client_id, None)
        self._disconnect_hooks.pop(client_id, None)

        if was_connected:
            logger.warning(f"❌ [ConnectionManager] 连接已断开: {client_id}")
            logger.info(f"📊 [ConnectionManager] 剩余活动连接数: {len(self.active_connections)}")

    async def send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """发送消息给特定客户端

        Returns:
            bool: 发送成功返回 True，失败返回 False
        """
        # 若已死亡，直接失败（避免刷屏）
        if client_id in self._dead_clients:
            return False

        websocket = self.active_connections.get(client_id)
        if not websocket:
            logger.warning(f"❌ [发送消息] 客户端不存在: {client_id}")
            logger.warning(f"📊 [发送消息] 当前活动连接: {list(self.active_connections.keys())}")
            # 把“不存在”视为断链：触发 stop/cleanup
            self.disconnect(client_id)
            return False

        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"❌ [发送消息] 发送失败(连接可能已关闭): {client_id}, error: {e}")
            logger.info(f"🔧 [发送消息] 将断开连接: {client_id}")
            self.disconnect(client_id)
            return False

    def get_session_data(self, client_id: str) -> Dict[str, Any]:
        """获取会话数据"""
        return self.session_data.get(client_id, {})


# 创建连接管理器实例
manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════
# WebSocket 端点
# ═══════════════════════════════════════════════════════════════
@router.websocket("/session/{agent_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    agent_id: str,
):
    """
    WebSocket端点 - JAIP协议实时通信

    路径: /api/v1/jaip/session/{agent_id}
    """
    client_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"
    logger.info(f"🔌 [WebSocket] 新连接请求 | agent_id={agent_id} | client_id={client_id}")

    # 创建 JAIP 处理器（注入 WebSocket 管理器）
    try:
        jaip_handler = JAIPHandler(websocket_manager=manager, client_id=client_id)
    except Exception as e:
        logger.error(f"❌ [WebSocket] 初始化 JAIP 处理器失败: {e}")
        await websocket.close(code=1011, reason=str(e))
        return

    try:
        # 建立连接
        await manager.connect(websocket, client_id)
        logger.info(f"✅ [WebSocket] 连接已建立 | client_id={client_id}")

        # ✅ 注册断开 hook：一旦断链/发送失败，立即 stop 下游任务并清理 storage
        async def _on_disconnect(cid: str):
            try:
                await jaip_handler.cleanup_async(reason="ws disconnected")
            except Exception as e:
                logger.warning("[WebSocket] cleanup_async failed on disconnect hook: %s", e)

        manager.register_disconnect_hook(client_id, _on_disconnect)

        session_data = manager.get_session_data(client_id)
        session_id = session_data["session_id"]

        # 发送欢迎消息
        await manager.send_message(client_id, {
            "type": "connected",
            "message": f"已连接到智能体 {agent_id}",
            "session_id": session_id,
            "agent_id": agent_id
        })

        # 兼容客户端乱序：某些情况下会先发 session.message 再发 session.initialize
        initialized = False
        auto_initialize_attempted = False
        pending_session_messages: list[tuple[Dict[str, Any], Any]] = []
        pending_session_messages_limit = 10

        # 消息循环
        while True:
            try:
                data = await websocket.receive_json()

                message_keys = list(data.keys())
                if len(message_keys) > 2:  # 只有多于traceId和jsonrpc的消息才记录
                    logger.info(f"📨 [WebSocket] 收到消息 | keys={message_keys} | id={data.get('id','N/A')}")
            except WebSocketDisconnect as e:
                code = getattr(e, "code", None)
                reason = getattr(e, "reason", None)
                logger.info(f"🔌 [WebSocket] 连接已断开 | client_id={client_id} | code={code} | reason={reason}")
            except WebSocketDisconnect as e:
                code = getattr(e, "code", None)
                reason = getattr(e, "reason", None)
                logger.info(f"🔌 [WebSocket] 连接已断开 | client_id={client_id} | code={code} | reason={reason}")
                break
            except Exception as recv_err:
                if "1000" in str(recv_err):
                    break
                logger.error(f"接收消息失败: {recv_err}")
                break

            # ═══════════════════════════════════════════════════════════════
            # 处理无 method 的消息（工具结果响应）
            # ═══════════════════════════════════════════════════════════════
            if "method" not in data:
                msg_id = data.get("id", "-1")

                has_result = "result" in data
                has_params = "params" in data
                has_data = "data" in data
                has_error = "error" in data

                # 如果消息只包含 traceId/jsonrpc，没有实际内容，直接忽略
                if not has_result and not has_params and not has_data and not has_error:
                    # Some relays send a "complete-only" ACK for tools.execute (especially when the external
                    # tool returns void). If the id matches a pending tool execution, treat it as an empty
                    # tool result to avoid the generator hanging until watchdog timeout.
                    try:
                        complete_val = data.get("complete")
                        complete_flag = complete_val is True
                        if not complete_flag and complete_val is not None:
                            try:
                                s = str(complete_val).strip().lower()
                                complete_flag = s in {"true", "1", "yes", "y", "ok", "done", "complete"}
                            except Exception:
                                complete_flag = False

                        execution_id = None
                        try:
                            if str(msg_id) in getattr(jaip_handler, "_pending_tool_results", {}):
                                execution_id = getattr(jaip_handler, "_pending_tool_results", {}).get(str(msg_id))
                            elif str(msg_id) in getattr(jaip_handler, "_pending_tools", {}):
                                execution_id = str(msg_id)
                        except Exception:
                            execution_id = None

                        # Only synthesize a tool result when the external-tool watchdog is active, meaning we
                        # have already forwarded `tools.execute` and are waiting for a result payload.
                        watching = False
                        try:
                            watchers = getattr(jaip_handler, "_external_tool_watchers", {}) or {}
                            watching = bool(execution_id) and str(execution_id) in watchers
                        except Exception:
                            watching = False

                        if watching and str(msg_id) != "-1":
                            tool_name = None
                            try:
                                ctx = (getattr(jaip_handler, "_pending_tools", {}) or {}).get(str(execution_id)) or {}
                                if isinstance(ctx, dict):
                                    tool_name = ctx.get("toolName")
                            except Exception:
                                tool_name = None

                            logger.info(
                                "📦 [WebSocket] 收到空工具结果（complete-only ack），继续恢复流程 | id=%s | execution_id=%s | tool=%s | complete=%r",
                                msg_id,
                                execution_id,
                                tool_name,
                                complete_val,
                            )
                            await jaip_handler.handle_tool_result(
                                str(msg_id),
                                {
                                    "jsonrpc": "2.0",
                                    "id": str(msg_id),
                                    "result": {
                                        "success": True,
                                        "result": None,
                                        "complete_only_ack": True,
                                        "executionId": execution_id,
                                    },
                                },
                                agent_id,
                            )
                    except Exception:
                        pass
                    continue

                # 如果有 error 字段，优先处理工具执行错误
                if "error" in data:
                    error_info = data.get("error")
                    logger.error(f"❌ [WebSocket] 收到工具执行错误 | id={msg_id} | error={error_info}")

                    # 将错误转换为工具结果格式，继续处理流程避免堵塞
                    error_message = None
                    error_code = None
                    error_data = None
                    if isinstance(error_info, dict):
                        error_message = error_info.get("message")
                        error_code = error_info.get("code")
                        error_data = error_info.get("data")
                    if not error_message:
                        error_message = str(error_info)

                    error_result = {
                        "result": {
                            "success": False,
                            "error": error_message,
                            "error_code": error_code,
                            "error_data": error_data,
                            "is_external_tool_error": True,
                        },
                        "id": msg_id,
                        "jsonrpc": "2.0"
                    }
                    logger.info(f"🔄 [WebSocket] 转换错误为工具结果格式，继续处理流程")
                    await jaip_handler.handle_tool_result(msg_id, error_result, agent_id)
                    continue

                # 如果有 result 字段，判断是确认消息还是工具结果
                if has_result:
                    result_value = data.get("result")

                    pending_exec = str(msg_id) in getattr(jaip_handler, "_pending_tools", {})

                    # tools.confirm 的第一阶段响应（Java中转常返回 {"wait": true} 表示已推送到前端，等待用户确认）
                    if isinstance(result_value, dict) and "wait" in result_value:
                        wait_flag = result_value.get("wait")
                        if wait_flag is True:
                            logger.info(
                                "⏳ [WebSocket] 收到 tools.confirm 等待响应 | id=%s | wait=true",
                                msg_id,
                            )
                            continue

                        logger.info(
                            "✅ [WebSocket] 收到 tools.confirm 确认响应 | id=%s | wait=%s",
                            msg_id,
                            wait_flag,
                        )
                        try:
                            await jaip_handler.handle_tool_confirmation(str(msg_id), result_value)
                        except Exception as confirm_err:
                            logger.error(
                                "❌ [WebSocket] 处理工具确认失败 | id=%s | err=%s",
                                msg_id,
                                confirm_err,
                                exc_info=True,
                            )
                        continue

                    # 兼容部分客户端直接返回布尔值 true/false
                    if pending_exec and result_value is True:
                        logger.info(f"✅ [WebSocket] 收到工具确认(true) | id={msg_id}")
                        try:
                            await jaip_handler.handle_tool_confirmation(str(msg_id), {"wait": False})
                        except Exception as confirm_err:
                            logger.error(
                                "❌ [WebSocket] 处理工具确认失败 | id=%s | err=%s",
                                msg_id,
                                confirm_err,
                                exc_info=True,
                            )
                        continue

                    if pending_exec and result_value is False:
                        logger.info(f"🛑 [WebSocket] 收到工具取消(false) | id={msg_id}")
                        cancel_result = {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "success": False,
                                "error": "用户取消工具调用",
                                "cancelled": True,
                            },
                        }
                        await jaip_handler.handle_tool_result(msg_id, cancel_result, agent_id)
                        continue

                    logger.info(f"📦 [WebSocket] 收到工具结果 | id={msg_id}")
                    await jaip_handler.handle_tool_result(msg_id, data, agent_id)
                    continue

                logger.info(f"📦 [WebSocket] 收到其他消息（可能是工具结果）| id={msg_id}")
                await jaip_handler.handle_tool_result(msg_id, data, agent_id)
                continue

            # ═══════════════════════════════════════════════════════════════
            # 处理 JSON-RPC 格式消息
            # ═══════════════════════════════════════════════════════════════
            else:
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
                    try:
                        logger.info("🧩 [session.initialize] raw params summary: %s", _summarize_initialize_payload(params))
                    except Exception:
                        pass
                    params = _normalize_initialize_params(params)
                    try:
                        logger.info(
                            "🧩 [session.initialize] normalized params summary: %s",
                            _summarize_initialize_payload(params),
                        )
                    except Exception:
                        pass
                    try:
                        init_result = await jaip_handler.handle_initialize(
                            params=params,
                            request_id=request_id,
                            agent_id=agent_id,
                            session_id=session_id
                        )
                        await manager.send_message(client_id, {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": init_result
                        })
                        initialized = True
                    except Exception as e:
                        logger.error(f"❌ [WebSocket] session.initialize 失败: {e}", exc_info=True)
                        await manager.send_message(
                            client_id,
                            _build_jsonrpc_error(request_id, f"session.initialize 失败: {e}", code=-32000)
                        )
                        continue

                    # 初始化完成后，处理之前缓存的消息（保持原始顺序）
                    if pending_session_messages:
                        logger.warning(
                            f"🔄 [WebSocket] 初始化后处理缓存消息: {len(pending_session_messages)} 条 | client_id={client_id}"
                        )
                    while pending_session_messages:
                        buffered_params, buffered_request_id = pending_session_messages.pop(0)
                        # JSON-RPC ack (buffered session.message)
                        if buffered_request_id is not None:
                            await manager.send_message(client_id, {
                                'jsonrpc': '2.0',
                                'id': buffered_request_id,
                                'result': {'accepted': True, 'buffered': True}
                            })
                        await jaip_handler.handle_message(
                            params=buffered_params,
                            request_id=buffered_request_id,
                            session_id=session_id
                        )

                # 处理用户消息（流式）
                elif method == "session.message":
                    if not initialized:
                        # 某些客户端不会显式发送 session.initialize（或初始化调用失败后重试只发 message）：
                        # 这里做一次“按需自动初始化”，尽量减少前端误报“未准备”。
                        if not auto_initialize_attempted:
                            auto_initialize_attempted = True
                            try:
                                _ = await jaip_handler.handle_initialize(
                                    params=_normalize_initialize_params({}),
                                    request_id=str(uuid.uuid4()),
                                    agent_id=agent_id,
                                    session_id=session_id,
                                )
                                initialized = True
                            except Exception as e:
                                logger.warning(
                                    "⚠️ [WebSocket] auto session.initialize failed: %s",
                                    e,
                                )

                        if initialized:
                            # 初始化成功后，先 flush 之前缓存的消息（保持顺序）
                            if pending_session_messages:
                                logger.warning(
                                    f"🔄 [WebSocket] auto-init 后处理缓存消息: {len(pending_session_messages)} 条 | client_id={client_id}"
                                )
                            while pending_session_messages:
                                buffered_params, buffered_request_id = pending_session_messages.pop(0)
                                if buffered_request_id is not None:
                                    await manager.send_message(client_id, {
                                        "jsonrpc": "2.0",
                                        "id": buffered_request_id,
                                        "result": {"accepted": True, "buffered": True},
                                    })
                                await jaip_handler.handle_message(
                                    params=buffered_params,
                                    request_id=buffered_request_id,
                                    session_id=session_id,
                                )

                    if not initialized:
                        if len(pending_session_messages) >= pending_session_messages_limit:
                            dropped_params, dropped_request_id = pending_session_messages.pop(0)
                            if dropped_request_id is not None:
                                await manager.send_message(
                                    client_id,
                                    _build_jsonrpc_error(
                                        dropped_request_id,
                                        "AI会话尚未准备就绪（已丢弃过早的请求，请先完成 session.initialize/initialize 后重试）",
                                        code=-32002
                                    )
                                )
                        # 提前 ACK，避免部分客户端因“未收到响应”误判为未准备
                        buffered_request_id = request_id
                        if request_id is not None:
                            await manager.send_message(client_id, {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": {"accepted": True, "buffered": True, "waitingForInitialize": True},
                            })
                            # 已ACK过，后续 flush 不再重复 ACK
                            buffered_request_id = None

                        pending_session_messages.append((params, buffered_request_id))
                        logger.warning(
                            f"⏳ [WebSocket] 收到 session.message 但尚未初始化，已缓存 | client_id={client_id} | pending={len(pending_session_messages)}"
                        )
                        continue

                    # JSON-RPC ack (session.message)
                    if request_id is not None:
                        await manager.send_message(client_id, {
                            'jsonrpc': '2.0',
                            'id': request_id,
                            'result': {'accepted': True}
                        })

                    await jaip_handler.handle_message(
                        params=params,
                        request_id=request_id,
                        session_id=session_id
                    )

                # 工具执行
                elif method == "tools.execute":
                    logger.info(f"🔧 [WebSocket] 收到 tools.execute 消息")
                    logger.info(f"🔧 [WebSocket] params: {json.dumps(params, ensure_ascii=False)[:500]}")

                    # JSON-RPC ack: some relays treat tools.execute as a request and expect a response.
                    # We keep it lightweight and continue the actual tool flow via the existing
                    # confirm/execute/result messages (executionId-based).
                    if request_id is not None:
                        try:
                            await manager.send_message(client_id, {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": {"accepted": True},
                            })
                        except Exception:
                            pass

                    await jaip_handler.handle_tool_execute(
                        params=params,
                        agent_id=agent_id,
                    )

                # 心跳
                elif method == "session.ping":
                    await manager.send_message(client_id, {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": int(time.time() * 1000)
                    })

                # 机器视觉任务
                elif method == "ComputerVisionTask":
                    logger.info(f"📹 [WebSocket] 收到 ComputerVisionTask | id={request_id}")
                    await _handle_computer_vision_task(
                        params=params,
                        request_id=request_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        jaip_handler=jaip_handler,
                        client_id=client_id
                    )

                else:
                    # 未知方法：显式返回错误，避免客户端“卡住”后误报“未准备”
                    await manager.send_message(
                        client_id,
                        _build_jsonrpc_error(
                            request_id,
                            f"method not found: {raw_method}",
                            code=-32601,
                        ),
                    )

    except WebSocketDisconnect as e:
        code = getattr(e, "code", None)
        reason = getattr(e, "reason", None)
        logger.info(f"🔌 [WebSocket] 客户端断开: {client_id} | code={code} | reason={reason}")
    except Exception as e:
        logger.error(f"❌ [WebSocket] 连接异常: {client_id} | error: {e}", exc_info=True)
    finally:
        logger.info(f"🧹 [WebSocket] 开始清理连接: {client_id}")
        # disconnect 会触发 hook：stop 下游 + 清理 storage
        manager.disconnect(client_id)
        try:
            await jaip_handler.cleanup_async(reason="endpoint finally")
        except Exception:
            pass
        logger.info(f"✅ [WebSocket] 清理完成: {client_id}")


@router.get("/connections")
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
