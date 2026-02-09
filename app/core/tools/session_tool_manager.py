"""
会话工具管理器 - 动态工具注册和管理

功能说明:
- 管理通过WebSocket动态传入的客户端工具
- 为每个会话维护独立的工具集
- 将客户端工具适配为BaseTool格式
- 支持工具调用的WebSocket回调机制

调用链路:
1. WebSocket收到tools定义 → register_session_tools()
2. Agent需要工具 → get_session_tools() → 返回工具实例
3. 工具执行 → SessionTool.call() → WebSocket回调 → 客户端执行
4. 客户端返回结果 → 解析回调 → 返回给Agent

核心功能:
- register_session_tools(): 注册会话工具
- get_session_tools(): 获取会话的所有工具
- update_session_tools(): 更新工具列表
- clean_session_tools(): 清理会话工具

工作原理:
1. 客户端通过WebSocket发送工具定义（id, name, parameters等）
2. SessionToolManager创建SessionTool实例包装
3. Agent调用工具时，通过WebSocket回调客户端执行
4. 实现了服务端Agent调用客户端工具的能力
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union


from app.core.tools.base import BaseTool

logger = logging.getLogger(__name__)


class SessionTool(BaseTool):
    """会话工具适配器 - 用于WebSocket传入的工具"""

    @staticmethod
    def _normalize_tool_name(tool_name: str) -> str:
        """Normalize tool id for loose matching (case/underscore insensitive)."""
        if not tool_name:
            return ""
        return re.sub(r"[^0-9a-zA-Z]+", "", tool_name).lower()

    def _resolve_registered_tool_name(self, tool_name: str, registry: Dict[str, Any]) -> str:
        """Resolve actual tool id in TOOL_REGISTRY for legacy/alias ids."""
        if tool_name in registry:
            return tool_name

        target = self._normalize_tool_name(tool_name)
        if not target:
            return tool_name

        matches = [k for k in registry.keys() if self._normalize_tool_name(k) == target]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            for k in matches:
                if k.lower() == tool_name.lower():
                    return k
            return matches[0]
        return tool_name
    
    def __init__(self, tool_def: Dict[str, Any], session_id: str, ws_handler=None):
        """初始化会话工具
        
        Args:
            tool_def: 工具定义（WebSocket传入的格式）
            session_id: 会话ID
            ws_handler: WebSocket处理器，用于回调
        """
        self.tool_def = tool_def
        self.session_id = session_id
        self.ws_handler = ws_handler

        # 设置工具基本属性（兼容 id 和 name 字段）
        self.name = tool_def.get('id') or tool_def.get('name', '')
        self.display_name = (
            tool_def.get("name")
            or tool_def.get("displayName")
            or tool_def.get("display_name")
            or tool_def.get("title")
            or self.name
        )
        self.description = tool_def.get('description', '')
        self.async_mode = tool_def.get('async', False)

        # 设置参数定义
        self.inputs = tool_def.get('inputs', [])
        self.output = tool_def.get('output', {
            'id': 'result',
            'name': '结果',
            'type': 'object'
        })

        # 保存原始 parameters（包含所有扩展字段，用于后续使用）
        self.raw_parameters = tool_def.get('parameters', {})

        # 额外属性
        self.category = tool_def.get('category', 'external')
        self.version = tool_def.get('version', '1.0.0')

        # 处理 parameters：
        # 为 qwen-agent 生成干净的 OpenAI 兼容 JSON Schema
        # 同时保留原始的 raw_parameters 供业务逻辑使用
        if 'parameters' in tool_def and isinstance(tool_def['parameters'], dict):
            # 客户端已经提供了 JSON Schema 格式的 parameters
            self.raw_parameters = tool_def['parameters'].copy()  # 保存原始参数

            # 🆕 调试：检查原始参数中的默认值
            if self.name == 'deviceService:device#GetMetadata':
                logger.info(f"🔧 [工具调试] {self.name} 原始 parameters: {json.dumps(self.raw_parameters, ensure_ascii=False, indent=2)}")

            # 为 qwen-agent 清理非标准字段，生成兼容版本
            cleaned_parameters = self._clean_parameters_for_llm(tool_def['parameters'])

            # ✅ 兼容：有些客户端会同时提供 inputs + parameters，但 parameters 可能缺字段。
            # 这里用 inputs 补齐 properties/required，避免出现“后端必填但 schema 未声明”的缺参问题。
            if self.inputs:
                try:
                    inputs_parameters = self._convert_inputs_to_parameters(self.inputs)
                    cleaned_parameters = self._merge_parameters(cleaned_parameters, inputs_parameters)
                except Exception as merge_error:
                    logger.debug("⚠️ tools.parameters merge inputs failed for %s: %s", self.name, merge_error)

            self.parameters = cleaned_parameters

            # 🆕 调试：检查清理后的参数中的默认值
            if self.name == 'deviceService:device#GetMetadata':
                logger.info(f"🔧 [工具调试] {self.name} 清理后 parameters: {json.dumps(self.parameters, ensure_ascii=False, indent=2)}")
                props = self.parameters.get('properties', {})
                for param_name, param_def in props.items():
                    default_val = param_def.get('default', 'NO_DEFAULT')
                    logger.info(f"🔧 [工具调试] {self.name} 参数 '{param_name}': 默认值={default_val}")

            logger.info(f"工具 {self.name} 使用客户端提供的 parameters（已为 LLM 清理）")
        else:
            # 将 inputs 转换为 qwen-agent 需要的 parameters 格式（JSON Schema）
            self.parameters = self._convert_inputs_to_parameters(self.inputs)
            logger.info(f"工具 {self.name} 从 inputs 转换生成 parameters")

        # 记录工具注册信息（调试用 - DEBUG 级别）
        logger.debug(f"初始化会话工具: {self.name}")
        # logger.debug(f"  描述: {self.description}")
        # logger.debug(f"  参数 (parameters): {json.dumps(self.parameters, ensure_ascii=False, indent=2)}")

        super().__init__()

    def _clean_parameters_for_llm(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """清理 parameters 以生成 OpenAI 兼容的 JSON Schema（仅用于 LLM）

        注意：经过测试，qwen-agent 实际上接受大部分扩展字段！
        只有以下字段会导致验证失败，需要移除：
        - returns (工具返回值定义，不属于输入参数)

        可以保留的字段：
        - $id (qwen-agent 会忽略它，但不报错)
        - title (标准 JSON Schema 字段)
        - description (标准 JSON Schema 字段)
        - x-* (扩展字段，qwen-agent 会忽略但不报错)

        原始的 parameters（包含所有字段）保存在 self.raw_parameters 中，
        供业务逻辑使用。

        Args:
            parameters: 原始 parameters 字典

        Returns:
            清理后的 parameters 字典（仅移除 returns）
        """
        # 创建副本
        cleaned_params = parameters.copy()

        # 只移除顶层的 returns 字段（这是唯一会导致验证失败的）
        if 'returns' in cleaned_params:
            del cleaned_params['returns']

        # 确保有 required 字段（即使是空数组）
        if 'required' not in cleaned_params:
            cleaned_params['required'] = []

        # 确保有 type 字段
        if 'type' not in cleaned_params:
            cleaned_params['type'] = 'object'

        return cleaned_params

    def _convert_inputs_to_parameters(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将客户端的 inputs 格式转换为 qwen-agent 需要的 JSON Schema parameters 格式

        Args:
            inputs: 客户端工具定义中的 inputs 列表
                    格式: [{"id": "param1", "name": "参数1", "type": "string", "required": true}, ...]

        Returns:
            JSON Schema 格式的 parameters
                    格式: {"type": "object", "properties": {...}, "required": [...]}
        """
        properties = {}
        required = []

        for input_def in inputs:
            param_id = input_def.get('id', '')
            param_name = input_def.get('name', param_id)
            param_type = input_def.get('type', 'string')
            param_desc = input_def.get('description', param_name)
            is_required = input_def.get('required', False)

            # 映射类型到 JSON Schema 类型
            type_mapping = {
                'string': 'string',
                'number': 'number',
                'integer': 'integer',
                'int': 'integer',
                'boolean': 'boolean',
                'array': 'array',
                'object': 'object'
            }
            json_type = type_mapping.get(param_type.lower(), 'string')

            # 构建属性定义
            properties[param_id] = {
                'type': json_type,
                'description': param_desc
            }

            # 添加到 required 列表
            if is_required:
                required.append(param_id)

        # 返回 JSON Schema 格式
        return {
            'type': 'object',
            'properties': properties,
            'required': required
        }

    def _merge_parameters(self, base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two JSON-Schema-like parameter definitions (best-effort).

        Rules:
        - properties: base wins on key conflicts
        - required: union (stable order)
        """
        if not isinstance(base, dict):
            base = {}
        if not isinstance(extra, dict):
            return base

        merged = dict(base)

        base_props = merged.get("properties") if isinstance(merged.get("properties"), dict) else {}
        extra_props = extra.get("properties") if isinstance(extra.get("properties"), dict) else {}
        props = dict(base_props)
        for k, v in extra_props.items():
            if k not in props:
                props[k] = v
        if props:
            merged["properties"] = props

        base_required = merged.get("required") if isinstance(merged.get("required"), list) else []
        extra_required = extra.get("required") if isinstance(extra.get("required"), list) else []
        required = []
        for item in list(base_required) + list(extra_required):
            if item and item not in required:
                required.append(item)
        merged["required"] = required

        if not merged.get("type"):
            merged["type"] = extra.get("type") or "object"

        return merged

    def run(self, **kwargs: Any) -> Any:
        """执行工具 - 优先检查是否为内部工具，否则通过WebSocket回调到客户端

        Args:
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        try:
            # ═══════════════════════════════════════════════════════════
            # 【关键修复】检查是否为内部工具
            # ═══════════════════════════════════════════════════════════
            from app.core.tools import internal_tool_registry

            # 如果是内部工具（如 _ai_service_#chatbi），直接执行而不是通过WebSocket
            if internal_tool_registry.is_internal_tool(self.name):
                logger.info(f"🔧 检测到内部工具: {self.name}，直接执行")

                # 获取实际的工具实例并执行
                from app.core.tools import TOOL_REGISTRY

                # 去掉 _ai_service_# 前缀获取实际tool_id
                actual_tool_id = self.name.replace("_ai_service_#", "", 1) if self.name.startswith("_ai_service_#") else self.name

                resolved_tool_id = self._resolve_registered_tool_name(actual_tool_id, TOOL_REGISTRY)
                if resolved_tool_id != actual_tool_id:
                    logger.info(f"🔁 工具别名映射: {actual_tool_id} -> {resolved_tool_id}")

                if resolved_tool_id in TOOL_REGISTRY:
                    # TOOL_REGISTRY 存储的是工具类，需要实例化
                    tool_class = TOOL_REGISTRY[resolved_tool_id]
                    actual_tool = tool_class()  # 实例化工具
                    logger.info(f"   找到工具类并实例化: {resolved_tool_id}")

                    # 直接调用工具的call方法（kwargs作为第一个位置参数传递给params）
                    result = actual_tool.call(kwargs)
                    logger.info(f"   工具 {resolved_tool_id} 执行完成")
                    return result
                else:
                    logger.error(f"   未找到工具实例: {actual_tool_id}")
                    return {
                        "status": "error",
                        "error": f"内部工具 {actual_tool_id} 未在TOOL_REGISTRY中注册"
                    }

            # 外部工具：继续原有的WebSocket回调逻辑
            if not self.ws_handler:
                logger.warning(f"会话工具 {self.name} 没有WebSocket处理器，返回模拟结果")
                return {
                    "status": "success",
                    "message": f"Tool {self.name} called with params: {kwargs}",
                    "result": {}
                }

            import asyncio
            import threading

            # 记录当前线程信息
            current_thread = threading.current_thread()
            logger.info(f"工具 {self.name} 在线程 {current_thread.name} (ID: {current_thread.ident}) 中被调用")
            # logger.info(f"🔍 [DEBUG] 工具调用 session_id: {self.session_id}")
            # logger.info(f"🔍 [DEBUG] 工具调用参数 kwargs: {kwargs}")
            # logger.info(f"🔍 [DEBUG] ws_handler 类型: {type(self.ws_handler).__name__ if self.ws_handler else 'None'}")

            # 检查是否有运行中的事件循环
            try:
                loop = asyncio.get_running_loop()
                logger.info(f"检测到运行中的事件循环: {loop}")

                # ⚠️ 关键修复：在同一个事件循环中不能使用 run_coroutine_threadsafe + result()
                # 这会导致死锁！应该使用 asyncio.create_task 或直接 await
                logger.warning(f"⚠️ 工具 {self.name} 在事件循环中被同步调用，这可能导致死锁！")
                logger.warning(f"   建议：修改 qwen-agent 工具调用为异步模式")

                # 临时解决方案：在新线程中运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    logger.info(f"在新线程中执行工具调用")
                    future = executor.submit(
                        self.ws_handler.call_client_tool,
                        self.session_id,
                        self.name,
                        kwargs
                    )
                    result = future.result(timeout=60)
                    logger.info(f"工具 {self.name} 调用成功（线程池模式）")
                    return result

            except RuntimeError as e:
                if "no running event loop" not in str(e).lower():
                    # 非预期的 RuntimeError
                    raise

                # 没有运行中的事件循环
                logger.info("没有检测到运行中的事件循环")

            # 不在事件循环中，使用同步方法
            logger.info(f"使用同步方法调用工具 {self.name}")
            result = self.ws_handler.call_client_tool(
                self.session_id,
                self.name,
                kwargs
            )
            logger.info(f"工具 {self.name} 调用成功（同步模式）")
            return result

        except Exception as e:
            logger.error(f"执行会话工具 {self.name} 失败: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def call(self, params: Union[str, dict], **kwargs) -> Any:
        """qwen-agent 调用工具的接口（继承自 BaseTool）

        Args:
            params: 参数（JSON 字符串或字典）
            **kwargs: 额外参数

        Returns:
            工具执行结果
        """
        # 解析参数
        if isinstance(params, str):
            import json
            params_dict = json.loads(params)
        else:
            params_dict = params

        # 调用 run 方法
        return self.run(**params_dict)

    def get_definition(self) -> Dict[str, Any]:
        """获取工具定义（返回完整的原始定义，包含所有扩展字段）"""
        return {
            'id': self.name,
            'name': self.tool_def.get('name', self.name),
            'description': self.description,
            'parameters': self.raw_parameters,  # 返回原始的 parameters（包含所有字段）
            'inputs': self.inputs,
            'output': self.output,
            'async': self.async_mode,
            'category': self.category,
            'version': self.version
        }


class SessionToolManager:
    """会话工具管理器 - 管理每个会话的工具列表"""
    
    def __init__(self):
        """初始化会话工具管理器"""
        # 存储每个会话的工具 {session_id: {tool_id: SessionTool}}
        self.session_tools: Dict[str, Dict[str, SessionTool]] = {}
        
    def register_session_tools(self, session_id: str, tools: List[Dict[str, Any]], ws_handler=None):
        """注册会话工具
        
        Args:
            session_id: 会话ID
            tools: 工具定义列表
            ws_handler: WebSocket处理器
        """
        if session_id not in self.session_tools:
            self.session_tools[session_id] = {}
        
        for tool_def in tools:
            try:
                # 兼容多种格式：优先使用 id，如果没有则使用 name
                tool_id = tool_def.get('id') or tool_def.get('name')
                if not tool_id:
                    logger.warning(f"工具定义缺少ID和name字段: {tool_def}")
                    continue

                # 如果只有 name 没有 id，记录 DEBUG 级别日志（不是错误，只是兼容性处理）
                if not tool_def.get('id') and tool_def.get('name'):
                    logger.debug(f"工具定义缺少ID，使用name作为ID: {tool_id}")

                logger.debug(f"注册工具: id={tool_id}, name={tool_def.get('name')}")

                # 创建会话工具实例
                session_tool = SessionTool(tool_def, session_id, ws_handler)

                # 轻量日志：工具参数概览（用于排查“schema 缺参导致后端报错”）
                try:
                    params = getattr(session_tool, "parameters", {}) or {}
                    props = params.get("properties") if isinstance(params, dict) else {}
                    required = params.get("required") if isinstance(params, dict) else []
                    input_ids = []
                    required_inputs = []
                    if isinstance(session_tool.inputs, list):
                        for item in session_tool.inputs:
                            if not isinstance(item, dict):
                                continue
                            pid = item.get("id")
                            if pid:
                                input_ids.append(pid)
                                if item.get("required") is True:
                                    required_inputs.append(pid)
                    logger.info(
                        "🧰 [register_session_tools] tool=%s | props=%s | required=%s | inputs=%s | inputs_required=%s",
                        tool_id,
                        list(props.keys())[:30] if isinstance(props, dict) else [],
                        required,
                        input_ids[:30],
                        required_inputs[:30],
                    )
                except Exception:
                    pass
                
                # 注册到会话工具列表
                self.session_tools[session_id][tool_id] = session_tool
                
                # logger.info(f"注册会话工具: {tool_id} (会话: {session_id})")
                
            except Exception as e:
                logger.error(f"注册会话工具失败: {e}, 工具定义: {tool_def}")
    
    def get_session_tool(self, session_id: str, tool_id: str) -> Optional[SessionTool]:
        """获取会话工具
        
        Args:
            session_id: 会话ID
            tool_id: 工具ID
            
        Returns:
            会话工具实例，如果不存在则返回None
        """
        if session_id in self.session_tools:
            return self.session_tools[session_id].get(tool_id)
        return None
    
    def get_session_tools(self, session_id: str) -> Dict[str, SessionTool]:
        """获取会话的所有工具

        Args:
            session_id: 会话ID

        Returns:
            工具字典 {tool_id: SessionTool}
        """
        # 调试日志：显示所有已注册的会话
        logger.info(f"🔍 [get_session_tools] 请求 session_id: {session_id}")
        logger.info(f"🔍 [get_session_tools] 当前所有已注册的会话: {list(self.session_tools.keys())}")

        if session_id in self.session_tools:
            tools = self.session_tools[session_id]
            logger.info(f"✅ [get_session_tools] 找到 {len(tools)} 个工具")
            return tools
        else:
            logger.warning(f"❌ [get_session_tools] 会话 {session_id} 未找到,可能未注册工具")
            return {}
    
    def list_session_tools(self, session_id: str) -> List[Dict[str, Any]]:
        """列出会话的所有工具定义
        
        Args:
            session_id: 会话ID
            
        Returns:
            工具定义列表
        """
        tools = []
        if session_id in self.session_tools:
            for tool in self.session_tools[session_id].values():
                tools.append(tool.get_definition())
        return tools
    
    def clear_session_tools(self, session_id: str):
        """清除会话的工具
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.session_tools:
            del self.session_tools[session_id]
            logger.info(f"清除会话工具: {session_id}")
    
    def update_session_tools(self, session_id: str, tools: List[Dict[str, Any]], ws_handler=None):
        """更新会话工具（先清除再注册）
        
        Args:
            session_id: 会话ID
            tools: 新的工具定义列表
            ws_handler: WebSocket处理器
        """
        self.clear_session_tools(session_id)
        self.register_session_tools(session_id, tools, ws_handler)
    
    def has_tool(self, session_id: str, tool_id: str) -> bool:
        """检查会话是否有指定工具
        
        Args:
            session_id: 会话ID
            tool_id: 工具ID
            
        Returns:
            是否存在该工具
        """
        return session_id in self.session_tools and tool_id in self.session_tools[session_id]


# 全局会话工具管理器实例
session_tool_manager = SessionToolManager()



if __name__ == "__main__":
    # 简单测试
    import asyncio

    async def test():
        # 模拟WebSocket处理器
        class MockWSHandler:
            def call_client_tool(self, session_id, tool_name, params):
                logger.info(f"Mock 调用客户端工具: {tool_name} (session: {session_id}) with params: {params}")
                return {"status": "success", "result": f"Executed {tool_name}"}

        ws_handler = MockWSHandler()
        
        # 注册工具
        tool_defs = [
            {
                "id": "echo_tool",
                "name": "Echo Tool",
                "description": "返回输入参数",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "消息内容"
                        }
                    },
                    "required": ["message"]
                }
            }
        ]
        
        session_id = "test_session"
        session_tool_manager.register_session_tools(session_id, tool_defs, ws_handler)
        
        # 获取工具并调用
        tools = session_tool_manager.get_session_tools(session_id)
        logger.info("tools=%s", tools)
        echo_tool = tools.get("echo_tool")
        
        if echo_tool:
            result = echo_tool.call({"message": "Hello, World!"})
            logger.info(f"工具调用结果: {result}")
        else:
            logger.error("未找到 echo_tool")

    asyncio.run(test())
