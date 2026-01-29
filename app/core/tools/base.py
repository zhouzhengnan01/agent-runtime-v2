"""
工具基类 - 独立实现，不依赖 qwen-agent

提供工具系统的基础架构，所有工具都继承自此基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional, List, AsyncIterator
import logging
import json

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    工具基类 - 独立实现

    不再依赖 qwen-agent，提供完整的工具接口
    """

    # 工具元信息
    name: str = None
    description: str = None
    parameters: Dict[str, Any] = None  # OpenAPI格式参数（向后兼容）

    # 新的参数定义格式
    inputs: List[Dict[str, Any]] = None  # 输入参数定义
    output: Dict[str, Any] = None  # 输出定义
    async_mode: bool = False  # 是否异步执行

    # 交互式确认配置
    require_confirmation: Union[bool, str] = False  # False=不确认, True=总是确认, "auto"=智能确认

    # 工具路由配置
    estimated_time: float = 1.0  # 预估执行时间（秒），用于判断是否后台执行
    support_streaming: bool = False  # 是否支持流式输出

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化工具"""
        self.config = config or {}
        self.setup()

    def setup(self):
        """工具初始化设置，子类可重写"""
        pass

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        执行工具的核心逻辑（同步版本）

        Args:
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        pass

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        调用工具的统一接口

        Args:
            params: 参数字典或JSON字符串
            **kwargs: 额外参数

        Returns:
            工具执行结果的字符串表示
        """
        try:
            # 处理参数
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {'input': params}

            # 调用run方法
            result = self.run(**params)

            # 格式化返回结果
            if isinstance(result, dict):
                if result.get('status') == 'error':
                    error_msg = result.get('error') or result.get('message') or '未知错误'
                    return f"工具执行失败: {error_msg}"
                else:
                    # 返回简洁的消息，而不是完整的 JSON
                    # LangChain 的 Agent 更容易处理简洁的文本
                    if 'message' in result:
                        return result['message']
                    else:
                        return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return str(result)

        except Exception as e:
            logger.error(f"工具 {self.name} 执行失败: {e}")
            return f"工具执行失败: {str(e)}"

    async def call_stream(self, params: Union[str, dict], **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        流式调用工具（异步生成器）

        用于长时间运行的工具，可以逐步返回部分结果。
        默认实现：回退到普通call()，一次性返回最终结果。

        Args:
            params: 参数字典或JSON字符串
            **kwargs: 额外参数

        Yields:
            Dict[str, Any]: 流式结果块，格式：
                {
                    "type": "partial_result" | "progress" | "final_result",
                    "data": Any,  # 结果数据
                    "message": str,  # 可选的说明信息
                    "progress": float  # 可选的进度 0.0-1.0
                }

        Example:
            async for chunk in tool.call_stream(params):
                if chunk["type"] == "partial_result":
                    print(f"部分结果: {chunk['data']}")
                elif chunk["type"] == "final_result":
                    print(f"最终结果: {chunk['data']}")
        """
        # 默认实现：回退到同步call()
        logger.warning(f"工具 {self.name} 未实现流式接口，回退到同步模式")

        try:
            result = self.call(params, **kwargs)

            # 返回最终结果
            yield {
                "type": "final_result",
                "data": result,
                "message": "工具执行完成"
            }

        except Exception as e:
            logger.error(f"工具 {self.name} 流式执行失败: {e}")
            yield {
                "type": "error",
                "data": None,
                "message": f"工具执行失败: {str(e)}"
            }

    def get_parameters_schema(self) -> Dict[str, Any]:
        """获取工具的参数模式"""
        if self.parameters:
            return self.parameters
        elif self.inputs:
            # 从新格式转换为OpenAPI格式
            return self._convert_inputs_to_schema()
        else:
            return {
                'type': 'object',
                'properties': {},
                'required': []
            }

    def _convert_inputs_to_schema(self) -> Dict[str, Any]:
        """将inputs格式转换为OpenAPI Schema格式"""
        properties = {}
        required = []

        if not self.inputs:
            return {'type': 'object', 'properties': {}, 'required': []}

        for input_param in self.inputs:
            param_id = input_param.get('id', input_param.get('name'))
            value_type = input_param.get('valueType', {})

            # 构建OpenAPI属性定义
            prop_def = {
                "description": input_param.get('description', input_param.get('name', ''))
            }

            # 根据类型设置OpenAPI类型
            type_name = value_type.get('type', 'string')
            if type_name in ['int', 'long']:
                prop_def["type"] = "integer"
            elif type_name in ['float', 'double']:
                prop_def["type"] = "number"
            elif type_name == 'boolean':
                prop_def["type"] = "boolean"
            elif type_name == 'array':
                prop_def["type"] = "array"
                if 'elementType' in value_type:
                    prop_def["items"] = {"type": value_type['elementType'].get('type', 'string')}
            elif type_name == 'object':
                prop_def["type"] = "object"
                if 'properties' in value_type:
                    prop_def["properties"] = value_type['properties']
            else:
                prop_def["type"] = "string"

            properties[param_id] = prop_def

            if input_param.get('required', False):
                required.append(param_id)

        return {
            'type': 'object',
            'properties': properties,
            'required': required
        }

    @property
    def function(self) -> dict:
        """返回工具的函数定义（OpenAI Function Calling格式）"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters or self.get_parameters_schema(),
        }

    def get_tool_definition(self) -> dict:
        """获取工具的完整定义（新格式）"""
        definition = {
            'id': self.name,
            'name': self.name,
            'description': self.description or '',
            'async': self.async_mode
        }

        # 添加输入参数定义
        if self.inputs:
            definition['inputs'] = self.inputs
        elif self.parameters:
            # 从OpenAPI格式转换
            definition['inputs'] = self._convert_schema_to_inputs()
        else:
            definition['inputs'] = []

        # 添加输出定义
        if self.output:
            definition['output'] = self.output
        else:
            definition['output'] = {
                'id': 'result',
                'name': '结果',
                'type': 'object'
            }

        return definition

    def _convert_schema_to_inputs(self) -> List[Dict[str, Any]]:
        """将OpenAPI Schema转换为inputs格式"""
        inputs = []

        if not self.parameters or 'properties' not in self.parameters:
            return inputs

        properties = self.parameters.get('properties', {})
        required = self.parameters.get('required', [])

        for prop_id, prop_def in properties.items():
            # 确定值类型
            prop_type = prop_def.get('type', 'string')
            value_type = {'type': 'string'}

            if prop_type == 'integer':
                value_type['type'] = 'int'
            elif prop_type == 'number':
                value_type['type'] = 'float'
            elif prop_type == 'boolean':
                value_type['type'] = 'boolean'
            elif prop_type == 'array':
                value_type['type'] = 'array'
                if 'items' in prop_def:
                    value_type['elementType'] = prop_def['items']
            elif prop_type == 'object':
                value_type['type'] = 'object'
                if 'properties' in prop_def:
                    value_type['properties'] = prop_def['properties']

            # 创建输入参数
            tool_input = {
                'id': prop_id,
                'name': prop_def.get('title', prop_id),
                'valueType': value_type,
                'required': prop_id in required,
                'description': prop_def.get('description')
            }

            if 'default' in prop_def:
                tool_input['default'] = prop_def['default']

            inputs.append(tool_input)

        return inputs


# 工具注册表
TOOL_REGISTRY = {}


def register_tool(name: str = None, allow_overwrite: bool = False):
    """
    工具注册装饰器

    Args:
        name: 工具名称
        allow_overwrite: 是否允许覆盖已存在的工具
    """
    def decorator(cls):
        tool_name = name or cls.name or cls.__name__

        if not hasattr(cls, 'name'):
            cls.name = tool_name

        if tool_name in TOOL_REGISTRY and not allow_overwrite:
            raise ValueError(f'工具 `{tool_name}` 已存在！')

        # 注册到本地注册表
        TOOL_REGISTRY[tool_name] = cls

        logger.info(f"✅ 工具已注册: {tool_name}")

        return cls
    return decorator


def get_tool(name: str, config: Dict[str, Any] = None) -> BaseTool:
    """
    获取工具实例

    Args:
        name: 工具名称
        config: 工具配置

    Returns:
        工具实例
    """
    if name in TOOL_REGISTRY:
        return TOOL_REGISTRY[name](config)
    return None


def list_tools() -> list:
    """列出所有可用工具"""
    return list(TOOL_REGISTRY.keys())