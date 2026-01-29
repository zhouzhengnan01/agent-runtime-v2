"""
百度搜索工具 - 基于千帆AppBuilder的智能搜索工具

功能说明:
- 集成百度搜索API进行网页搜索
- 支持自然语言搜索查询
- 返回结构化的搜索结果
- 支持多种输出格式和过滤选项
- 提供搜索结果分析和摘要

使用场景:
- "搜索北京旅游景区"
- "查找Python最新教程"
- "搜索人工智能最新发展动态"
- "查找特定技术文档和资料"

工作流程:
用户查询 -> 构建搜索参数 -> 调用千帆API -> 解析结果 -> 返回结构化数据
"""

import logging
import requests
import json
import time
import warnings
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.tools.base import BaseTool, register_tool
from app.config import settings

# 禁用SSL证书验证警告
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

logger = logging.getLogger(__name__)


@register_tool("baidu_search")
class BaiduSearchTool(BaseTool):
    """
    百度搜索工具 - 基于千帆AppBuilder的智能搜索引擎

    核心能力:
    1. **智能搜索**: 支持自然语言查询，返回高质量搜索结果
    2. **中文优化**: 针对中文搜索优化，提供更准确的结果
    3. **结果过滤**: 支持网站过滤、时间过滤、资源类型过滤等
    4. **多种输出**: 支持JSON、HTML、摘要等不同输出格式
    5. **实时搜索**: 获取最新的网络信息和资料

    工作流程:
    用户查询 -> 参数验证 -> 调用千帆API -> 结果处理 -> 返回结构化数据

    典型用例:
    - "搜索北京旅游景区"
    - "查找Python编程教程"
    - "搜索人工智能最新发展"
    - "查找特定技术文档"
    """

    name: str = "baidu_search"
    description: str = "智能百度搜索工具。通过自然语言进行网页搜索，返回最新的网络信息、技术文档、新闻资讯等。针对中文搜索优化，支持结果过滤、多种输出格式。集成千帆AppBuilder服务，提供高质量的中文搜索结果。支持高级搜索参数：网站过滤(site)、时间过滤(search_recency_filter)、资源类型过滤(resource_type_filter)、结果数量(top_k)等。注意：此工具为同步执行，会等待完整搜索结果后返回。"

    parameters: Dict[str, Any] = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': '搜索关键词或自然语言查询',
                'examples': [
                    "北京旅游景区",
                    "Python编程教程",
                    "人工智能最新发展",
                    "机器学习论文",
                    "数据分析工具推荐"
                ]
            },
            'top_k': {
                'type': 'integer',
                'description': '返回结果数量',
                'default': 10,
                'minimum': 1,
                'maximum': 50,
                'examples': [5, 10, 20, 50]
            },
            'search_source': {
                'type': 'string',
                'description': '搜索源类型',
                'enum': ['baidu_search_v2'],
                'default': 'baidu_search_v2',
                'examples': ["baidu_search_v2"]
            },
            'site_filter': {
                'type': 'array',
                'description': '网站过滤（限制搜索结果的网站域名）',
                'items': {
                    'type': 'string'
                },
                'examples': [
                    ["www.weather.com.cn"],
                    ["zhihu.com", "csdn.net"],
                    ["baidu.com", "google.com"]
                ]
            },
            'resource_type_filter': {
                'type': 'array',
                'description': '资源类型过滤',
                'items': {
                    'type': 'object',
                    'properties': {
                        'type': {
                            'type': 'string',
                            'enum': ['web', 'news', 'image', 'video', 'doc'],
                            'description': '资源类型'
                        },
                        'top_k': {
                            'type': 'integer',
                            'description': '该类型返回数量'
                        }
                    }
                },
                'examples': [
                    [{"type": "web", "top_k": 20}],
                    [{"type": "news", "top_k": 10}, {"type": "web", "top_k": 10}]
                ]
            },
            'search_recency_filter': {
                'type': 'string',
                'description': '时间过滤器（搜索结果的时间范围）',
                'enum': ['day', 'week', 'month', 'year', 'none'],
                'default': 'none',
                'examples': ["day", "week", "month", "year", "none"]
            },
            'api_key': {
                'type': 'string',
                'description': '千帆AppBuilder API密钥（可选，未提供时使用环境变量配置）',
                'examples': ["bce-v3/ALTAK-xxxxxxxxxxx"]
            },
            'output': {
                'type': 'string',
                'description': '输出格式',
                'enum': ['json'],
                'default': 'json',
                'examples': ["json"]
            },
            'timeout': {
                'type': 'integer',
                'description': '搜索超时时间（秒）',
                'default': 30,
                'minimum': 5,
                'maximum': 120,
                'examples': [10, 30, 60, 120]
            }
        },
        'required': ['query']
    }

    output: Dict[str, Any] = {
        'type': 'object',
        'properties': [
            {
                'id': 'status',
                'name': '搜索状态',
                'valueType': {'type': 'string'},
                'description': '执行状态（success/error/timeout）'
            },
            {
                'id': 'search_id',
                'name': '搜索ID',
                'valueType': {'type': 'string'},
                'description': '唯一搜索标识符'
            },
            {
                'id': 'query',
                'name': '搜索查询',
                'valueType': {'type': 'string'},
                'description': '原始搜索查询'
            },
            {
                'id': 'total_results',
                'name': '结果总数',
                'valueType': {'type': 'integer'},
                'description': '返回的搜索结果数量'
            },
            {
                'id': 'search_time',
                'name': '搜索耗时',
                'valueType': {'type': 'float'},
                'description': '搜索执行时间（秒）'
            },
            {
                'id': 'results',
                'name': '搜索结果',
                'valueType': {'type': 'array'},
                'description': '搜索结果列表，包含标题、链接、摘要等'
            },
            {
                'id': 'timestamp',
                'name': '搜索时间',
                'valueType': {'type': 'string'},
                'description': '搜索执行时间戳'
            },
            {
                'id': 'error',
                'name': '错误信息',
                'valueType': {'type': 'string'},
                'description': '错误信息（如果失败）'
            }
        ]
    }

    require_confirmation: bool = False  # 不需要确认

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # API配置 - 使用千帆AppBuilder API
        # 优先使用BAIDU_API_KEY，如果没有则使用SERPAPI_KEY作为备用
        self.api_key = settings.BAIDU_API_KEY or settings.SERPAPI_KEY or "bce-v3/ALTAK-bCulybWE4TDwzU7MicwiR/bc75c0b5ff9ce42e99e43a4fc09a3f2aa0fac7a5"

        # 修复API URL（确保正确的URL）
        self.api_base_url = settings.BAIDU_API_BASE_URL
        if self.api_base_url.endswith('web_searc'):
            self.api_base_url = self.api_base_url + 'h'

        # 搜索配置
        self.default_search_source = settings.BAIDU_SEARCH_ENGINE or "baidu_search_v2"
        self.default_results = settings.BAIDU_SEARCH_NUM_RESULTS or 10
        self.default_timeout = settings.BAIDU_DEFAULT_TIMEOUT or 30

        # 网络配置
        self.use_proxy = settings.USE_SEARCH_PROXY or False
        self.proxy_host = settings.PROXY_HOST or ""
        self.proxy_port = settings.PROXY_PORT or ""

        # 安全配置
        self.readonly_mode = True  # 只读搜索模式
        self.max_results = settings.BAIDU_MAX_RESULTS or 50    # 最大结果数限制

        logger.info(f"[BaiduSearch] 初始化成功，API地址: {self.api_base_url}")
        logger.info(f"[BaiduSearch] 使用代理: {self.use_proxy}")

        if not self.api_key:
            logger.warning("[BaiduSearch] 未配置API密钥，使用默认密钥")

    def run(
        self,
        query: str,
        top_k: int = 10,
        search_source: str = 'baidu_search_v2',
        site_filter: Optional[List[str]] = None,
        resource_type_filter: Optional[List[Dict[str, Any]]] = None,
        search_recency_filter: str = 'none',
        api_key: Optional[str] = None,
        output: str = 'json',
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        执行百度搜索

        Args:
            query: 搜索关键词或自然语言查询
            top_k: 返回结果数量 (1-50)
            search_source: 搜索源类型
            site_filter: 网站过滤列表
            resource_type_filter: 资源类型过滤
            search_recency_filter: 时间过滤器
            api_key: 千帆API密钥（可选，未提供时使用环境变量）
            output: 输出格式（json）
            timeout: 搜索超时时间（秒）

        Returns:
            搜索结果的JSON数据
        """
        try:
            start_time = time.time()
            search_id = str(int(time.time() * 1000))  # 使用时间戳作为搜索ID

            # 参数验证
            if not query or not query.strip():
                return {
                    'status': 'error',
                    'search_id': search_id,
                    'query': query,
                    'error': '搜索查询不能为空',
                    'timestamp': datetime.now().isoformat()
                }

            # 使用提供的API密钥或默认密钥
            search_api_key = api_key or self.api_key

            # 构建搜索参数
            search_params = {
                'messages': [
                    {
                        'content': query.strip(),
                        'role': 'user'
                    }
                ],
                'search_source': search_source
            }

            # 添加资源类型过滤
            if resource_type_filter:
                search_params['resource_type_filter'] = resource_type_filter
            else:
                # 默认网页搜索
                search_params['resource_type_filter'] = [{'type': 'web', 'top_k': min(top_k, self.max_results)}]

            # 添加网站过滤
            if site_filter:
                search_params['search_filter'] = {
                    'match': {
                        'site': site_filter
                    }
                }

            # 添加时间过滤
            if search_recency_filter != 'none':
                search_params['search_recency_filter'] = search_recency_filter

            # 执行搜索请求
            search_result = self._execute_search(search_params, search_api_key, timeout)

            if search_result['status'] != 'success':
                return search_result

            # 处理搜索结果
            processed_result = self._process_search_result(
                search_result['data'],
                query,
                search_id,
                start_time
            )

            return processed_result

        except Exception as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"[BaiduSearch] 搜索失败: {e}", exc_info=True)
            return {
                'status': 'error',
                'search_id': search_id if 'search_id' in locals() else str(int(time.time() * 1000)),
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _execute_search(self, params: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, Any]:
        """
        执行搜索请求

        Args:
            params: 搜索参数
            api_key: API密钥
            timeout: 超时时间

        Returns:
            搜索请求结果
        """
        try:
            # 配置代理
            proxies = None
            if self.use_proxy and self.proxy_host:
                proxies = {
                    'http': f'http://{self.proxy_host}:{self.proxy_port}',
                    'https': f'http://{self.proxy_host}:{self.proxy_port}'
                }
                logger.info(f"[BaiduSearch] 使用代理: {proxies}")

            # 发送搜索请求
            response = requests.post(
                self.api_base_url,
                json=params,
                timeout=timeout,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                proxies=proxies,
                verify=False  # 忽略SSL证书验证
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"[BaiduSearch] 成功获取搜索数据")
                    return {'status': 'success', 'data': data}
                except ValueError as e:
                    return {'status': 'error', 'error': 'JSON解析失败'}
            else:
                return {'status': 'error', 'error': f'HTTP错误: {response.status_code}'}

        except Exception as e:
            logger.warning(f"[BaiduSearch] 搜索请求失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def _process_search_result(
        self,
        data: Dict[str, Any],
        original_query: str,
        search_id: str,
        start_time: float
    ) -> Dict[str, Any]:
        """
        处理搜索结果，只返回原始搜索数据

        Args:
            data: 原始搜索数据
            original_query: 原始查询
            search_id: 搜索ID
            start_time: 开始时间

        Returns:
            搜索结果的JSON数据
        """
        # 千帆API返回的数据结构
        results = []

        # 打印完整的响应数据用于调试
        logger.info(f"[BaiduSearch] API完整响应: {json.dumps(data, ensure_ascii=False)[:2000]}...")
        logger.info(f"[BaiduSearch] 顶级键: {list(data.keys())}")

        # 提取 references 字段中的搜索结果
        if 'references' in data:
            refs = data['references']
            logger.info(f"[BaiduSearch] references字段类型: {type(refs)}")
            if isinstance(refs, list):
                results = refs
                logger.info(f"[BaiduSearch] 从references获取到 {len(results)} 个结果")
            else:
                logger.warning(f"[BaiduSearch] references 不是列表类型: {type(refs)}")

        # 如果没有结果，检查其他字段
        if not results and 'result' in data:
            result_data = data['result']
            logger.info(f"[BaiduSearch] result字段类型: {type(result_data)}")
            if isinstance(result_data, dict):
                if 'web_results' in result_data:
                    results = result_data['web_results']
                    logger.info(f"[BaiduSearch] 从result.web_results获取到 {len(results)} 个结果")
                elif 'results' in result_data:
                    results = result_data['results']
                    logger.info(f"[BaiduSearch] 从result.results获取到 {len(results)} 个结果")
                elif 'items' in result_data:
                    results = result_data['items']
                    logger.info(f"[BaiduSearch] 从result.items获取到 {len(results)} 个结果")
            elif isinstance(result_data, list):
                results = result_data
                logger.info(f"[BaiduSearch] result是列表，包含 {len(results)} 个结果")

        # 检查其他可能的字段
        if not results:
            for key, value in data.items():
                if key not in ['status', 'search_id', 'query', 'request_id', 'timestamp'] and isinstance(value, list):
                    results = value
                    logger.info(f"[BaiduSearch] 从字段 '{key}' 获取到 {len(results)} 个结果")
                    break

        logger.info(f"[BaiduSearch] 最终结果数量: {len(results)}")

        # 构建响应
        response = {
            'status': 'success',
            'search_id': search_id,
            'query': original_query,
            'total_results': len(results),
            'search_time': time.time() - start_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        return response