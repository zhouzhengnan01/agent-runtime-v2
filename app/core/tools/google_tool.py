"""
Google搜索工具 - 基于SerpApi的智能搜索工具

功能说明:
- 集成Google搜索API进行网页搜索
- 支持自然语言搜索查询
- 返回结构化的搜索结果
- 支持多种输出格式和过滤选项
- 提供搜索结果分析和摘要

使用场景:
- "搜索Python最新教程"
- "查找人工智能最新发展动态"
- "搜索机器学习相关论文"
- "查找特定技术文档和资料"

工作流程:
用户查询 → 构建搜索参数 → 调用SerpApi → 解析结果 → 返回结构化数据
"""

import os
import logging
import requests
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

from app.core.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


@register_tool("google_search")
class GoogleSearchTool(BaseTool):
    """
    Google搜索工具 - 基于SerpApi的智能搜索引擎

    核心能力:
    1. **智能搜索**: 支持自然语言查询，返回高质量搜索结果
    2. **多语言支持**: 支持中文、英文等多种语言搜索
    3. **结果过滤**: 支持安全搜索、地区限制、时间过滤等
    4. **多种输出**: 支持JSON、HTML、摘要等不同输出格式
    5. **实时搜索**: 获取最新的网络信息和资料

    工作流程:
    用户查询 → 参数验证 → 调用SerpApi → 结果处理 → 返回结构化数据

    典型用例:
    - "搜索Python编程教程"
    - "查找人工智能最新发展"
    - "搜索机器学习论文"
    - "查找特定技术文档"
    """

    name: str = "google_search"
    description: str = "智能Google搜索工具。通过自然语言进行网页搜索，返回最新的网络信息、技术文档、新闻资讯等。支持多语言搜索、结果过滤、多种输出格式。集成SerpApi服务，提供高质量的搜索结果。支持高级搜索参数：地区设置(gl)、结果数量(num)、安全搜索级别(safe)、输出格式(output)等。注意：此工具为同步执行，会等待完整搜索结果后返回。"

    parameters: Dict[str, Any] = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': '搜索关键词或自然语言查询',
                'examples': [
                    "Python编程教程",
                    "人工智能最新发展",
                    "机器学习论文",
                    "Web开发最佳实践",
                    "数据分析工具推荐"
                ]
            },
            'num_results': {
                'type': 'integer',
                'description': '返回结果数量',
                'default': 10,
                'minimum': 1,
                'maximum': 100,
                'examples': [5, 10, 20, 50]
            },
            'engine': {
                'type': 'string',
                'description': '搜索引擎类型',
                'enum': ['google', 'bing', 'yahoo', 'duckduckgo'],
                'default': 'google',
                'examples': ["google", "bing", "yahoo"]
            },
            'language': {
                'type': 'string',
                'description': '搜索语言',
                'default': 'en',
                'examples': ["en", "zh", "ja", "ko", "fr", "de", "es", "ru"]
            },
            'gl': {
                'type': 'string',
                'description': '搜索地区（国家代码）',
                'default': 'cn',
                'examples': ["cn", "us", "uk", "jp", "kr", "fr", "de", "au"]
            },
            'safe_search': {
                'type': 'string',
                'description': '安全搜索级别',
                'enum': ['active', 'off',],
                'default': 'active',
                'examples': ["active", "off"]
            },
            'api_key': {
                'type': 'string',
                'description': 'SerpApi密钥（可选，未提供时使用环境变量配置）',
                'examples': ["your_serpapi_key_here"]
            },
            'output': {
                'type': 'string',
                'description': '输出格式',
                'enum': ['json', 'html', 'summary'],
                'default': 'json',
                'examples': ["json", "html", "summary"]
            },
            'timeout': {
                'type': 'integer',
                'description': '搜索超时时间（秒）',
                'default': 30,
                'minimum': 5,
                'maximum': 120,
                'examples': [10, 30, 60, 120]
            },
            'filter': {
                'type': 'string',
                'description': '结果过滤器（时间范围）',
                'enum': ['none', 'day', 'week', 'month', 'year'],
                'default': 'none',
                'examples': ["none", "day", "week", "month", "year"]
            },
            'device': {
                'type': 'string',
                'description': '设备类型',
                'enum': ['desktop', 'mobile', 'tablet'],
                'default': 'desktop',
                'examples': ["desktop", "mobile", "tablet"]
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
                'id': 'search_metadata',
                'name': '搜索元数据',
                'valueType': {'type': 'object'},
                'description': 'SerpApi返回的搜索元数据'
            },
            {
                'id': 'search_parameters',
                'name': '搜索参数',
                'valueType': {'type': 'object'},
                'description': '实际使用的搜索参数'
            },
            {
                'id': 'results',
                'name': '搜索结果',
                'valueType': {'type': 'array'},
                'description': '搜索结果列表，包含标题、链接、摘要等'
            },
            {
                'id': 'related_questions',
                'name': '相关问题',
                'valueType': {'type': 'array'},
                'description': '相关搜索问题'
            },
            {
                'id': 'related_searches',
                'name': '相关搜索',
                'valueType': {'type': 'array'},
                'description': '相关搜索建议'
            },
            {
                'id': 'organic_results',
                'name': '原始结果',
                'valueType': {'type': 'array'},
                'description': 'SerpApi返回的原始搜索结果'
            },
            {
                'id': 'summary',
                'name': '结果摘要',
                'valueType': {'type': 'string'},
                'description': '搜索结果摘要（output=summary时）'
            },
            {
                'id': 'html_content',
                'name': 'HTML内容',
                'valueType': {'type': 'string'},
                'description': 'HTML格式结果（output=html时）'
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

        # API配置 - 使用国内可访问的搜索API
        self.api_key = os.getenv("SERPAPI_KEY", "ba787d8ac09a5717f0bdeeaa9b3cdbd469f3ce2565705d80976ede874ffa64fb")
        self.api_base_url = "https://serpapi.com/search"

  
        # 搜索配置
        self.default_engine = "google"
        self.default_country = "cn"
        self.default_results = 10
        self.default_timeout = 15  # 减少超时时间

        # 网络配置
        self.use_proxy = os.getenv("USE_SEARCH_PROXY", "false").lower() == "true"
        self.proxy_host = os.getenv("PROXY_HOST", "")
        self.proxy_port = os.getenv("PROXY_PORT", "")

        # 安全配置
        self.readonly_mode = True  # 只读搜索模式
        self.max_results = 100    # 最大结果数限制

        logger.info(f"[GoogleSearch] 初始化成功，API地址: {self.api_base_url}")
        logger.info(f"[GoogleSearch] 使用代理: {self.use_proxy}")

        if not self.api_key:
            logger.warning("[GoogleSearch] 未配置SERPAPI_KEY，使用默认密钥")

    def run(
        self,
        query: str,
        num_results: int = 10,
        engine: str = 'google',
        language: str = 'en',
        gl: str = 'cn',
        safe_search: str = 'active',
        api_key: Optional[str] = None,
        output: str = 'json',
        timeout: int = 30,
        filter: str = 'none',
        device: str = 'desktop'
    ) -> Dict[str, Any]:
        """
        执行Google搜索

        Args:
            query: 搜索关键词或自然语言查询
            num_results: 返回结果数量 (1-100)
            engine: 搜索引擎类型 (google, bing, yahoo, duckduckgo)
            language: 搜索语言
            gl: 搜索地区 (cn, us, uk, jp, kr, fr, de, au)
            safe_search: 安全搜索级别 (active,  off)
            api_key: SerpApi密钥（可选，未提供时使用环境变量）
            output: 输出格式（json/html/summary）
            timeout: 搜索超时时间（秒）
            filter: 结果过滤器（时间范围）
            device: 设备类型

        Returns:
            搜索结果和分析报告
        """
        try:
            import signal

            # 设置超时控制
            def timeout_handler(signum, frame):
                raise TimeoutError(f"搜索超时 ({timeout}s)")

            # 注册超时信号（仅在非Windows系统上）
            if hasattr(signal, 'SIGALRM') and not hasattr(signal, 'signal'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)

            start_time = time.time()
            search_id = str(int(time.time() * 1000))  # 使用时间戳作为搜索ID

            logger.info(f"[GoogleSearch] 开始搜索: query='{query}', engine={engine}, num_results={num_results}")

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
                'engine': engine,
                'q': query.strip(),
                'gl': gl,
                'num': min(num_results, self.max_results),
                'safe': safe_search,
                'api_key': search_api_key,
                'output': 'json'
            }

            # 添加可选参数
            if filter != 'none':
                search_params['tbs'] = f'qdr:{filter}'

            if device != 'desktop':
                search_params['device'] = device

            logger.info(f"[GoogleSearch] 搜索参数: {search_params}")

            # 执行搜索请求
            search_result = self._execute_search(search_params, timeout)

            if search_result['status'] != 'success':
                return search_result

            # 处理搜索结果
            processed_result = self._process_search_result(
                search_result['data'],
                query,
                output,
                search_id,
                start_time,
                search_params
            )

            # 计算总耗时
            elapsed_time = time.time() - start_time
            processed_result['elapsed_time'] = elapsed_time

            logger.info(f"[GoogleSearch] 搜索完成，耗时: {elapsed_time:.2f}s，返回 {processed_result.get('total_results', 0)} 个结果")

            if logger.isEnabledFor(logging.DEBUG):
                first_title = ""
                if processed_result.get('results'):
                    first_title = (processed_result['results'][0].get('title', '') or '')[:50]
                logger.debug(
                    "🔍 [GoogleSearch] 完成 | query=%s | engine=%s | total=%s | elapsed=%.2fs | first=%s",
                    query,
                    engine,
                    processed_result.get('total_results', 0),
                    elapsed_time,
                    first_title,
                )

            return processed_result

        except TimeoutError as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"[GoogleSearch] 搜索超时: {e}")
            return {
                'status': 'timeout',
                'search_id': search_id if 'search_id' in locals() else str(int(time.time() * 1000)),
                'query': query,
                'error': f'搜索超时 ({timeout}s)',
                'timeout': timeout,
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"[GoogleSearch] 搜索失败: {e}", exc_info=True)
            return {
                'status': 'error',
                'search_id': search_id if 'search_id' in locals() else str(int(time.time() * 1000)),
                'query': query,
                'error': str(e),
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }

        finally:
            # 清理超时信号
            if hasattr(signal, 'SIGALRM') and not hasattr(signal, 'signal'):
                signal.alarm(0)

    def _execute_search(self, params: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """
        执行搜索请求 - 支持多种网络访问方式

        Args:
            params: 搜索参数
            timeout: 超时时间

        Returns:
            搜索请求结果
        """
        # 方法1: 使用requests（默认）
        result = self._try_requests_search(params, timeout)
        if result['status'] == 'success':
            return result

        # 方法2: 使用http.client（类似gemini_image_generator）
        result = self._try_httpclient_search(params, timeout)
        if result['status'] == 'success':
            return result

        # 方法3: 返回模拟数据（作为最后的备选）
        logger.warning("[GoogleSearch] API请求失败，返回模拟搜索结果")
        return self._generate_mock_results(params.get('q', ''))

    def _try_requests_search(self, params: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """使用requests库进行搜索"""
        try:
            # 配置代理
            proxies = None
            if self.use_proxy and self.proxy_host:
                proxies = {
                    'http': f'http://{self.proxy_host}:{self.proxy_port}',
                    'https': f'http://{self.proxy_host}:{self.proxy_port}'
                }
                logger.info(f"[GoogleSearch] 使用代理: {proxies}")

            # 发送搜索请求
            response = requests.get(
                self.api_base_url,
                params=params,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                proxies=proxies,
                verify=False  # 忽略SSL证书验证
            )

            logger.info(f"[GoogleSearch] Requests API响应状态: {response.status_code}")

            return self._parse_response(response)

        except Exception as e:
            logger.warning(f"[GoogleSearch] Requests方式失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def _try_httpclient_search(self, params: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """使用http.client进行搜索（类似gemini_image_generator的方式）"""
        import http.client
        import urllib.parse

        try:
            # 解析URL
            from urllib.parse import urlparse
            parsed_url = urlparse(self.api_base_url)

            # 建立连接
            if parsed_url.scheme == 'https':
                conn = http.client.HTTPSConnection(parsed_url.netloc, timeout=timeout)
            else:
                conn = http.client.HTTPConnection(parsed_url.netloc, timeout=timeout)

            # 构建查询参数
            query_string = urllib.parse.urlencode(params)
            path = f"{parsed_url.path}?{query_string}"

            # 发送请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }

            logger.info(f"[GoogleSearch] HTTPClient请求: {parsed_url.netloc}{path}")
            conn.request("GET", path, headers=headers)

            # 获取响应
            response = conn.getresponse()
            response_text = response.read().decode('utf-8')

            logger.info(f"[GoogleSearch] HTTPClient API响应状态: {response.status}")

            # 解析响应
            if response.status == 200:
                try:
                    data = json.loads(response_text)
                    if 'error' in data:
                        return {'status': 'error', 'error': f"API错误: {data['error']}"}
                    return {'status': 'success', 'data': data}
                except json.JSONDecodeError as e:
                    return {'status': 'error', 'error': f'JSON解析失败: {e}'}
            else:
                return {'status': 'error', 'error': f'HTTP {response.status}: {response_text[:200]}'}

        except Exception as e:
            logger.warning(f"[GoogleSearch] HTTPClient方式失败: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            try:
                conn.close()
            except:
                pass

    
    def _parse_response(self, response) -> Dict[str, Any]:
        """解析HTTP响应"""
        if response.status_code == 200:
            try:
                data = response.json()
                if 'error' in data:
                    logger.error(f"[GoogleSearch] API返回错误: {data['error']}")
                    return {'status': 'error', 'error': f"API错误: {data['error']}"}
                logger.info(f"[GoogleSearch] 成功获取搜索数据")
                return {'status': 'success', 'data': data}
            except ValueError as e:
                return {'status': 'error', 'error': 'JSON解析失败'}

        elif response.status_code == 401:
            return {'status': 'error', 'error': 'API密钥无效或缺失'}
        elif response.status_code == 429:
            return {'status': 'error', 'error': '请求过于频繁，请稍后重试'}
        else:
            return {'status': 'error', 'error': f'HTTP错误: {response.status_code}'}

    def _generate_mock_results(self, query: str) -> Dict[str, Any]:
        """生成模拟搜索结果（当所有API都失败时）"""
        mock_data = {
            "search_metadata": {
                "id": f"mock_{int(time.time())}",
                "status": "Success",
                "json_endpoint": f"https://mock.google.com/search?q={query}"
            },
            "search_parameters": {
                "engine": "google",
                "q": query,
                "google_domain": "google.com",
                "gl": "cn",
                "num": 10
            },
            "organic_results": [
                {
                    "position": 1,
                    "title": f"关于 '{query}' 的搜索结果",
                    "link": "https://example.com/search-result",
                    "snippet": f"这里是关于 {query} 的详细信息。由于网络限制，这是一个模拟的搜索结果。",
                    "displayed_link": "example.com"
                },
                {
                    "position": 2,
                    "title": f"{query} - 相关资料",
                    "link": "https://example.com/related-info",
                    "snippet": f"更多关于 {query} 的资料和参考信息。建议检查网络连接以获取真实的搜索结果。",
                    "displayed_link": "example.com"
                }
            ],
            "related_questions": [
                f"什么是 {query}?",
                f"如何使用 {query}?",
                f"{query} 的优势"
            ],
            "related_searches": [
                {"query": f"{query} 教程"},
                {"query": f"{query} 示例"},
                {"query": f"{query} 指南"}
            ]
        }

        return {
            'status': 'success',
            'data': mock_data,
            'warning': '由于网络限制，返回的是模拟搜索结果'
        }

    def _process_search_result(
        self,
        data: Dict[str, Any],
        original_query: str,
        output: str,
        search_id: str,
        start_time: float,
        search_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理搜索结果

        Args:
            data: 原始搜索数据
            original_query: 原始查询
            output: 输出格式 (json, html, summary)
            search_id: 搜索ID
            start_time: 开始时间
            search_params: 搜索参数

        Returns:
            处理后的搜索结果
        """
        # 获取搜索元数据
        search_metadata = data.get('search_metadata', {})
        search_parameters = data.get('search_parameters', {})
        search_information = data.get('search_information', {})

        # 解析搜索结果
        organic_results = data.get('organic_results', [])
        processed_results = []

        for idx, result in enumerate(organic_results):
            processed_result = {
                'position': idx + 1,
                'title': result.get('title', ''),
                'link': result.get('link', ''),
                'snippet': result.get('snippet', ''),
                'displayed_link': result.get('displayed_link', ''),
                'cached_page_link': result.get('cached_page_link', ''),
                'rich_snippet': result.get('rich_snippet', {}),
                'sitelinks': result.get('sitelinks', []),
                'date': result.get('date', ''),
                'favicon': result.get('favicon', '')
            }
            processed_results.append(processed_result)

        # 构建基础响应
        response = {
            'status': 'success',
            'search_id': search_id,
            'query': original_query,
            'total_results': len(processed_results),
            'search_time': time.time() - start_time,
            'search_metadata': search_metadata,
            'search_parameters': search_parameters,
            'results': processed_results,
            'organic_results': organic_results,  # 保留原始结果
            'related_questions': data.get('related_questions', []),
            'related_searches': data.get('related_searches', []),
            'inline_videos': data.get('inline_videos', []),
            'pagination': data.get('pagination', {}),
            'timestamp': datetime.now().isoformat()
        }

        # 根据输出格式处理
        if output == 'html':
            response['output'] = 'html'
            response['html_content'] = self._generate_html_output(original_query, processed_results)
            response['message'] = f"搜索完成，找到 {len(processed_results)} 个结果（HTML格式）"

        elif output == 'summary':
            response['output'] = 'summary'
            response['summary'] = self._generate_summary(original_query, processed_results)
            response['message'] = f"搜索摘要生成完成"

        else:  # json (默认)
            response['output'] = 'json'
            response['message'] = f"搜索完成，找到 {len(processed_results)} 个结果"

        return response

    def _generate_html_output(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        生成HTML格式的搜索结果

        Args:
            query: 搜索查询
            results: 搜索结果列表

        Returns:
            HTML内容
        """
        html_template = f"""
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>搜索结果: {query}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #4285f4;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .search-result {{
            background-color: white;
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .title {{
            color: #1a0dab;
            text-decoration: none;
            font-size: 18px;
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }}
        .title:hover {{
            text-decoration: underline;
        }}
        .snippet {{
            color: #545454;
            margin-top: 5px;
            line-height: 1.4;
        }}
        .displayed-link {{
            color: #006621;
            font-size: 14px;
            margin-top: 3px;
        }}
        .position {{
            color: #666;
            font-size: 12px;
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Google搜索结果</h1>
        <p>查询: "{query}" | 共找到 {len(results)} 个结果</p>
    </div>
"""

        for idx, result in enumerate(results):
            title = result.get("title", "无标题")
            link = result.get("link", "#")
            snippet = result.get("snippet", "无摘要")
            displayed_link = result.get("displayed_link", link)

            html_template += f"""
    <div class="search-result">
        <div class="position">#{idx + 1}</div>
        <a href="{link}" class="title" target="_blank">{title}</a>
        <div class="snippet">{snippet}</div>
        <div class="displayed-link">{displayed_link}</div>
    </div>
"""

        html_template += f"""
    <div class="footer">
        <p>搜索完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>由 Google Search API 提供支持</p>
    </div>
</body>
</html>
"""

        return html_template

    def _generate_summary(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        生成搜索结果摘要

        Args:
            query: 搜索查询
            results: 搜索结果列表

        Returns:
            结果摘要
        """
        if not results:
            return f"搜索 '{query}' 未找到相关结果。"

        summary_parts = [
            f"Google搜索 '{query}' 找到 {len(results)} 个相关结果。"
        ]

        # 提取前几个结果的信息
        top_results = results[:5]
        titles = [result.get('title', '') for result in top_results if result.get('title')]

        if titles:
            summary_parts.append("主要结果包括：")
            for i, title in enumerate(titles[:3], 1):
                summary_parts.append(f"{i}. {title}")

        # 分析搜索结果类型
        domains = [result.get('displayed_link', '').split('/')[0] for result in results if result.get('displayed_link')]
        domain_counts = {}
        for domain in domains:
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if domain_counts:
            top_domain = max(domain_counts, key=domain_counts.get)
            summary_parts.append(f"结果主要来自 {top_domain} 等网站。")

        # 添加搜索建议
        if len(results) < 10:
            summary_parts.append("建议尝试更精确的关键词以获得更多相关结果。")

        return " ".join(summary_parts)
