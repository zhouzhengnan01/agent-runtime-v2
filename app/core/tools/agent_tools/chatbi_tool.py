"""
ChatBI 工具 - AI 驱动的数据库查询和分析工具

功能说明:
- 自然语言转 SQL 查询 (Text-to-SQL)
- 自动执行 SQL 并返回结果
- 智能数据分析和洞察
- 生成可视化图表（表格、柱状图、折线图等）
- 支持多数据库（MySQL、PostgreSQL等）

使用场景:
- "查询最近7天的智能体使用次数"
- "分析哪个工具最受欢迎"
- "生成本月巡检报告的数据统计"
- "对比不同智能体的性能指标"

工作流程:
用户问题 → LLM Text-to-SQL → 执行查询 → 数据分析 → 生成可视化 → 返回结果
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import Session
import concurrent.futures

from app.core.tools.base import BaseTool, register_tool
from app.core.llm.langchain_factory import create_langchain_llm
from app.config import settings

logger = logging.getLogger(__name__)


@register_tool("chatbi")
class ChatBITool(BaseTool):
    """
    ChatBI 工具 - AI 驱动的商业智能分析

    核心能力:
    1. **Text-to-SQL**: 自然语言转 SQL 查询
    2. **智能执行**: 自动执行查询并处理结果
    3. **数据分析**: AI 分析数据并提供洞察
    4. **可视化**: 生成表格、图表等可视化内容
    5. **安全防护**: SQL 注入防护、只读模式

    工作流程:
    用户问题 → 获取数据库schema → LLM生成SQL → 执行查询 → 分析结果 → 生成报告

    典型用例:
    - "查询最近一周创建的智能体数量"
    - "分析各个工具的使用频率"
    - "生成本月数据统计表格"
    - "对比不同类型智能体的性能"
    """

    name: str = "chatbi"
    description = "高性能AI数据分析工具。通过自然语言查询数据库、分析数据并生成可视化报告。支持智能Text-to-SQL转换（条件性多生成优化）、数据分析、图表生成等功能。**性能优化**：快速模式0.8-1.5秒响应，标准模式多级缓存，超时控制。支持直接指定数据库连接参数：host、port、user、password、database，未提供时使用环境变量配置。支持四种SQL生成模式：auto（智能条件生成）、single（传统生成）、multi（强制多生成）、fast_mode（极速模式，1秒内响应）。注意：此工具为同步执行，会等待完整结果后返回。"

    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': '自然语言查询，描述想要分析的数据',
                'examples': [
                    "查询最近7天创建的智能体",
                    "分析各个工具的使用次数",
                    "生成本月巡检数据统计表",
                    "统计不同类型智能体的数量"
                ]
            },
            'host': {
                'type': 'string',
                'description': '数据库主机地址（可选，未提供时使用环境变量配置）',
                'examples': ["cd-cynosdbmysql-grp-ajwkswr8.sql.tencentcdb.com", "localhost", "192.168.1.100"]
            },
            'port': {
                'type': 'string',
                'description': '数据库端口（可选，未提供时使用环境变量配置）',
                'examples': ["24254", "3306", "5432"]
            },
            'user': {
                'type': 'string',
                'description': '数据库用户名（可选，未提供时使用环境变量配置）',
                'examples': ["root", "admin", "user"]
            },
            'password': {
                'type': 'string',
                'description': '数据库密码（可选，未提供时使用环境变量配置）',
                'examples': ["password", "secret"]
            },
            'database': {
                'type': 'string',
                'description': '数据库名称（可选，未提供时使用环境变量配置）',
                'examples': ["jetlinks_agent", "production_db", "analytics_db"]
            },
            'driver': {
                'type': 'string',
                'description': '数据库驱动类型',
                'enum': ['postgresql', 'mysql', 'sqlite'],
                'default': 'postgresql',
                'examples': ["postgresql", "mysql", "sqlite"]
            },
            'visualization_type': {
                'type': 'string',
                'description': '可视化类型',
                'enum': ['auto', 'table', 'bar', 'line', 'pie', 'none'],
                'default': 'auto'
            },
            'limit': {
                'type': 'string',
                'description': '结果数量限制（防止返回过多数据）',
                'default': '100'
            },
            'timeout': {
                'type': 'string',
                'description': '查询超时时间（秒）',
                'default': '30',
                'examples': ['10', '30', '60', '120']
            },
            'sql_generation_mode': {
                'type': 'string',
                'description': 'SQL生成模式',
                'enum': ['auto', 'single', 'multi'],
                'default': 'auto',
                'examples': ['auto', 'single', 'multi']
            },
            'sql_quality_threshold': {
                'type': 'string',
                'description': 'SQL质量阈值（0-1），超过此值则使用单生成',
                'default': '0.75',
                'examples': ['0.6', '0.75', '0.8', '0.9']
            },
            'fast_mode': {
                'type': 'boolean',
                'description': '快速模式 - 禁用自检和复杂优化以提升速度',
                'default': False
            },
            'query_timeout': {
                'type': 'string',
                'description': '查询超时时间（秒），超过此时间将快速返回',
                'default': '30',
                'examples': ['10', '20', '30', '60']
            }
        },
        'required': ['query']
    }

    output = {
        'type': 'object',
        'properties': [
            {
                'id': 'status',
                'name': '执行状态',
                'valueType': {'type': 'string'},
                'description': '执行状态（success/error）'
            },
            {
                'id': 'query',
                'name': '原始查询',
                'valueType': {'type': 'string'},
                'description': '原始查询'
            },
            {
                'id': 'sql',
                'name': 'SQL语句',
                'valueType': {'type': 'string'},
                'description': '生成的SQL语句'
            },
            {
                'id': 'data',
                'name': '查询结果',
                'valueType': {'type': 'array'},
                'description': '查询结果数据'
            },
            {
                'id': 'row_count',
                'name': '结果行数',
                'valueType': {'type': 'int'},
                'description': '结果行数'
            },
            {
                'id': 'columns',
                'name': '列名列表',
                'valueType': {'type': 'array'},
                'description': '列名列表'
            },
            {
                'id': 'analysis',
                'name': '分析结果',
                'valueType': {'type': 'object'},
                'description': 'AI数据分析结果'
            },
            {
                'id': 'visualization',
                'name': '可视化配置',
                'valueType': {'type': 'object'},
                'description': '可视化配置'
            },
            {
                'id': 'timestamp',
                'name': '查询时间',
                'valueType': {'type': 'string'},
                'description': '查询时间戳'
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

        # 数据库配置（默认使用环境变量）
        db_type = (getattr(settings, "DB_TYPE", "mysql") or "mysql").lower()
        if db_type in {"postgres", "postgresql", "pg"}:
            self.db_url = (
                f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            )
        else:
            self.db_url = (
                f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            )
        self.engine = None

        # 使用统一的LLM配置系统
        logger.info("[ChatBI] 开始初始化LLM配置...")

        try:
            # 使用修复后的统一配置系统
            self.llm = create_langchain_llm(
                temperature=0.3,
                max_tokens=2000
            )

            # 获取配置信息用于日志
            from app.core.llm.client_factory import get_model_config_manager
            config_manager = get_model_config_manager()

            # 确保配置管理器已初始化
            if not hasattr(config_manager, '_config') or config_manager._config is None:
                config_manager.initialize()

            model_config = config_manager.get_model_config("llm")

            self.model_name = model_config.model_name
            self.base_url = model_config.base_url
            self.original_provider = model_config.provider_name

            logger.info(f"✅ [ChatBI] LLM配置初始化成功")
            logger.info(f"  模型: {self.model_name}")
            logger.info(f"  服务商: {self.original_provider}")
            logger.info(f"  Base URL: {self.base_url}")

        except Exception as e:
            logger.error(f"❌ [ChatBI] LLM配置初始化失败: {e}")
            raise ValueError(
                f"❌ ChatBI工具LLM配置初始化失败！\n"
                f"错误信息: {str(e)}\n"
                "请确保环境变量中配置了有效的LLM参数：\n"
                "1. LLM_API_KEY + LLM_BASE_URL + LLM_MODEL\n"
                "2. 或 DASHSCOPE_API_KEY\n"
                "3. 或 OPENAI_API_KEY + OPENAI_BASE_URL"
            )

        # 缓存数据库schema和连接
        self._schema_cache = None
        self._schema_cache_time = None
        self._engine_cache = {}  # 按数据库连接字符串缓存引擎

        # 安全配置
        self.readonly_mode = True  # 只读模式，禁止 UPDATE/DELETE/DROP
        self.max_query_time = 30   # 最大查询时间（秒）

    def run(
        self,
        query: str,
        host: str = None,
        port: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
        driver: str = 'postgresql',
        visualization_type: str = 'auto',
        limit: int = 100,
        sql_generation_mode: str = 'auto',
        sql_quality_threshold: float = 0.75,
        fast_mode: bool = False,
        query_timeout: int = 30  # 增加默认超时时间到30秒
    ) -> Dict[str, Any]:
        """
        执行 ChatBI 查询

        Args:
            query: 自然语言查询
            host: 数据库主机地址（可选，未提供时使用环境变量）
            port: 数据库端口（可选，未提供时使用环境变量）
            user: 数据库用户名（可选，未提供时使用环境变量）
            password: 数据库密码（可选，未提供时使用环境变量）
            database: 数据库名称（可选，未提供时使用环境变量）
            driver: 数据库驱动类型
            visualization_type: 可视化类型
            limit: 结果数量限制
            sql_generation_mode: SQL生成模式 ('auto'=条件性多生成, 'single'=传统单生成, 'multi'=强制多生成)
            sql_quality_threshold: SQL质量阈值(0-1)，仅在auto模式下有效
            fast_mode: 快速模式 - 禁用自检和复杂优化以提升速度
        query_timeout: 查询超时时间（秒），超过此时间将快速返回

        Returns:
            查询结果和分析报告
        """
        import signal

        # 设置超时控制
        def timeout_handler(signum, frame):
            raise TimeoutError(f"查询超时 ({query_timeout}s)")

        # 注册超时信号（仅在非Windows系统上）
        if hasattr(signal, 'SIGALRM') and not hasattr(signal, 'signal'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(query_timeout)

        try:
            # 智能检测：对简单查询自动启用快速模式
            simple_query_keywords = ['多少个', '数量', '总数', '统计', '查询', '列表', '显示']
            is_simple_query = any(kw in query for kw in simple_query_keywords) and len(query) < 20

            # 如果是简单查询且未明确设置模式，自动启用快速模式
            auto_fast_mode = fast_mode or is_simple_query

            logger.info(f"[ChatBI] 收到查询: {query}, timeout={query_timeout}s, fast_mode={fast_mode}, auto_fast={auto_fast_mode}")
            start_time = time.time()

            # 🔧 优化的数据库连接配置
            # 构建连接字符串
            if host or port or user or password or database:
                # 使用提供的参数，未提供的使用默认值
                db_host = host or settings.DB_HOST
                db_port = port or settings.DB_PORT
                db_user = user or settings.DB_USER
                db_password = password or settings.DB_PASSWORD
                db_name = database or settings.DB_NAME

                logger.info(f"[ChatBI] 使用自定义数据库配置: {driver}://{db_host}:{db_port}/{db_name}")

                # 构建连接字符串
                if driver == 'mysql':
                    self.db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                elif driver == 'postgresql':
                    self.db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                elif driver == 'sqlite':
                    self.db_url = f"sqlite:///{db_name}"
                else:
                    raise ValueError(f"支持的数据库驱动: {driver}")
            else:
                logger.info(f"[ChatBI] 使用环境变量数据库配置")
                db_type = (getattr(settings, "DB_TYPE", "mysql") or "mysql").lower()
                if db_type in {"postgres", "postgresql", "pg"}:
                    self.db_url = (
                        f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}"
                        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
                    )
                else:
                    self.db_url = (
                        f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}"
                        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
                    )

            # 使用引擎缓存避免重复创建连接
            if self.db_url not in self._engine_cache:
                self._engine_cache[self.db_url] = create_engine(
                    self.db_url,
                    pool_pre_ping=True,
                    pool_size=5 if not auto_fast_mode else 2,  # 快速模式使用更小的连接池
                    max_overflow=10 if not auto_fast_mode else 5,
                    echo=False  # 关闭SQL日志以提升速度
                )
                logger.info(f"[ChatBI] 创建新的数据库引擎连接")

            self.engine = self._engine_cache[self.db_url]

            # 0. 根据连接类型配置数据库连接（保留原有方法以支持向后兼容）
            # self._setup_database_connection(connection_type, database, connection_string, db_config)

            # 1. 获取数据库schema
            schema_info = self._get_database_schema()

            # 2. Text-to-SQL 转换（支持条件性多生成）
            # 确定生成策略
            enable_multi_generation = sql_generation_mode != 'single'

            sql_result = self._text_to_sql(
                query,
                schema_info,
                limit,
                enable_multi_generation=enable_multi_generation and not auto_fast_mode,
                quality_threshold=sql_quality_threshold,
                max_candidates=2 if auto_fast_mode else 3,  # 快速模式减少候选数
                fast_mode=auto_fast_mode
            )

            if sql_result['status'] != 'success':
                return sql_result

            sql_query = sql_result['sql']
            logger.info(f"[ChatBI] 生成SQL: {sql_query}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "🔍 [ChatBI] 生成SQL | query=%s | sql=%s | mode=%s | score=%.2f",
                    query,
                    sql_query,
                    sql_result.get('generation_mode', 'unknown'),
                    float(sql_result.get('score', 0) or 0),
                )

            # 3. 安全检查
            if not self._validate_sql_safety(sql_query):
                return {
                    'status': 'error',
                    'error': '检测到不安全的SQL操作（UPDATE/DELETE/DROP等），ChatBI仅支持只读查询'
                }

            # 4. 执行SQL查询
            query_result = self._execute_sql(sql_query)

            if query_result['status'] != 'success':
                return query_result

            # 5. 数据分析（快速模式跳过数据分析以提升速度）
            if auto_fast_mode:
                analysis = {
                    'summary': f'查询完成，返回 {len(query_result["data"])} 行数据',
                    'insights': ['快速模式：已跳过详细数据分析'],
                    'recommendations': ['建议使用标准模式获取详细分析']
                }
            else:
                analysis = self._analyze_data(
                    data=query_result['data'],
                    original_query=query,
                    sql_query=sql_query,
                    fast_mode=auto_fast_mode
                )

            # 6. 生成可视化
            visualization = self._generate_visualization(
                data=query_result['data'],
                viz_type=visualization_type,
                query=query
            )

            # 7. 组装结果（包含SQL生成信息）
            elapsed_time = time.time() - start_time
            result = {
                'status': 'success',
                'query': query,
                'sql': sql_query,
                'data': query_result['data'],
                'row_count': query_result['row_count'],
                'columns': query_result['columns'],
                'analysis': analysis,
                'visualization': visualization,
                'timestamp': datetime.now().isoformat(),
                'elapsed_time': elapsed_time,
                'performance_info': {
                    'fast_mode': auto_fast_mode,
                    'auto_fast_mode': is_simple_query if 'is_simple_query' in locals() else False,
                    'elapsed_time': f"{elapsed_time:.2f}s",
                    'sql_generation_mode': sql_generation_mode
                }
            }

            # 添加SQL生成元信息
            if 'generation_mode' in sql_result:
                result['sql_generation_info'] = {
                    'mode': sql_result.get('generation_mode', 'unknown'),
                    'score': sql_result.get('score', 0),
                    'candidate_count': sql_result.get('candidate_count', 1),
                    'selection_reason': sql_result.get('selection_reason', ''),
                    'base_score': sql_result.get('base_score'),
                    'all_scores': sql_result.get('all_scores', [])
                }

            return result

        except Exception as e:
            # 清理超时信号
            if hasattr(signal, 'SIGALRM') and not hasattr(signal, 'signal'):
                signal.alarm(0)

            # 计算总耗时
            elapsed_time = time.time() - start_time
            logger.info(f"[ChatBI] 查询完成，耗时: {elapsed_time:.2f}s")

            # 如果是超时错误，返回快速响应
            if isinstance(e, TimeoutError):
                logger.warning(f"[ChatBI] 查询超时: {e}")
                return {
                    'status': 'timeout',
                    'error': f'查询超时 ({query_timeout}s)，请简化查询或使用快速模式',
                    'timeout': query_timeout,
                    'elapsed_time': elapsed_time,
                    'fast_mode_suggestion': fast_mode or '建议使用 fast_mode=True 获得更快响应'
                }
            else:
                logger.error(f"[ChatBI] 执行失败: {e}", exc_info=True)
                return {
                    'status': 'error',
                    'error': str(e),
                    'elapsed_time': elapsed_time
                }

    def _get_database_schema(self) -> Dict[str, Any]:
        """
        获取数据库schema信息（缓存5分钟）

        Returns:
            数据库schema信息
        """
        # 检查缓存
        now = datetime.now()
        if self._schema_cache and self._schema_cache_time:
            cache_age = (now - self._schema_cache_time).total_seconds()
            if cache_age < 300:  # 5分钟内
                return self._schema_cache

        try:
            # 创建引擎
            if not self.engine:
                self.engine = create_engine(self.db_url, pool_pre_ping=True)

            # 使用 inspector 获取表信息
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()

            schema_info = {
                'database': settings.DB_NAME,
                'tables': {}
            }

            # 获取每个表的详细信息
            for table_name in tables:
                columns = inspector.get_columns(table_name)

                schema_info['tables'][table_name] = {
                    'columns': [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col['nullable'],
                            'default': str(col['default']) if col['default'] else None
                        }
                        for col in columns
                    ],
                    'comment': f"表: {table_name}"
                }

            # 缓存schema
            self._schema_cache = schema_info
            self._schema_cache_time = now

            logger.info(f"[ChatBI] 获取到 {len(tables)} 个表的schema信息")
            return schema_info

        except Exception as e:
            logger.error(f"[ChatBI] 获取数据库schema失败: {e}")
            return {'database': settings.DB_NAME, 'tables': {}}

    def _text_to_sql(
        self,
        user_query: str,
        schema_info: Dict[str, Any],
        limit: int = 100,
        enable_multi_generation: bool = True,
        quality_threshold: float = 0.5,
        max_candidates: int = 3,
        fast_mode: bool = False
    ) -> Dict[str, Any]:
        """
        使用 LLM 将自然语言转换为 SQL（支持条件性多生成）

        Args:
            user_query: 用户的自然语言查询
            schema_info: 数据库schema信息
            limit: 结果限制
            enable_multi_generation: 是否启用多生成策略
            quality_threshold: 质量阈值，超过此值则不生成更多候选
            max_candidates: 最大候选数量
            fast_mode: 快速模式 - 简化流程提升速度

        Returns:
            包含SQL和生成信息的结果
        """
        try:
            if fast_mode or not enable_multi_generation:
                # 快速模式或单生成模式 - 直接生成，不评分
                return self._fast_sql_generation(user_query, schema_info, limit, fast_mode)

            # 条件性多生成模式
            logger.info(f"[ChatBI] 启用条件性多生成策略，质量阈值: {quality_threshold}")

            # 阶段1：生成基础SQL（标准风格）
            logger.info("[ChatBI] 阶段1: 生成基础SQL")
            base_sql = self._generate_sql_with_style(user_query, schema_info, limit, "standard", fast_mode)

            if not base_sql.strip():
                return {
                    'status': 'error',
                    'error': '基础SQL生成失败'
                }

            # 阶段2：快速评分
            logger.info("[ChatBI] 阶段2: 快速评分基础SQL")
            base_score = self._quick_sql_score(base_sql, user_query, schema_info)
            logger.info(f"[ChatBI] 基础SQL评分: {base_score:.2f}")

            # 阶段3：条件性决策
            if base_score >= quality_threshold:
                logger.info(f"[ChatBI] 基础SQL质量足够({base_score:.2f} >= {quality_threshold})，直接使用")
                return {
                    'status': 'success',
                    'sql': base_sql,
                    'score': base_score,
                    'generation_mode': 'single',
                    'candidate_count': 1,
                    'selection_reason': f'基础SQL质量评分{base_score:.2f}超过阈值{quality_threshold}'
                }

            # 阶段4：生成更多候选（并行）
            logger.info(f"[ChatBI] 基础SQL质量不足({base_score:.2f} < {quality_threshold})，生成更多候选")
            additional_candidates = self._parallel_generate_sql_candidates(
                user_query, schema_info, limit, max_candidates - 1, fast_mode
            )

            # 所有候选（包括基础的）
            all_candidates = [base_sql] + additional_candidates
            logger.info(f"[ChatBI] 总共生成了{len(all_candidates)}个SQL候选")

            # 打印所有SQL候选用于调试
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "🔍 [ChatBI] SQL候选对比 | query=%s | base_score=%.2f | threshold=%.2f | candidates=%d",
                    user_query,
                    base_score,
                    quality_threshold,
                    len(all_candidates),
                )
                for i, sql in enumerate(all_candidates):
                    score = self._quick_sql_score(sql, user_query, schema_info)
                    mode_name = ["基础", "保守", "标准", "激进"][i] if i < 4 else f"候选{i}"
                    logger.debug("  %d. %s: %.2f | SQL: %s", i + 1, mode_name, score, sql)

            # 阶段5：选择最佳SQL
            logger.info("[ChatBI] 阶段5: 选择最佳SQL")
            best_result = self._fast_select_best_sql(all_candidates, user_query, schema_info)

            if not best_result['sql']:
                return {
                    'status': 'error',
                    'error': '所有SQL候选都无效'
                }

            return {
                'status': 'success',
                'sql': best_result['sql'],
                'score': best_result['score'],
                'generation_mode': 'multi',
                'candidate_count': best_result['candidate_count'],
                'selection_reason': best_result['selection_reason'],
                'all_scores': best_result.get('all_scores', []),
                'base_score': base_score
            }

        except Exception as e:
            logger.error(f"[ChatBI] Text-to-SQL转换失败: {e}")
            return {
                'status': 'error',
                'error': f'SQL生成失败: {str(e)}'
            }

    def _fast_sql_generation(
        self,
        user_query: str,
        schema_info: Dict[str, Any],
        limit: int,
        fast_mode: bool
    ) -> Dict[str, Any]:
        """
        快速SQL生成 - 跳过复杂优化，直接生成

        Args:
            user_query: 用户查询
            schema_info: 数据库schema
            limit: 结果限制
            fast_mode: 是否为快速模式

        Returns:
            生成结果
        """
        try:
            if fast_mode:
                # 快速模式：使用简化的提示词
                prompt = self._build_fast_prompt(user_query, schema_info, limit)
            else:
                # 标准单生成模式：使用完整提示词
                prompt = self._build_text_to_sql_prompt(user_query, schema_info, limit, fast_mode)

            # 格式化为消息列表
            messages = [{"role": "user", "content": prompt}]

            # 调用 LLM（传递快速模式参数）
            llm_response = self._call_llm_sync(messages, fast_mode=fast_mode)

            # 解析 SQL
            sql_query = self._extract_sql_from_response(llm_response)

            if not sql_query:
                return {
                    'status': 'error',
                    'error': '无法从LLM响应中提取SQL语句'
                }

            # 快速模式跳过评分，标准模式计算评分（用于日志）
            score = 0.0
            if not fast_mode:
                score = self._quick_sql_score(sql_query, user_query, schema_info)

            # 快速模式也打印SQL
            if fast_mode:
                logger.debug(
                    "🚀 [ChatBI] 快速模式SQL | query=%s | sql=%s | note=%s",
                    user_query,
                    sql_query,
                    "极速生成-跳过质量评分",
                )

            return {
                'status': 'success',
                'sql': sql_query,
                'score': score,
                'generation_mode': 'fast' if fast_mode else 'single',
                'candidate_count': 1,
                'selection_reason': '快速生成模式' if fast_mode else '传统单生成模式'
            }

        except Exception as e:
            logger.error(f"[ChatBI] 快速SQL生成失败: {e}")
            return {
                'status': 'error',
                'error': f'SQL生成失败: {str(e)}'
            }

    def _build_fast_prompt(
        self,
        user_query: str,
        schema_info: Dict[str, Any],
        limit: int
    ) -> str:
        """
        构建超简化的快速提示词 - 最大化减少token消耗
        """
        # 极简的schema信息 - 只保留表名和核心字段
        schema_text = "表: "
        table_names = list(schema_info['tables'].keys())[:3]  # 最多3个表
        schema_text += ", ".join(table_names)

        return f"""生成MySQL查询。

数据库: {schema_text}

问题: {user_query}

要求: SELECT, WHERE, ORDER BY, LIMIT {limit}

SQL:"""

    def _single_sql_generation(
        self,
        user_query: str,
        schema_info: Dict[str, Any],
        limit: int,
        fast_mode: bool = False
    ) -> Dict[str, Any]:
        """
        传统单SQL生成方法

        Args:
            user_query: 用户查询
            schema_info: 数据库schema
            limit: 结果限制
            fast_mode: 是否为快速模式

        Returns:
            生成结果
        """
        try:
            # 构建 Text-to-SQL 提示词
            prompt = self._build_text_to_sql_prompt(user_query, schema_info, limit, fast_mode)

            # 格式化为消息列表
            messages = [{"role": "user", "content": prompt}]

            # 调用 LLM（异步转同步）
            llm_response = self._call_llm_sync(messages, fast_mode=fast_mode)

            # 解析 SQL
            sql_query = self._extract_sql_from_response(llm_response)

            if not sql_query:
                return {
                    'status': 'error',
                    'error': '无法从LLM响应中提取SQL语句'
                }

            # 计算评分（用于日志）
            score = self._quick_sql_score(sql_query, user_query, schema_info)

            return {
                'status': 'success',
                'sql': sql_query,
                'score': score,
                'generation_mode': 'single',
                'candidate_count': 1,
                'selection_reason': '传统单生成模式',
                'explanation': llm_response
            }

        except Exception as e:
            logger.error(f"[ChatBI] 单SQL生成失败: {e}")
            return {
                'status': 'error',
                'error': f'SQL生成失败: {str(e)}'
            }

    def _build_text_to_sql_prompt(
        self,
        user_query: str,
        schema_info: Dict[str, Any],
        limit: int,
        fast_mode: bool = False
    ) -> str:
        """构建增强的 Text-to-SQL 提示词"""

        # 智能过滤相关schema信息（快速模式跳过复杂过滤）
        if fast_mode:
            filtered_schema = self._fast_schema_filter(schema_info)
        else:
            filtered_schema = self._filter_relevant_schema(user_query, schema_info)

        # 格式化schema信息
        schema_text = f"数据库: {filtered_schema['database']}\n\n可用的表:\n\n"

        for table_name, table_info in filtered_schema['tables'].items():
            schema_text += f"表名: {table_name}\n"
            schema_text += "列:\n"
            for col in table_info['columns']:
                schema_text += f"  - {col['name']} ({col['type']})"
                if not col['nullable']:
                    schema_text += " NOT NULL"
                schema_text += "\n"
            schema_text += "\n"

        # 添加查询示例
        examples = self._get_relevant_examples(user_query, filtered_schema)

        # 自适应选择提示词策略
        query_type = self._classify_query_type(user_query)
        prompt_prefix = self._get_adaptive_prompt_prefix(query_type)

        prompt = f"""你是一个专业的MySQL专家。请基于以下数据库结构和用户问题生成准确、高效的SQL查询。

{prompt_prefix}

数据库结构:
{schema_text}

{examples}

用户问题: {user_query}

**生成步骤:**
1. 理解用户意图：识别查询目标、时间范围、统计需求
2. 选择相关表：基于表名和字段名匹配度选择最相关的表
3. 构建WHERE条件：使用合适的字段和操作符
4. 添加排序：按时间或相关字段排序

5. 限制结果：添加LIMIT {limit}

**重要要求:**
- 必须生成符合MySQL语法的SELECT查询
- 明确指定列名，避免使用SELECT *
- 使用合适的WHERE条件，避免全表扫描
- 时间相关查询使用: DATE_SUB(NOW(), INTERVAL n DAY)
- 包含ORDER BY进行合理排序
- 必须添加LIMIT {limit}

**错误示例避免:**
- ❌ SELECT * FROM agents (应该明确指定列)
- ❌ 缺少WHERE条件的时间查询
- ❌ 忘记LIMIT限制
- ❌ 使用不存在的字段名

请逐步思考并生成SQL查询:"""

        return prompt

    def _filter_relevant_schema(self, user_query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """基于查询关键词智能过滤相关表和字段"""
        query_keywords = set(user_query.lower().split())

        # 扩展关键词，包含常见的同义词
        extended_keywords = set()
        for kw in query_keywords:
            if '时间' in kw or 'date' in kw or 'time' in kw:
                extended_keywords.update(['time', 'date', 'created', 'updated', '时间'])
            if '智能体' in kw or 'agent' in kw:
                extended_keywords.update(['agent', '智能体', 'ai'])
            if '使用' in kw or 'use' in kw or 'call' in kw:
                extended_keywords.update(['use', 'call', 'usage', '使用', '调用'])
            if '工具' in kw or 'tool' in kw:
                extended_keywords.update(['tool', '工具', 'function'])
            extended_keywords.add(kw)

        relevant_tables = {}

        for table_name, table_info in schema_info['tables'].items():
            # 检查表名相关性
            table_relevant = False
            if any(kw in table_name.lower() for kw in extended_keywords):
                table_relevant = True

            # 检查列名相关性
            relevant_columns = []
            for col in table_info['columns']:
                if any(kw in col['name'].lower() for kw in extended_keywords):
                    relevant_columns.append(col)
                    table_relevant = True

                # 特殊字段总是相关
                if any(essential in col['name'].lower() for essential in ['id', 'name', 'created', 'updated']):
                    relevant_columns.append(col)

            # 如果表相关或有关联字段，保留表信息
            if table_relevant or relevant_columns:
                filtered_table = table_info.copy()
                if relevant_columns:
                    # 去重并保持顺序
                    seen = set()
                    unique_columns = []
                    for col in relevant_columns + table_info['columns']:
                        col_key = col['name']
                        if col_key not in seen:
                            seen.add(col_key)
                            unique_columns.append(col)
                    filtered_table['columns'] = unique_columns[:10]  # 限制列数量
                relevant_tables[table_name] = filtered_table

        return {**schema_info, 'tables': relevant_tables}

    def _fast_schema_filter(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        快速Schema过滤 - 简化处理
        """
        filtered_tables = {}

        for table_name, table_info in schema_info['tables'].items():
            # 只保留前3个表和每表前5个字段
            if len(filtered_tables) >= 3:
                break

            filtered_table = table_info.copy()
            # 只保留重要字段
            important_cols = []
            for col in table_info['columns']:
                if len(important_cols) >= 5:
                    break
                # 保留关键字段
                if any(imp in col['name'].lower() for imp in ['id', 'name', 'created', 'updated', 'status']):
                    important_cols.append(col)
                elif len(important_cols) < 3:  # 补充一些字段
                    important_cols.append(col)

            filtered_table['columns'] = important_cols
            filtered_tables[table_name] = filtered_table

        return {**schema_info, 'tables': filtered_tables}

    def _classify_query_type(self, user_query: str) -> str:
        """分类查询类型"""
        query_lower = user_query.lower()

        if any(kw in query_lower for kw in ['最近', '时间', '日期', 'today', 'yesterday']):
            return 'time_based'
        elif any(kw in query_lower for kw in ['统计', '总数', '数量', 'count', 'sum', 'avg']):
            return 'aggregation'
        elif any(kw in query_lower for kw in ['对比', '比较', '排名', 'rank', 'top']):
            return 'comparison'
        elif any(kw in query_lower for kw in ['列表', '查询', '显示', 'show']):
            return 'listing'
        else:
            return 'general'

    def _get_adaptive_prompt_prefix(self, query_type: str) -> str:
        """根据查询类型获取自适应提示词前缀"""
        prefixes = {
            'time_based': """**时间查询专家模式:**
- 专注处理时间相关的查询
- 使用DATE_SUB, NOW(), CURDATE()等时间函数
- 注意时区和时间格式转换""",

            'aggregation': """**聚合统计专家模式:**
- 专注处理统计和聚合查询
- 使用COUNT, SUM, AVG, MAX, MIN等聚合函数
- 考虑GROUP BY和HAVING子句""",

            'comparison': """**对比分析专家模式:**
- 专注处理对比和排名查询
- 使用ORDER BY进行排序
- 考虑使用CASE WHEN进行条件判断""",

            'listing': """**列表查询专家模式:**
- 专注处理数据展示查询
- 确保返回清晰可读的结果集
- 合理选择显示字段""",

            'general': """**通用查询专家模式:**
- 综合分析用户需求
- 选择最合适的查询策略"""
        }

        return prefixes.get(query_type, prefixes['general'])

    def _get_relevant_examples(self, user_query: str, schema_info: Dict[str, Any]) -> str:
        """根据查询类型提供相关示例"""
        query_type = self._classify_query_type(user_query)
        table_names = list(schema_info['tables'].keys())

        # 使用实际的表名生成示例
        example_table = table_names[0] if table_names else 'table_name'

        examples = {
            'time_based': f"""
**时间查询示例:**
用户问题: "查询最近7天的{example_table}"
SQL: SELECT id, name, created_at FROM {example_table} WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) ORDER BY created_at DESC LIMIT 100;""",

            'aggregation': f"""
**统计查询示例:**
用户问题: "统计{example_table}的总数"
SQL: SELECT COUNT(*) as total_count FROM {example_table} LIMIT 1;""",

            'comparison': f"""
**对比查询示例:**
用户问题: "查询{example_table}中使用次数最多的前5个"
SQL: SELECT name, use_count FROM {example_table} ORDER BY use_count DESC LIMIT 5;""",

            'listing': f"""
**列表查询示例:**
用户问题: "查询所有{example_table}的名称和ID"
SQL: SELECT id, name FROM {example_table} WHERE status = 'active' ORDER BY created_at DESC LIMIT 100;""",

            'general': f"""
**通用查询示例:**
用户问题: "查询{example_table}信息"
SQL: SELECT id, name, created_at FROM {example_table} WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) ORDER BY created_at DESC LIMIT 100;"""
        }

        return examples.get(query_type, examples['general'])

    def _self_validate_and_fix_sql(self, sql: str, user_query: str, schema_info: Dict[str, Any]) -> str:
        """SQL自检和修正机制"""

        validation_prompt = f"""请检查以下SQL查询是否正确并进行必要的修正：

用户问题：{user_query}
生成的SQL：{sql}
数据库结构：{self._format_schema_for_prompt(schema_info)}

**检查项目:**
1. 表名和字段名是否在数据库中存在
2. WHERE条件是否合理和相关
3. JOIN条件是否正确
4. SQL语法是否有错误
5. 是否符合用户查询意图
6. 是否包含适当的LIMIT子句

**修正要求:**
- 如果SQL完全正确，直接返回原SQL
- 如果有问题，请修正SQL
- 确保修正后的SQL语法正确且能执行
- 保持原有的查询意图

请只返回修正后的SQL语句，不要其他解释："""

        try:
            validation_response = self._call_llm_sync([{"role": "user", "content": validation_prompt}])
            fixed_sql = self._extract_sql_from_response(validation_response)

            if fixed_sql and fixed_sql.strip():
                logger.info(f"[ChatBI] SQL自检修正成功")
                return fixed_sql
            else:
                logger.warning(f"[ChatBI] SQL自检修正失败，保持原SQL")
                return sql

        except Exception as e:
            logger.error(f"[ChatBI] SQL自检过程出错: {e}")
            return sql

    def _extract_sql_from_response(self, response: str) -> Optional[str]:
        """从LLM响应中提取SQL语句"""
        import re

        # 检查是否是错误响应
        if response.strip().startswith('ERROR:'):
            logger.warning(f"[ChatBI] LLM返回错误: {response}")
            return None

        # 移除markdown代码块标记
        response = re.sub(r'```sql\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        response = response.strip()

        # 移除前后的说明文字，只保留SQL
        lines = response.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('#'):
                sql_lines.append(line)

        sql = ' '.join(sql_lines)

        # 确保以分号结尾
        if not sql.endswith(';'):
            sql += ';'

        return sql

    def _validate_sql_safety(self, sql: str) -> bool:
        """
        验证SQL安全性（只读模式）

        Args:
            sql: SQL语句

        Returns:
            是否安全
        """
        if not self.readonly_mode:
            return True

        # 转换为大写检查
        sql_upper = sql.upper()

        # 危险关键词
        dangerous_keywords = [
            'UPDATE', 'DELETE', 'DROP', 'INSERT',
            'ALTER', 'CREATE', 'TRUNCATE', 'REPLACE',
            'GRANT', 'REVOKE'
        ]

        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(f"[ChatBI] 检测到危险SQL关键词: {keyword}")
                return False

        return True

    def _execute_sql(self, sql: str) -> Dict[str, Any]:
        """
        执行SQL查询

        Args:
            sql: SQL查询语句

        Returns:
            查询结果
        """
        try:
            if not self.engine:
                self.engine = create_engine(self.db_url, pool_pre_ping=True)

            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))

                # 获取列名
                columns = list(result.keys())

                # 获取数据
                rows = result.fetchall()

                # 转换为字典列表
                data = [
                    {col: self._convert_value(row[i]) for i, col in enumerate(columns)}
                    for row in rows
                ]

            logger.info(f"[ChatBI] 查询成功，返回 {len(data)} 行数据")

            return {
                'status': 'success',
                'data': data,
                'row_count': len(data),
                'columns': columns
            }

        except Exception as e:
            logger.error(f"[ChatBI] SQL执行失败: {e}")
            return {
                'status': 'error',
                'error': f'SQL执行失败: {str(e)}'
            }

    def _convert_value(self, value: Any) -> Any:
        """转换数据库值为JSON可序列化格式"""
        from datetime import date
        from decimal import Decimal

        if value is None:
            return None
        elif isinstance(value, (datetime, date)):  # 处理日期时间类型
            return value.isoformat()
        elif isinstance(value, Decimal):  # 处理 Decimal 类型
            return float(value)
        elif isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        else:
            return value

    def _analyze_data(
        self,
        data: List[Dict],
        original_query: str,
        sql_query: str,
        fast_mode: bool = False
    ) -> Dict[str, Any]:
        """
        使用 LLM 分析数据并提供洞察

        Args:
            data: 查询结果数据
            original_query: 原始用户问题
            sql_query: 执行的SQL
            fast_mode: 是否为快速模式

        Returns:
            分析结果
        """
        try:
            # 如果数据为空
            if not data:
                return {
                    'summary': '查询未返回任何数据',
                    'insights': [],
                    'recommendations': []
                }

            # 准备数据摘要
            data_summary = self._prepare_data_summary(data)

            # 构建分析提示词
            analysis_prompt = f"""请分析以下数据查询结果，并提供有价值的洞察。

用户问题: {original_query}
执行的SQL: {sql_query}

数据摘要:
- 总行数: {len(data)}
- 列: {', '.join(data[0].keys())}
- 前5行数据: {json.dumps(data[:5], ensure_ascii=False, indent=2)}

数据统计:
{json.dumps(data_summary, ensure_ascii=False, indent=2)}

请提供:
1. 简短总结（1-2句话）
2. 3-5个关键洞察
3. 2-3个建议或行动项

返回JSON格式:
{{
  "summary": "总结",
  "insights": ["洞察1", "洞察2", ...],
  "recommendations": ["建议1", "建议2", ...]
}}"""

            # 格式化为消息列表
            messages = [{"role": "user", "content": analysis_prompt}]

            # 调用 LLM（异步转同步）
            llm_response = self._call_llm_sync(messages, fast_mode=fast_mode)


            # 解析响应
            analysis = self._parse_analysis_response(llm_response)

            return analysis

        except Exception as e:
            logger.error(f"[ChatBI] 数据分析失败: {e}")
            return {
                'summary': f'查询返回 {len(data)} 行数据',
                'insights': [],
                'recommendations': [],
                'error': str(e)
            }

    def _prepare_data_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """准备数据统计摘要"""
        if not data:
            return {}

        summary = {}

        # 使用 pandas 进行统计
        try:
            df = pd.DataFrame(data)

            for col in df.columns:
                col_summary = {}

                if df[col].dtype in ['int64', 'float64']:
                    # 数值列统计
                    col_summary['type'] = 'numeric'
                    col_summary['count'] = int(df[col].count())
                    col_summary['mean'] = float(df[col].mean()) if not df[col].isna().all() else None
                    col_summary['min'] = float(df[col].min()) if not df[col].isna().all() else None
                    col_summary['max'] = float(df[col].max()) if not df[col].isna().all() else None
                else:
                    # 分类列统计
                    col_summary['type'] = 'categorical'
                    col_summary['unique_count'] = int(df[col].nunique())
                    col_summary['top_values'] = df[col].value_counts().head(5).to_dict()

                summary[col] = col_summary

        except Exception as e:
            logger.warning(f"[ChatBI] 数据统计失败: {e}")

        return summary

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """解析LLM的分析响应"""
        import re

        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: 简单解析
                return {
                    'summary': response[:200],
                    'insights': [],
                    'recommendations': []
                }
        except:
            return {
                'summary': response[:200],
                'insights': [],
                'recommendations': []
            }

    def _quick_sql_score(self, sql: str, user_query: str, schema_info: Dict[str, Any]) -> float:
        """
        快速SQL评分（基于规则，无需执行）

        Args:
            sql: SQL语句
            user_query: 用户原始查询
            schema_info: 数据库schema

        Returns:
            评分 (0.0-1.0)
        """
        score = 0.0
        sql_upper = sql.upper()
        sql_lower = sql.lower()

        # 1. 基础语法检查 (0-3分)
        if 'SELECT' in sql_upper:
            score += 1.0
        if 'FROM' in sql_upper:
            score += 1.0
        if sql.rstrip().endswith(';'):
            score += 1.0

        # 2. 安全性和完整性 (0-2分)
        if 'LIMIT' in sql_upper:
            score += 1.0
        if not any(kw in sql_upper for kw in ['UPDATE', 'DELETE', 'DROP', 'INSERT']):
            score += 1.0

        # 3. 查询意图匹配 (0-2分)
        query_lower = user_query.lower()

        # 时间相关查询
        time_keywords = ['最近', '今天', '昨天', '本周', '本月', '最近7天', '最近30天', '时间', '日期']
        if any(kw in query_lower for kw in time_keywords):
            if any(time_func in sql_upper for time_func in ['DATE_SUB', 'NOW()', 'CURDATE()', 'DATE_FORMAT']):
                score += 1.0

        # 排序相关查询
        sort_keywords = ['排序', '最新', '最多', '最少', '最高', '最低', '前', '排']
        if any(kw in query_lower for kw in sort_keywords):
            if 'ORDER BY' in sql_upper:
                score += 1.0

        # 统计相关查询
        stat_keywords = ['统计', '数量', '总数', '平均', '求和', '最大', '最小', '个数']
        if any(kw in query_lower for kw in stat_keywords):
            if any(func in sql_upper for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                score += 1.0

        # 4. 表名匹配 (0-1分)
        table_names = list(schema_info.get('tables', {}).keys())
        matched_tables = sum(1 for table in table_names if table.lower() in sql_lower)
        if matched_tables > 0:
            score += min(matched_tables / max(len(table_names), 1), 1.0)

        # 5. 复杂度惩罚 (最多扣1分)
        complexity_penalty = 0
        if sql_lower.count('join') > 2:
            complexity_penalty += 0.3
        if sql_lower.count('subquery') or sql_lower.count('(select') > 1:
            complexity_penalty += 0.2
        if len(sql.split()) > 60:
            complexity_penalty += 0.2
        if sql_lower.count('union') > 1:
            complexity_penalty += 0.3

        score = max(0, score - complexity_penalty)

        # 归一化到0-1范围 (满分8分)
        return min(score / 8.0, 1.0)

    def _generate_sql_with_style(
        self,
        user_query: str,
        schema_info: Dict[str, Any],
        limit: int,
        style: str = "standard",
        fast_mode: bool = False
    ) -> str:
        """
        生成不同风格的SQL

        Args:
            user_query: 用户查询
            schema_info: 数据库schema
            limit: 结果限制
            style: 风格类型 (conservative, standard, aggressive)

        Returns:
            生成的SQL语句
        """
        schema_text = self._format_schema_for_prompt(schema_info)

        if style == "conservative":
            # 保守风格：简单查询，避免复杂操作
            prompt = f"""你是一个谨慎的SQL专家。请生成最简单、最安全的SQL查询。

数据库结构:
{schema_text}

用户问题: {user_query}

**保守风格要求:**
1. 只使用单表查询，避免JOIN
2. 使用最基础的WHERE条件
3. 必须包含LIMIT {limit}
4. 使用简单的列名，避免复杂表达式
5. 优先确保查询能执行，而不是追求完美

请生成保守的SQL查询:"""

        elif style == "aggressive":
            # 激进风格：复杂查询，尽可能满足需求
            prompt = f"""你是一个高级SQL专家。请生成功能最强、最准确的SQL查询。

数据库结构:
{schema_text}

用户问题: {user_query}

**激进风格要求:**
1. 可以使用JOIN、子查询等高级特性
2. 使用精确的WHERE条件和时间函数
3. 包含合适的聚合函数和排序
4. 优化查询性能，但确保功能完整
5. 必须添加LIMIT {limit}

请生成高级的SQL查询:"""

        else:  # standard
            # 标准风格：平衡的查询
            prompt = f"""你是一个专业的SQL专家。请生成准确、高效的SQL查询。

数据库结构:
{schema_text}

用户问题: {user_query}

**标准风格要求:**
1. 根据需要使用JOIN（如果明确需要多表数据）
2. 使用合适的WHERE条件过滤数据
3. 包含ORDER BY进行合理排序
4. 必须添加LIMIT {limit}限制结果数量
5. 平衡功能性和简洁性

请生成优化的SQL查询:"""

        # 格式化为消息列表
        messages = [{"role": "user", "content": prompt}]

        # 调用LLM
        llm_response = self._call_llm_sync(messages)

        # 提取SQL
        sql_query = self._extract_sql_from_response(llm_response)

        if not sql_query or not sql_query.strip():
            return ""

        # 自检和修正（仅在非快速模式的standard和aggressive模式下）
        if style in ['standard', 'aggressive'] and not fast_mode:
            sql_query = self._self_validate_and_fix_sql(sql_query, user_query, schema_info)

        return sql_query or ""

    def _format_schema_for_prompt(self, schema_info: Dict[str, Any]) -> str:
        """格式化schema信息用于提示词"""
        schema_text = f"数据库: {schema_info['database']}\n\n可用的表:\n\n"

        for table_name, table_info in schema_info['tables'].items():
            schema_text += f"表名: {table_name}\n"
            schema_text += "列:\n"
            for col in table_info['columns']:
                schema_text += f"  - {col['name']} ({col['type']})"
                if not col['nullable']:
                    schema_text += " NOT NULL"
                schema_text += "\n"
            schema_text += "\n"

        return schema_text

    def _fast_select_best_sql(
        self,
        sql_candidates: List[str],
        user_query: str,
        schema_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        从多个SQL候选中快速选择最佳选项

        Args:
            sql_candidates: SQL候选列表
            user_query: 用户查询
            schema_info: 数据库schema

        Returns:
            最佳SQL和评分信息
        """
        if not sql_candidates:
            return {
                'sql': '',
                'score': 0.0,
                'candidate_count': 0,
                'selection_reason': '无有效候选'
            }

        # 并行评分所有候选
        scored_candidates = []
        for i, sql in enumerate(sql_candidates):
            if sql.strip():  # 只评分非空SQL
                score = self._quick_sql_score(sql, user_query, schema_info)
                scored_candidates.append({
                    'sql': sql,
                    'score': score,
                    'index': i
                })

        if not scored_candidates:
            return {
                'sql': '',
                'score': 0.0,
                'candidate_count': len(sql_candidates),
                'selection_reason': '所有候选SQL都无效'
            }

        # 按分数排序，选择最高分
        best_candidate = max(scored_candidates, key=lambda x: x['score'])

        selection_reason = f"从{len(scored_candidates)}个有效候选中选择最佳，评分: {best_candidate['score']:.2f}"
        
        return {
            'sql': best_candidate['sql'],
            'score': best_candidate['score'],
            'candidate_count': len(scored_candidates),
            'selection_reason': selection_reason,
            'all_scores': [(c['index'], c['score']) for c in scored_candidates]
        }

    def _parallel_generate_sql_candidates(
        self,
        user_query: str,
        schema_info: Dict[str, Any],
        limit: int,
        max_candidates: int = 3,
        fast_mode: bool = False
    ) -> List[str]:
        """
        并行生成多个SQL候选

        Args:
            user_query: 用户查询
            schema_info: 数据库schema
            limit: 结果限制
            max_candidates: 最大候选数量

        Returns:
            SQL候选列表
        """
        styles = ["conservative", "standard", "aggressive"][:max_candidates]
        sql_candidates = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_candidates, 3)) as executor:
            # 提交所有生成任务
            future_to_style = {
                executor.submit(self._generate_sql_with_style, user_query, schema_info, limit, style, fast_mode): style
                for style in styles
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_style, timeout=120):
                style = future_to_style[future]
                try:
                    sql = future.result(timeout=15)
                    if sql.strip():
                        sql_candidates.append(sql)
                        logger.info(f"[ChatBI] {style}风格生成SQL成功")
                    else:
                        logger.warning(f"[ChatBI] {style}风格生成SQL为空")
                except Exception as e:
                    logger.error(f"[ChatBI] {style}风格生成SQL失败: {e}")

        return sql_candidates

    def _generate_visualization(
        self,
        data: List[Dict],
        viz_type: str,
        query: str
    ) -> Dict[str, Any]:
        """
        生成可视化配置

        Args:
            data: 数据
            viz_type: 可视化类型
            query: 原始查询

        Returns:
            可视化配置
        """
        if not data:
            return {'type': 'none', 'message': '无数据可视化'}

        try:
            # 自动检测最佳可视化类型
            if viz_type == 'auto':
                viz_type = self._detect_best_viz_type(data)

            if viz_type == 'table':
                return self._generate_table_viz(data)
            elif viz_type == 'bar':
                return self._generate_bar_viz(data)
            elif viz_type == 'line':
                return self._generate_line_viz(data)
            elif viz_type == 'pie':
                return self._generate_pie_viz(data)
            else:
                return self._generate_table_viz(data)

        except Exception as e:
            logger.error(f"[ChatBI] 生成可视化失败: {e}")
            return {'type': 'error', 'message': str(e)}

    def _detect_best_viz_type(self, data: List[Dict]) -> str:
        """自动检测最佳可视化类型"""
        if len(data) > 50:
            return 'table'

        # 检查是否有时间列
        columns = data[0].keys()
        has_time = any('time' in col.lower() or 'date' in col.lower() for col in columns)

        if has_time and len(data) > 2:
            return 'line'
        elif len(data) <= 10:
            return 'bar'
        else:
            return 'table'

    def _generate_table_viz(self, data: List[Dict]) -> Dict[str, Any]:
        """生成表格可视化"""
        return {
            'type': 'table',
            'columns': list(data[0].keys()) if data else [],
            'rows': data,
            'format': 'markdown'
        }

    def _generate_bar_viz(self, data: List[Dict]) -> Dict[str, Any]:
        """生成柱状图配置"""
        if not data:
            return {'type': 'bar', 'data': []}

        columns = list(data[0].keys())
        x_col = columns[0]
        y_col = columns[1] if len(columns) > 1 else columns[0]

        return {
            'type': 'bar',
            'x_axis': x_col,
            'y_axis': y_col,
            'data': data
        }

    def _generate_line_viz(self, data: List[Dict]) -> Dict[str, Any]:
        """生成折线图配置"""
        if not data:
            return {'type': 'line', 'data': []}

        columns = list(data[0].keys())
        x_col = columns[0]
        y_col = columns[1] if len(columns) > 1 else columns[0]

        return {
            'type': 'line',
            'x_axis': x_col,
            'y_axis': y_col,
            'data': data
        }

    def _generate_pie_viz(self, data: List[Dict]) -> Dict[str, Any]:
        """生成饼图配置"""
        if not data:
            return {'type': 'pie', 'data': []}

        columns = list(data[0].keys())
        label_col = columns[0]
        value_col = columns[1] if len(columns) > 1 else columns[0]

        return {
            'type': 'pie',
            'label_column': label_col,
            'value_column': value_col,
            'data': data
        }

    def _call_llm_sync(self, messages: List[Dict[str, str]], timeout: int = 120, fast_mode: bool = False) -> str:
        """
        同步调用LLM的辅助方法（专用于同步工具）
        使用LangChain统一配置系统，支持快速模式

        Args:
            messages: 消息列表
            timeout: 超时时间（秒）
            fast_mode: 是否为快速模式

        Returns:
            LLM响应
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        import httpx
        from openai import APITimeoutError

        # 优化的超时配置
        if fast_mode:
            # 快速模式：更少token，但给足够的响应时间
            max_tokens = 300
            temperature = 0.0
            timeout = min(timeout, 20)  # 增加到20秒，避免不必要的超时
        else:
            # 标准模式：合理配置
            max_tokens = 600
            temperature = 0.1
            timeout = min(timeout, 30)  # 增加到30秒

        try:
            logger.info(f"[ChatBI] 同步调用LLM: {self.model_name}, timeout={timeout}s, fast_mode={fast_mode}")

            # 转换消息格式
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))

            # 创建临时LLM实例用于同步调用
            temp_llm = create_langchain_llm(
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=timeout,
            )

            response = temp_llm.invoke(langchain_messages)
            logger.info(f"[ChatBI] LLM调用成功，响应长度: {len(response.content)}")
            return response.content

        except (APITimeoutError, httpx.TimeoutException):
            logger.error(f"[ChatBI] LLM调用超时: {timeout}s")
            raise Exception(f"LLM调用超时: {timeout}s")
        except Exception as e:
            logger.error(f"[ChatBI] LLM调用失败: {e}")
            raise Exception(f"LLM调用失败: {str(e)}")

    
