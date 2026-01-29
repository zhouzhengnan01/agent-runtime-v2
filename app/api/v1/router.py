"""
API路由主入口
"""
from fastapi import APIRouter
from app.api.v1.websocket import router as websocket_router
from app.api.v1.agent import router as agent_router
from app.api.v1.tools import router as tools_router
from app.api.v1.prompt import router as prompt_router
from app.api.v1.upload import router as upload_router
from app.api.v1.files import router as files_router
# from app.api.v1.auto_execution import router as auto_execution_router  # 文件已删除
from app.api.v1.schema import router as schema_router
from app.api.v1.agents_json import router as agents_json_router
from app.api.v1.review_records import router as review_records_router
from app.api.v1.review_reports import router as review_reports_router
from app.api.v1.hls import router as hls_router
from app.api.v1.clip import router as clip_router
from app.api.v1.patrol_streams import router as patrol_streams_router
# from app.api.v1.api_logs import router as api_logs_router  # 暂时禁用

api_router = APIRouter()

# 注册各个路由
api_router.include_router(websocket_router, prefix="/jaip", tags=["JAIP WebSocket"])
# 同时也注册到 /websocket 以保持兼容性
api_router.include_router(websocket_router, prefix="/websocket", tags=["WebSocket"])

# 注册自动模式WebSocket（如果存在）
try:
    from app.api.v1.websocket_auto import router as websocket_auto_router
    api_router.include_router(websocket_auto_router, prefix="/jaip", tags=["JAIP Auto Mode"])
    api_router.include_router(websocket_auto_router, prefix="/websocket", tags=["WebSocket Auto Mode"])
except ImportError:
    pass  # 如果没有自动模式模块，跳过
api_router.include_router(agent_router, prefix="/agent", tags=["Agents"])
api_router.include_router(tools_router, prefix="/tools", tags=["Tools"])
api_router.include_router(prompt_router, tags=["Prompt"])
api_router.include_router(upload_router, prefix="/upload", tags=["Upload"])
api_router.include_router(files_router, tags=["Files"])
# 注册自动执行路由（已禁用，文件已删除）
# api_router.include_router(auto_execution_router, prefix="/auto", tags=["Auto Execution"])
# 注册Schema查询路由
api_router.include_router(schema_router, prefix="/schemas", tags=["Schemas"])
# 注册JSON智能体接口
api_router.include_router(agents_json_router, tags=["JSON Agents"])
# 复判/调用记录（落盘 + 查询）
api_router.include_router(review_records_router, tags=["Review Records"])
api_router.include_router(review_reports_router, tags=["Review Reports"])
# HLS 直播转发
api_router.include_router(hls_router, tags=["HLS"])
# 视频片段剪辑
api_router.include_router(clip_router, tags=["Clip"])
# 巡检流可达性检测
api_router.include_router(patrol_streams_router, tags=["Patrol Streams"])
# 注册API日志查询路由（暂时禁用）
# api_router.include_router(api_logs_router, prefix="/api-logs", tags=["API Logs"])
# 注册JetLinks适配器路由（不加前缀，直接挂载到根路径）

@api_router.get("/")
async def api_root():
    """API根路径"""
    return {
        "message": "JetLinks Agent API v1",
        "endpoints": [
            "/jaip/session/{agent_id} - WebSocket接口（JAIP协议，完整功能）",
            "/agent - 智能体管理接口",
            "/agent/chat - 智能体对话接口（需要agent_id）",
            "/agent/agent-templates - 智能体模板列表",
            "/agent/create - 创建智能体",
            "/agents/{agent_id}/chat/json - JSON格式智能体对话接口",
            "/agents/json/health - JSON处理器健康检查",
            "/agents/{agent_id}/json/info - 智能体JSON配置信息",
            "/tools - 工具管理接口",
            "/prompt/optimize - 优化提示词",
            "/prompt/analyze - 分析提示词",
            "/prompt/templates - 提示词模板",
            "/schemas/template/{template_name}/schemas - 查询模板支持的JSON结构"
        ],
        "protocol_features": [
            "会话管理（session.create, session.heartbeat）",
            "消息传递（session.message, user.message）",
            "AI响应（ai.responseStart, ai.responseChunk, ai.responseComplete）",
            "工具调用（tool.confirm, tool.call, tool.callClient）",
            "双向透传（agent.command, agent.message）",
            "流式输出支持",
            "JSON-RPC 2.0标准格式"
        ]
    }
