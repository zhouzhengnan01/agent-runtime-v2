"""
API日志查询接口
提供日志查询、统计、清理等功能
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime

from app.services.api_log_service import APILogService
from app.schemas.pagination import StandardResponse

router = APIRouter()


@router.post("/query")
async def query_logs(
    path: Optional[str] = None,
    method: Optional[str] = Query(None, description="请求方法: GET/POST/PUT/DELETE"),
    status_code: Optional[int] = Query(None, description="状态码"),
    agent_id: Optional[str] = Query(None, description="智能体ID"),
    session_id: Optional[str] = Query(None, description="会话ID"),
    user_id: Optional[str] = Query(None, description="用户ID"),
    start_time: Optional[str] = Query(None, description="开始时间 (ISO格式)"),
    end_time: Optional[str] = Query(None, description="结束时间 (ISO格式)"),
    min_duration: Optional[float] = Query(None, description="最小耗时（毫秒）"),
    max_duration: Optional[float] = Query(None, description="最大耗时（毫秒）"),
    page_index: int = Query(0, ge=0, description="页码"),
    page_size: int = Query(50, ge=1, le=200, description="每页大小")
):
    """
    查询API日志

    支持多维度过滤：
    - 路径（模糊匹配）
    - 请求方法
    - 状态码
    - 智能体ID
    - 会话ID
    - 用户ID
    - 时间范围
    - 耗时范围

    返回格式：
    ```json
    {
        "status": 200,
        "message": "success",
        "result": {
            "total": 100,
            "pageIndex": 0,
            "pageSize": 50,
            "data": [...]
        }
    }
    ```
    """
    try:
        # 解析时间参数
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        service = APILogService()
        result = await service.query_logs(
            path=path,
            method=method,
            status_code=status_code,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            start_time=start_dt,
            end_time=end_dt,
            min_duration=min_duration,
            max_duration=max_duration,
            page_index=page_index,
            page_size=page_size
        )

        return StandardResponse(
            status=200,
            message="success",
            result=result
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid time format: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(
    start_time: Optional[str] = Query(None, description="开始时间 (ISO格式，默认24小时前)"),
    end_time: Optional[str] = Query(None, description="结束时间 (ISO格式，默认当前时间)")
):
    """
    获取API调用统计

    统计信息包括：
    - 总请求数
    - 状态码分布
    - 请求方法分布
    - 平均响应时间
    - 最慢的10个接口
    - 错误率最高的10个接口

    示例：
    ```
    GET /api/v1/api-logs/statistics
    GET /api/v1/api-logs/statistics?start_time=2025-01-01T00:00:00&end_time=2025-01-02T00:00:00
    ```

    返回格式：
    ```json
    {
        "status": 200,
        "message": "success",
        "result": {
            "time_range": {...},
            "total_requests": 1000,
            "status_distribution": {...},
            "method_distribution": {...},
            "avg_duration_ms": 123.45,
            "slowest_apis": [...],
            "error_apis": [...]
        }
    }
    ```
    """
    try:
        # 解析时间参数
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        service = APILogService()
        result = await service.get_statistics(
            start_time=start_dt,
            end_time=end_dt
        )

        return StandardResponse(
            status=200,
            message="success",
            result=result
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid time format: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup")
async def cleanup_old_logs(
    days: int = Query(30, ge=1, le=365, description="保留天数（删除超过该天数的日志）")
):
    """
    清理旧日志

    删除超过指定天数的日志记录。

    示例：
    ```
    DELETE /api/v1/api-logs/cleanup?days=30
    ```

    返回格式：
    ```json
    {
        "status": 200,
        "message": "success",
        "result": {
            "deleted_count": 1234
        }
    }
    ```
    """
    try:
        service = APILogService()
        deleted_count = await service.delete_old_logs(days=days)

        return StandardResponse(
            status=200,
            message=f"Deleted {deleted_count} old logs (older than {days} days)",
            result={
                "deleted_count": deleted_count,
                "retention_days": days
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
