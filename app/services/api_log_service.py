"""
API日志服务
处理日志的创建、查询、统计等业务逻辑
"""
import logging
import asyncio
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.models.api_log import APILog
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


class APILogService:
    """API日志服务类"""

    async def create_log(self, log_data: dict) -> Optional[int]:
        return await asyncio.to_thread(self._create_log_sync, log_data)

    def _create_log_sync(self, log_data: dict) -> Optional[int]:
        """
        创建日志记录

        Args:
            log_data: 日志数据字典

        Returns:
            日志ID，失败返回None
        """
        db: Session = SessionLocal()
        try:
            log = APILog(**log_data)
            db.add(log)
            db.commit()
            db.refresh(log)
            return log.id

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create API log: {e}")
            return None

        finally:
            db.close()

    async def query_logs(
        self,
        path: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_duration: Optional[float] = None,  # 最小耗时（毫秒）
        max_duration: Optional[float] = None,  # 最大耗时（毫秒）
        page_index: int = 0,
        page_size: int = 50
    ) -> Dict:
        """
        查询日志

        Args:
            path: 路径（模糊匹配）
            method: 请求方法
            status_code: 状态码
            agent_id: 智能体ID
            session_id: 会话ID
            user_id: 用户ID
            start_time: 开始时间
            end_time: 结束时间
            min_duration: 最小耗时
            max_duration: 最大耗时
            page_index: 页码
            page_size: 每页大小

        Returns:
            分页结果
        """
        return await asyncio.to_thread(
            self._query_logs_sync,
            path,
            method,
            status_code,
            agent_id,
            session_id,
            user_id,
            start_time,
            end_time,
            min_duration,
            max_duration,
            page_index,
            page_size,
        )

    def _query_logs_sync(
        self,
        path: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        page_index: int = 0,
        page_size: int = 50,
    ) -> Dict:
        db: Session = SessionLocal()
        try:
            query = db.query(APILog)

            # 应用过滤条件
            if path:
                query = query.filter(APILog.path.like(f"%{path}%"))
            if method:
                query = query.filter(APILog.method == method)
            if status_code:
                query = query.filter(APILog.status_code == status_code)
            if agent_id:
                query = query.filter(APILog.agent_id == agent_id)
            if session_id:
                query = query.filter(APILog.session_id == session_id)
            if user_id:
                query = query.filter(APILog.user_id == user_id)
            if start_time:
                query = query.filter(APILog.created_at >= start_time)
            if end_time:
                query = query.filter(APILog.created_at <= end_time)
            if min_duration is not None:
                query = query.filter(APILog.duration_ms >= min_duration)
            if max_duration is not None:
                query = query.filter(APILog.duration_ms <= max_duration)

            # 计算总数
            total = query.count()

            # 分页查询
            logs = query.order_by(APILog.created_at.desc()) \
                .offset(page_index * page_size) \
                .limit(page_size) \
                .all()

            return {
                "total": total,
                "pageIndex": page_index,
                "pageSize": page_size,
                "data": [log.to_dict() for log in logs]
            }

        finally:
            db.close()

    async def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        return await asyncio.to_thread(self._get_statistics_sync, start_time, end_time)

    def _get_statistics_sync(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """
        获取统计信息

        Args:
            start_time: 开始时间（默认最近24小时）
            end_time: 结束时间（默认当前时间）

        Returns:
            统计数据
        """
        db: Session = SessionLocal()
        try:
            # 默认统计最近24小时
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(hours=24)

            # 基础查询
            base_query = db.query(APILog).filter(
                and_(
                    APILog.created_at >= start_time,
                    APILog.created_at <= end_time
                )
            )

            # 总请求数
            total_requests = base_query.count()

            # 按状态码统计
            status_stats = db.query(
                APILog.status_code,
                func.count(APILog.id).label('count')
            ).filter(
                and_(
                    APILog.created_at >= start_time,
                    APILog.created_at <= end_time
                )
            ).group_by(APILog.status_code).all()

            # 按方法统计
            method_stats = db.query(
                APILog.method,
                func.count(APILog.id).label('count')
            ).filter(
                and_(
                    APILog.created_at >= start_time,
                    APILog.created_at <= end_time
                )
            ).group_by(APILog.method).all()

            # 平均响应时间
            avg_duration = db.query(
                func.avg(APILog.duration_ms)
            ).filter(
                and_(
                    APILog.created_at >= start_time,
                    APILog.created_at <= end_time
                )
            ).scalar() or 0

            # 最慢的10个接口
            slowest_apis = db.query(
                APILog.method,
                APILog.path,
                func.avg(APILog.duration_ms).label('avg_duration'),
                func.count(APILog.id).label('count')
            ).filter(
                and_(
                    APILog.created_at >= start_time,
                    APILog.created_at <= end_time
                )
            ).group_by(APILog.method, APILog.path) \
                .order_by(func.avg(APILog.duration_ms).desc()) \
                .limit(10).all()

            # 错误率最高的接口
            error_apis = db.query(
                APILog.method,
                APILog.path,
                func.count(APILog.id).label('error_count')
            ).filter(
                and_(
                    APILog.created_at >= start_time,
                    APILog.created_at <= end_time,
                    APILog.status_code >= 400
                )
            ).group_by(APILog.method, APILog.path) \
                .order_by(func.count(APILog.id).desc()) \
                .limit(10).all()

            return {
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "total_requests": total_requests,
                "status_distribution": {
                    str(code): count for code, count in status_stats
                },
                "method_distribution": {
                    method: count for method, count in method_stats
                },
                "avg_duration_ms": round(avg_duration, 2),
                "slowest_apis": [
                    {
                        "method": method,
                        "path": path,
                        "avg_duration_ms": round(avg_dur, 2),
                        "request_count": count
                    }
                    for method, path, avg_dur, count in slowest_apis
                ],
                "error_apis": [
                    {
                        "method": method,
                        "path": path,
                        "error_count": count
                    }
                    for method, path, count in error_apis
                ]
            }

        finally:
            db.close()

    async def delete_old_logs(self, days: int = 30) -> int:
        return await asyncio.to_thread(self._delete_old_logs_sync, days)

    def _delete_old_logs_sync(self, days: int = 30) -> int:
        """
        删除旧日志

        Args:
            days: 保留天数（删除超过该天数的日志）

        Returns:
            删除的记录数
        """
        db: Session = SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            deleted_count = db.query(APILog).filter(
                APILog.created_at < cutoff_date
            ).delete()

            db.commit()

            logger.info(f"Deleted {deleted_count} old API logs (older than {days} days)")
            return deleted_count

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete old logs: {e}")
            return 0

        finally:
            db.close()
