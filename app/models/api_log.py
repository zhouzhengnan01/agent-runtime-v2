"""
API日志数据模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.sql import func
from app.db.session import Base


class APILog(Base):
    """API日志表"""
    __tablename__ = "api_logs"

    id = Column(Integer, primary_key=True, index=True, comment="主键ID")

    # 请求信息
    method = Column(String(10), nullable=False, comment="请求方法")
    path = Column(String(500), nullable=False, comment="请求路径")
    query_params = Column(Text, comment="查询参数")
    request_headers = Column(Text, comment="请求头")
    request_body = Column(Text, comment="请求体")

    # 响应信息
    status_code = Column(Integer, comment="响应状态码")
    response_headers = Column(Text, comment="响应头")
    response_body = Column(Text, comment="响应体")

    # 性能信息
    response_time = Column(Float, comment="响应时间（毫秒）")

    # 业务信息
    agent_id = Column(String(100), comment="智能体ID")
    user_id = Column(String(100), comment="用户ID")
    session_id = Column(String(100), comment="会话ID")

    # 错误信息
    error_message = Column(Text, comment="错误信息")

    # 客户端信息
    client_ip = Column(String(50), comment="客户端IP")
    user_agent = Column(Text, comment="用户代理")

    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "method": self.method,
            "path": self.path,
            "query_params": self.query_params,
            "request_headers": self.request_headers,
            "request_body": self.request_body,
            "status_code": self.status_code,
            "response_headers": self.response_headers,
            "response_body": self.response_body,
            "response_time": self.response_time,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "error_message": self.error_message,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }