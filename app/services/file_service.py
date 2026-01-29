"""
文件服务 - 统一的文件上传、存储和访问服务
"""
import os
import uuid
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import shutil
import mimetypes
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)


class FileService:
    """统一的文件服务，处理所有文件相关操作"""
    
    def __init__(self):
        # 基础存储路径（绝对路径）
        self.base_path = Path(os.path.abspath("storage/uploads"))
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 各类型子目录
        self.paths = {
            "image": self.base_path / "images",
            "video": self.base_path / "videos", 
            "document": self.base_path / "documents",
            "audio": self.base_path / "audio",
            "temp": self.base_path / "temp"
        }
        
        # 创建所有子目录
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        # 文件信息缓存（实际生产环境应该使用数据库）
        self._file_registry = {}
        
        logger.info(f"FileService initialized with base path: {self.base_path}")
    
    async def upload_file(
        self, 
        file: UploadFile,
        file_type: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        上传文件并返回访问信息
        
        Args:
            file: 上传的文件对象
            file_type: 文件类型（可选，会自动检测）
            metadata: 额外的元数据
            
        Returns:
            {
                "file_id": "unique_id",
                "filename": "original_name.ext",
                "url": "/static/uploads/images/xxx.jpg",
                "path": "/absolute/path/storage/uploads/images/xxx.jpg",
                "type": "image",
                "size": 1024,
                "mime_type": "image/jpeg",
                "created_at": "2024-01-01T00:00:00"
            }
        """
        try:
            # 1. 读取文件内容
            content = await file.read()
            await file.seek(0)  # 重置文件指针
            
            # 2. 生成唯一文件ID和文件名
            file_ext = Path(file.filename).suffix.lower()
            file_id = str(uuid.uuid4())
            file_hash = hashlib.md5(content).hexdigest()[:8]
            new_filename = f"{file_id}_{file_hash}{file_ext}"
            
            # 3. 检测文件类型
            if not file_type:
                file_type = self._detect_file_type(file.content_type, file_ext)
            
            # 4. 确定存储路径
            target_dir = self.paths.get(file_type, self.paths["temp"])
            file_path = target_dir / new_filename
            
            # 5. 保存文件
            with open(file_path, "wb") as f:
                f.write(content)
            
            # 6. 生成访问信息
            relative_path = file_path.relative_to(Path("storage"))
            
            file_info = {
                "file_id": file_id,
                "filename": file.filename,
                "url": f"/static/{relative_path.as_posix()}",  # 使用POSIX路径格式
                "path": str(file_path.absolute()),  # 绝对路径
                "type": file_type,
                "size": len(content),
                "mime_type": file.content_type or mimetypes.guess_type(file.filename)[0],
                "created_at": datetime.now().isoformat(),
                "hash": file_hash
            }
            
            # 7. 添加元数据
            if metadata:
                file_info["metadata"] = metadata
            
            # 8. 注册文件信息
            self._file_registry[file_id] = file_info
            
            logger.info(f"File uploaded successfully: {file_id} -> {file_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    
    def get_file_info(self, file_id: str) -> Optional[Dict]:
        """根据file_id获取文件信息"""
        return self._file_registry.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """根据file_id获取文件的绝对路径"""
        # 优先从注册表获取
        file_info = self._file_registry.get(file_id)
        if file_info:
            return Path(file_info["path"])
        
        # 否则在所有目录中查找
        for dir_path in self.paths.values():
            for file_path in dir_path.glob(f"{file_id}*"):
                return file_path.absolute()
        return None
    
    def get_file_url(self, file_id: str) -> Optional[str]:
        """根据file_id获取文件的URL"""
        # 优先从注册表获取
        file_info = self._file_registry.get(file_id)
        if file_info:
            return file_info["url"]
        
        # 否则根据文件路径生成
        file_path = self.get_file_path(file_id)
        if file_path:
            relative_path = file_path.relative_to(Path("storage"))
            return f"/static/{relative_path.as_posix()}"
        return None
    
    def get_file_info_from_url(self, url: str) -> Optional[Dict]:
        """从URL反推文件信息"""
        # 提取相对路径部分
        if "/static/" in url:
            relative_path = url.split("/static/")[-1]
            file_path = Path("storage") / relative_path
            
            if file_path.exists():
                # 尝试从文件名提取file_id
                filename = file_path.name
                if "_" in filename:
                    file_id = filename.split("_")[0]
                    
                    # 尝试从注册表获取完整信息
                    file_info = self._file_registry.get(file_id)
                    if file_info:
                        return file_info
                    
                    # 否则构建基本信息
                    return {
                        "file_id": file_id,
                        "path": str(file_path.absolute()),
                        "url": url,
                        "type": self._detect_file_type_from_path(file_path)
                    }
        return None
    
    def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        file_path = self.get_file_path(file_id)
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                # 从注册表移除
                if file_id in self._file_registry:
                    del self._file_registry[file_id]
                logger.info(f"File deleted: {file_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete file {file_id}: {e}")
        return False
    
    def list_files(self, file_type: Optional[str] = None) -> List[Dict]:
        """列出文件"""
        files = []
        
        if file_type and file_type in self.paths:
            # 列出特定类型的文件
            target_dir = self.paths[file_type]
            for file_path in target_dir.glob("*"):
                if file_path.is_file():
                    files.append(self._get_file_basic_info(file_path))
        else:
            # 列出所有文件
            for dir_path in self.paths.values():
                for file_path in dir_path.glob("*"):
                    if file_path.is_file():
                        files.append(self._get_file_basic_info(file_path))
        
        return files
    
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """清理临时文件"""
        import time
        current_time = time.time()
        temp_dir = self.paths["temp"]
        
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
                if file_age_hours > older_than_hours:
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up temp file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to clean up {file_path}: {e}")
    
    def _detect_file_type(self, content_type: str, file_ext: str = "") -> str:
        """检测文件类型"""
        # 根据MIME类型判断
        if content_type:
            if content_type.startswith("image/"):
                return "image"
            elif content_type.startswith("video/"):
                return "video"
            elif content_type.startswith("audio/"):
                return "audio"
            elif content_type in ["application/pdf", "text/plain", 
                                 "application/msword", "application/vnd.openxmlformats-officedocument"]:
                return "document"
        
        # 根据文件扩展名判断
        if file_ext:
            ext = file_ext.lower()
            if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
                return "image"
            elif ext in [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"]:
                return "video"
            elif ext in [".mp3", ".wav", ".flac", ".aac", ".ogg"]:
                return "audio"
            elif ext in [".pdf", ".txt", ".doc", ".docx", ".xls", ".xlsx"]:
                return "document"
        
        return "temp"
    
    def _detect_file_type_from_path(self, file_path: Path) -> str:
        """从文件路径检测类型"""
        # 根据父目录判断
        parent_name = file_path.parent.name
        if parent_name in ["images", "videos", "documents", "audio"]:
            return parent_name.rstrip("s")  # 移除复数s
        
        # 根据扩展名判断
        return self._detect_file_type("", file_path.suffix)
    
    def _get_file_basic_info(self, file_path: Path) -> Dict:
        """获取文件基本信息"""
        relative_path = file_path.relative_to(Path("storage"))
        return {
            "filename": file_path.name,
            "url": f"/static/{relative_path.as_posix()}",
            "path": str(file_path.absolute()),
            "size": file_path.stat().st_size,
            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }


# 全局文件服务实例
file_service = FileService()