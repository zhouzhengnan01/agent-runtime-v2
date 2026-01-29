"""
文件管理API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional, List
from app.services.file_service import file_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_type: Optional[str] = Query(None, description="文件类型: image/video/document/audio")
):
    """
    上传文件
    
    返回文件信息，包括：
    - file_id: 文件唯一标识
    - url: HTTP访问URL（给前端用）
    - path: 本地绝对路径（给AI模型用）
    """
    try:
        result = await file_service.upload_file(file, file_type)
        return {
            "success": True,
            "message": "文件上传成功",
            "data": result
        }
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...)
):
    """批量上传文件"""
    results = []
    errors = []
    
    for file in files:
        try:
            result = await file_service.upload_file(file)
            results.append(result)
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "success": len(errors) == 0,
        "message": f"成功上传 {len(results)} 个文件，失败 {len(errors)} 个",
        "data": {
            "uploaded": results,
            "failed": errors
        }
    }


@router.get("/info/{file_id}")
async def get_file_info(file_id: str):
    """获取文件信息"""
    file_info = file_service.get_file_info(file_id)
    
    if not file_info:
        # 尝试从文件系统查找
        file_path = file_service.get_file_path(file_id)
        if file_path:
            file_url = file_service.get_file_url(file_id)
            file_info = {
                "file_id": file_id,
                "path": str(file_path),
                "url": file_url,
                "exists": file_path.exists()
            }
        else:
            raise HTTPException(status_code=404, detail="文件不存在")
    
    return {
        "success": True,
        "data": file_info
    }


@router.get("/url/{file_id}")
async def get_file_url(file_id: str):
    """获取文件访问URL"""
    url = file_service.get_file_url(file_id)
    
    if not url:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return {
        "success": True,
        "data": {
            "file_id": file_id,
            "url": url
        }
    }


@router.get("/path/{file_id}")
async def get_file_path(file_id: str):
    """获取文件本地路径（内部使用）"""
    path = file_service.get_file_path(file_id)
    
    if not path:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return {
        "success": True,
        "data": {
            "file_id": file_id,
            "path": str(path),
            "exists": path.exists()
        }
    }


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """删除文件"""
    success = file_service.delete_file(file_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="文件不存在或删除失败")
    
    return {
        "success": True,
        "message": "文件删除成功"
    }


@router.get("/list")
async def list_files(
    file_type: Optional[str] = Query(None, description="文件类型筛选")
):
    """列出文件"""
    files = file_service.list_files(file_type)
    
    return {
        "success": True,
        "data": {
            "total": len(files),
            "files": files
        }
    }


@router.post("/cleanup/temp")
async def cleanup_temp_files(
    older_than_hours: int = Query(24, description="清理多少小时前的临时文件")
):
    """清理临时文件"""
    file_service.cleanup_temp_files(older_than_hours)
    
    return {
        "success": True,
        "message": f"已清理 {older_than_hours} 小时前的临时文件"
    }