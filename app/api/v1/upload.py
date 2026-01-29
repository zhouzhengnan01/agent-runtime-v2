"""
文件上传接口 - 支持图片、视频等多模态内容
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List, Optional
from pathlib import Path
import shutil
import uuid
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

# 配置上传目录（由 main.py 挂载到 /static/uploads）
UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 创建视频和图片子目录
VIDEO_DIR = UPLOAD_DIR / "video"
IMAGE_DIR = UPLOAD_DIR / "image"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 支持的文件类型
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v'}
ALLOWED_DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt', '.md'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

@router.post("/video")
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频文件
    """
    try:
        # 检查文件类型
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Only video files are allowed")
        
        # 生成唯一文件名
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = VIDEO_DIR / unique_filename
        
        # 保存文件
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Video uploaded successfully: {file_path}")
        
        return {
            "success": True,
            "file_path": str(file_path.absolute()),
            "filename": unique_filename,
            "original_filename": file.filename,
            "size": file_path.stat().st_size,
            "url": f"/static/uploads/video/{unique_filename}"
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片文件
    """
    try:
        # 检查文件类型
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported image format: {file_extension}")
        
        # 生成唯一文件名
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = IMAGE_DIR / unique_filename
        
        # 保存文件
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE/1024/1024:.0f}MB limit")
        
        with file_path.open("wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Image uploaded successfully: {file_path}")
        
        return {
            "success": True,
            "file_path": str(file_path.absolute()),
            "filename": unique_filename,
            "original_filename": file.filename,
            "size": len(content),
            "media_type": "image",
            "url": f"/static/uploads/image/{unique_filename}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/file") 
async def upload_file(file: UploadFile = File(...)):
    """
    上传通用文件（自动检测类型）
    """
    try:
        # 检测文件类型
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension in ALLOWED_IMAGE_EXTENSIONS:
            media_type = "image"
        elif file_extension in ALLOWED_VIDEO_EXTENSIONS:
            media_type = "video"
        elif file_extension in ALLOWED_DOCUMENT_EXTENSIONS:
            media_type = "document"
        else:
            media_type = "unknown"
        
        # 生成唯一文件名
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # 根据类型选择目录
        if media_type == "video":
            file_path = VIDEO_DIR / unique_filename
        elif media_type == "image":
            file_path = IMAGE_DIR / unique_filename
        else:
            file_path = UPLOAD_DIR / unique_filename
        
        # 保存文件
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE/1024/1024:.0f}MB limit")
        
        with file_path.open("wb") as buffer:
            buffer.write(content)
        
        logger.info(f"File uploaded successfully: {file_path}, type: {media_type}")
        
        return {
            "success": True,
            "file_path": str(file_path.absolute()),
            "filename": unique_filename,
            "original_filename": file.filename,
            "size": len(content),
            "content_type": file.content_type,
            "media_type": media_type,
            "url": f"/static/uploads/{media_type}/{unique_filename}" if media_type in ["video", "image"] else f"/static/uploads/{unique_filename}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    上传多个文件
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed at once")
    
    results = []
    for file in files:
        try:
            result = await upload_file(file)
            results.append(result)
        except HTTPException as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": e.detail
            })
    
    return {
        "success": True,
        "total": len(files),
        "successful": sum(1 for r in results if r.get("success")),
        "results": results
    }
