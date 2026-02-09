"""
文件上传接口 - 支持图片、视频等多模态内容
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import uuid
import logging
import os
from datetime import datetime

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

ALLOWED_CV_MODEL_EXTENSIONS = {".pt", ".onnx", ".engine", ".pth", ".torchscript"}
MAX_CV_MODEL_SIZE = 1024 * 1024 * 1024  # 1GB

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


def _get_cv_model_dir() -> Path:
    try:
        from app.shared.jetlinks_video.configs.cv_config import build_cv_config
        cfg = build_cv_config()
        model_dir = cfg.model_dir or "storage/models/cv"
    except Exception as e:
        logger.warning("Failed to read cv model config, fallback to storage/models/cv: %s", e)
        model_dir = "storage/models/cv"
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


@router.get("/cv-models")
async def list_cv_models() -> Dict[str, Any]:
    """
    查询本地 CV/YOLO 模型列表
    """
    model_dir = _get_cv_model_dir()
    models: List[Dict[str, Any]] = []
    for item in model_dir.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in ALLOWED_CV_MODEL_EXTENSIONS:
            continue
        stat = item.stat()
        models.append({
            "name": item.name,
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    models.sort(key=lambda x: x["name"].lower())
    return {
        "success": True,
        "model_dir": str(model_dir),
        "models": models,
        "allowed_extensions": sorted(ALLOWED_CV_MODEL_EXTENSIONS),
        "max_size_bytes": MAX_CV_MODEL_SIZE,
    }


@router.post("/cv-model")
async def upload_cv_model(file: UploadFile = File(...), overwrite: bool = False) -> Dict[str, Any]:
    """
    上传 CV/YOLO 模型文件到 storage/models/cv
    """
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_extension = Path(filename).suffix.lower()
    if file_extension not in ALLOWED_CV_MODEL_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported model format: {file_extension}")

    model_dir = _get_cv_model_dir()
    dest_path = model_dir / filename

    if dest_path.exists() and not overwrite:
        raise HTTPException(status_code=409, detail="Model file already exists. Set overwrite=true to replace.")
    if dest_path.exists() and overwrite:
        dest_path.unlink()

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    size = dest_path.stat().st_size
    if size > MAX_CV_MODEL_SIZE:
        try:
            dest_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Model file exceeds size limit")

    logger.info("CV model uploaded successfully: %s", dest_path)
    return {
        "success": True,
        "filename": filename,
        "file_path": str(dest_path.absolute()),
        "size": size,
    }
