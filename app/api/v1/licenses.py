"""
开源声明和许可证API
"""
from fastapi import APIRouter
from typing import List, Dict, Any

router = APIRouter()


@router.get("/licenses", summary="获取开源组件列表", tags=["System"])
async def get_licenses() -> Dict[str, Any]:
    """
    返回本项目使用的所有开源组件及其许可证信息

    根据AGPLv3.0要求，必须向用户提供获取源码的途径
    """
    return {
        "project": "JetLinks AI Open Platform",
        "version": "1.0.0",
        "notice": "本项目使用了以下开源组件，详见各组件的许可证",
        "components": [
            {
                "name": "MinIO",
                "version": "RELEASE.2024-01-01T16-36-33Z",
                "license": "AGPLv3.0",
                "license_url": "https://www.gnu.org/licenses/agpl-3.0.html",
                "source_code": "https://github.com/minio/minio/tree/RELEASE.2024-01-01T16-36-33Z",
                "homepage": "https://min.io/",
                "modified": False,
                "usage": "作为Milvus向量数据库的对象存储后端",
                "notice": "根据AGPLv3.0 Section 13，用户可从上述地址获取MinIO源代码"
            },
            {
                "name": "Milvus",
                "version": "2.3.x",
                "license": "Apache-2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "source_code": "https://github.com/milvus-io/milvus",
                "homepage": "https://milvus.io/",
                "modified": False,
                "usage": "向量数据库，用于语义搜索和知识检索"
            },
            {
                "name": "FastAPI",
                "version": "0.100+",
                "license": "MIT",
                "license_url": "https://opensource.org/licenses/MIT",
                "source_code": "https://github.com/tiangolo/fastapi",
                "homepage": "https://fastapi.tiangolo.com/",
                "modified": False,
                "usage": "Web框架"
            },
            {
                "name": "LangChain",
                "version": "0.1+",
                "license": "MIT",
                "license_url": "https://opensource.org/licenses/MIT",
                "source_code": "https://github.com/langchain-ai/langchain",
                "homepage": "https://langchain.com/",
                "modified": False,
                "usage": "AI应用框架"
            }
        ],
        "agpl_compliance": {
            "notice": "AGPLv3.0合规声明",
            "statement": "本项目使用MinIO (AGPLv3.0)作为存储服务，未对其进行任何修改。根据AGPLv3.0 Section 13的要求，我们在此向所有用户提供获取MinIO源代码的途径。",
            "source_access": "https://github.com/minio/minio/tree/RELEASE.2024-01-01T16-36-33Z",
            "how_to_get_source": [
                "访问GitHub仓库：https://github.com/minio/minio",
                "切换到tag: RELEASE.2024-01-01T16-36-33Z",
                "或使用git命令：git clone https://github.com/minio/minio.git && cd minio && git checkout RELEASE.2024-01-01T16-36-33Z"
            ]
        },
        "full_notice_document": "/api/v1/system/open-source-notice"
    }


@router.get("/open-source-notice", summary="获取完整开源声明", tags=["System"])
async def get_open_source_notice() -> Dict[str, str]:
    """
    返回完整的开源组件声明文档
    """
    import os
    notice_file = "OPEN_SOURCE_NOTICE.md"

    if os.path.exists(notice_file):
        with open(notice_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            "format": "markdown",
            "content": content
        }
    else:
        return {
            "format": "text",
            "content": "开源声明文档未找到，请查看项目根目录的OPEN_SOURCE_NOTICE.md文件"
        }


@router.get("/agpl-compliance", summary="AGPLv3.0合规信息", tags=["System"])
async def get_agpl_compliance() -> Dict[str, Any]:
    """
    专门的AGPLv3.0合规信息端点

    根据AGPLv3.0 Section 13的要求提供
    """
    return {
        "license": "AGPLv3.0",
        "component": "MinIO",
        "version": "RELEASE.2024-01-01T16-36-33Z",
        "modified": False,
        "source_code_access": {
            "method": "GitHub Repository",
            "url": "https://github.com/minio/minio/tree/RELEASE.2024-01-01T16-36-33Z",
            "instructions": [
                "1. 访问 https://github.com/minio/minio",
                "2. 点击 'Tags' 或 'Releases'",
                "3. 找到 RELEASE.2024-01-01T16-36-33Z",
                "4. 下载源代码或使用 git clone"
            ]
        },
        "license_text_url": "https://www.gnu.org/licenses/agpl-3.0.txt",
        "compliance_statement": "本服务使用MinIO作为独立的对象存储后端，通过标准S3 API进行通信。我们未对MinIO进行任何修改，所有用户均可从上述地址获取MinIO的完整源代码。",
        "contact": "如有任何关于开源许可证的问题，请联系: opensource@jetlinks.com"
    }
