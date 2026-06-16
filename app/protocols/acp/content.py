from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from acp.schema import (
    AudioContentBlock,
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)

from app.schemas import Attachment


AcpPromptBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)


def prompt_parts_from_dict_blocks(prompt: object) -> tuple[str, list[Attachment]]:
    if isinstance(prompt, str):
        return prompt.strip(), []
    if not isinstance(prompt, list):
        return "", []
    text_parts: list[str] = []
    attachments: list[Attachment] = []
    for index, block in enumerate(prompt):
        if isinstance(block, str):
            text = block.strip()
            if text:
                text_parts.append(text)
            continue
        if not isinstance(block, dict):
            continue
        text, attachment = _dict_block_parts(block, index)
        if text:
            text_parts.append(text)
        if attachment is not None:
            attachments.append(attachment)
    return "\n".join(text_parts), attachments


def prompt_parts_from_sdk_blocks(prompt: list[AcpPromptBlock]) -> tuple[str, list[Attachment]]:
    text_parts: list[str] = []
    attachments: list[Attachment] = []
    for index, block in enumerate(prompt):
        if isinstance(block, TextContentBlock):
            text = block.text.strip()
            if text:
                text_parts.append(text)
        elif isinstance(block, ImageContentBlock):
            attachments.append(
                Attachment(
                    name=_attachment_name(block.uri, f"image-{index + 1}"),
                    mime_type=block.mime_type,
                    data_base64=block.data,
                    metadata={"acp_type": "image", "uri": block.uri},
                )
            )
        elif isinstance(block, AudioContentBlock):
            attachments.append(
                Attachment(
                    name=f"audio-{index + 1}",
                    mime_type=block.mime_type,
                    data_base64=block.data,
                    metadata={"acp_type": "audio"},
                )
            )
        elif isinstance(block, ResourceContentBlock):
            attachments.append(
                Attachment(
                    name=block.name,
                    path=block.uri,
                    mime_type=block.mime_type,
                    metadata={
                        "acp_type": "resource_link",
                        "uri": block.uri,
                        "title": block.title,
                        "description": block.description,
                        "size": block.size,
                    },
                )
            )
        elif isinstance(block, EmbeddedResourceContentBlock):
            text, attachment = _embedded_resource_parts(block.resource, index)
            if text:
                text_parts.append(text)
            if attachment is not None:
                attachments.append(attachment)
    return "\n".join(text_parts), attachments


def _dict_block_parts(block: dict[str, Any], index: int) -> tuple[str, Attachment | None]:
    block_type = str(block.get("type") or "").strip()
    if block_type == "text" or "text" in block or "content" in block:
        text = block.get("text") or block.get("content")
        return (text.strip(), None) if isinstance(text, str) and text.strip() else ("", None)
    if block_type == "image":
        data = _string(block.get("data"))
        mime_type = _string(block.get("mimeType") or block.get("mime_type")) or "image/*"
        uri = _string(block.get("uri") or block.get("url") or block.get("path"))
        if data is None and uri is None:
            return "", None
        return "", Attachment(
            name=_attachment_name(uri, f"image-{index + 1}"),
            path=uri if data is None else None,
            mime_type=mime_type,
            data_base64=data,
            metadata={**_block_meta(block), "acp_type": "image", "uri": uri, "url": _string(block.get("url"))},
        )
    if block_type == "audio":
        data = _string(block.get("data"))
        mime_type = _string(block.get("mimeType") or block.get("mime_type")) or "audio/*"
        if data is None:
            return "", None
        return "", Attachment(
            name=f"audio-{index + 1}",
            mime_type=mime_type,
            data_base64=data,
            metadata={**_block_meta(block), "acp_type": "audio"},
        )
    if block_type == "resource_link":
        uri = _string(block.get("uri") or block.get("url") or block.get("path"))
        name = _string(block.get("name")) or _attachment_name(uri, f"resource-{index + 1}")
        if uri is None:
            return "", None
        mime_type = _string(block.get("mimeType") or block.get("mime_type") or block.get("mediaType"))
        return "", Attachment(
            name=name,
            path=uri,
            mime_type=mime_type,
            metadata={
                **_block_meta(block),
                "acp_type": "resource_link",
                "uri": uri,
                "url": _string(block.get("url")),
                "path": _string(block.get("path")),
                "title": _string(block.get("title")),
                "description": _string(block.get("description")),
                "size": block.get("size") if isinstance(block.get("size"), int) else None,
            },
        )
    if block_type == "resource" and isinstance(block.get("resource"), dict):
        return _dict_embedded_resource_parts(block["resource"], index)
    return "", None


def _embedded_resource_parts(resource: TextResourceContents | BlobResourceContents, index: int) -> tuple[str, Attachment | None]:
    if isinstance(resource, TextResourceContents):
        label = resource.uri or f"embedded-resource-{index + 1}"
        text = f"Embedded resource ({label}):\n{resource.text}".strip()
        return text, Attachment(
            name=_attachment_name(resource.uri, f"resource-{index + 1}.txt"),
            path=resource.uri,
            mime_type=resource.mime_type or "text/plain",
            metadata={"acp_type": "embedded_text_resource", "uri": resource.uri, "text": resource.text},
        )
    return "", Attachment(
        name=_attachment_name(resource.uri, f"resource-{index + 1}"),
        path=resource.uri,
        mime_type=resource.mime_type,
        data_base64=resource.blob,
        metadata={"acp_type": "embedded_blob_resource", "uri": resource.uri},
    )


def _dict_embedded_resource_parts(resource: dict[str, Any], index: int) -> tuple[str, Attachment | None]:
    uri = _string(resource.get("uri")) or f"embedded-resource-{index + 1}"
    mime_type = _string(resource.get("mimeType") or resource.get("mime_type"))
    text = _string(resource.get("text"))
    if text is not None:
        rendered = f"Embedded resource ({uri}):\n{text}".strip()
        return rendered, Attachment(
            name=_attachment_name(uri, f"resource-{index + 1}.txt"),
            path=uri,
            mime_type=mime_type or "text/plain",
            metadata={**_block_meta(resource), "acp_type": "embedded_text_resource", "uri": uri, "text": text},
        )
    blob = _string(resource.get("blob"))
    if blob is None:
        return "", None
    return "", Attachment(
        name=_attachment_name(uri, f"resource-{index + 1}"),
        path=uri,
        mime_type=mime_type,
        data_base64=blob,
        metadata={**_block_meta(resource), "acp_type": "embedded_blob_resource", "uri": uri},
    )


def _attachment_name(uri: str | None, fallback: str) -> str:
    if not uri:
        return fallback
    name = PurePosixPath(uri).name
    return name or fallback


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _block_meta(block: dict[str, Any]) -> dict[str, Any]:
    meta = block.get("_meta")
    return dict(meta) if isinstance(meta, dict) else {}
