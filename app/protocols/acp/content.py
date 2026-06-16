from __future__ import annotations

import mimetypes
import re
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

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


_DATA_URL_RE = re.compile(r"^data:([^;,]+)?(?:;[^,]*)?;base64,(.*)$", re.IGNORECASE | re.DOTALL)
_ATTACHMENT_COLLECTION_KEYS: tuple[tuple[str, str | None], ...] = (
    ("attachments", None),
    ("attachment", None),
    ("files", None),
    ("file", None),
    ("fileResults", None),
    ("fileResult", None),
    ("resources", None),
    ("resource", None),
    ("media", None),
    ("medias", None),
    ("images", "image"),
    ("image", "image"),
    ("imageUrl", "image"),
    ("image_url", "image"),
    ("pictures", "image"),
    ("picture", "image"),
    ("snapshots", "image"),
    ("snapshot", "image"),
    ("videos", "video"),
    ("video", "video"),
    ("videoUrl", "video"),
    ("video_url", "video"),
    ("audios", "audio"),
    ("audio", "audio"),
    ("fileUrl", None),
    ("file_url", None),
    ("resourceUrl", None),
    ("resource_url", None),
)
_DIRECT_URI_KEYS = (
    "uri",
    "url",
    "path",
    "href",
    "src",
    "downloadUrl",
    "download_url",
    "fileUrl",
    "file_url",
    "resourceUrl",
    "resource_url",
    "imageUrl",
    "image_url",
    "videoUrl",
    "video_url",
)
_MIME_KEYS = ("mimeType", "mime_type", "mediaType", "media_type", "contentType", "content_type")
_NAME_KEYS = ("name", "filename", "fileName", "file_name", "title")
_DATA_KEYS = ("data", "dataBase64", "data_base64", "base64", "blob")
_RELATED_VIDEO_URI_KEYS = (
    "record",
    "recordUrl",
    "record_url",
    "mp4",
    "video",
    "videoUrl",
    "video_url",
)
_VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".mpeg",
    ".mpg",
    ".ts",
    ".m3u8",
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
            attachments.extend(_related_video_attachments_from_dict_block(block, index, primary=attachment))
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
    if block_type == "resource" and isinstance(block.get("resource"), dict):
        return _dict_embedded_resource_parts(block["resource"], index)
    attachment = attachment_from_dict_block(block, index)
    if attachment is not None:
        return "", attachment
    if block_type == "text" or "text" in block or "content" in block:
        text = block.get("text") or block.get("content")
        return (text.strip(), None) if isinstance(text, str) and text.strip() else ("", None)
    return "", None


def attachments_from_dict_collection(value: object, *, default_kind: str | None = None) -> list[Attachment]:
    return _attachments_from_dict_collection(value, default_kind=default_kind, index_prefix=default_kind or "attachment")


def attachments_from_payload_fields(source: dict[str, Any]) -> list[Attachment]:
    attachments: list[Attachment] = []
    for key, default_kind in _ATTACHMENT_COLLECTION_KEYS:
        if key not in source:
            continue
        attachments.extend(attachments_from_dict_collection(source[key], default_kind=default_kind))
    attachments.extend(_attachments_from_file_result_containers(source.get("data")))
    return attachments


def _attachments_from_file_result_containers(value: object, *, depth: int = 0) -> list[Attachment]:
    if depth > 4:
        return []
    if isinstance(value, list):
        attachments: list[Attachment] = []
        for item in value:
            attachments.extend(_attachments_from_file_result_containers(item, depth=depth + 1))
        return attachments
    if not isinstance(value, dict):
        return []
    attachments = []
    for key in ("fileResults", "fileResult"):
        if key in value:
            attachments.extend(attachments_from_dict_collection(value[key]))
    for key in ("data", "records", "rows", "items", "list"):
        if key in value:
            attachments.extend(_attachments_from_file_result_containers(value[key], depth=depth + 1))
    return attachments


def attachment_from_dict_block(
    block: dict[str, Any],
    index: int,
    *,
    default_kind: str | None = None,
) -> Attachment | None:
    block_type = _normalized_type(block.get("type"))
    if block_type in {"text", "input_text"}:
        return None
    source = block.get("source") if isinstance(block.get("source"), dict) else None
    nested_image_url = _nested_url(block.get("image_url"))
    nested_video_url = _nested_url(block.get("video_url"))
    nested_file_url = _nested_url(block.get("file"))
    data_url = _first_string(block, ("dataUrl", "data_url"))
    uri = (
        nested_image_url
        or nested_video_url
        or nested_file_url
        or _first_string(block, _DIRECT_URI_KEYS)
        or (source and _first_string(source, _DIRECT_URI_KEYS))
    )
    data = _first_string(block, _DATA_KEYS)
    if data is None and source is not None:
        data = _first_string(source, _DATA_KEYS)
    mime_type = _first_string(block, _MIME_KEYS) or (source and _first_string(source, _MIME_KEYS))

    if data_url is None and isinstance(uri, str) and uri.strip().lower().startswith("data:"):
        data_url = uri
        uri = None
    if data_url is not None:
        parsed_mime, parsed_data = _parse_data_url(data_url)
        if parsed_data is None:
            return None
        data = parsed_data
        mime_type = mime_type or parsed_mime

    inferred_kind = _attachment_kind(block_type, mime_type, default_kind, uri)
    if inferred_kind is None:
        return None
    if data is None and uri is None:
        return None

    mime_type = mime_type or _default_mime_type(inferred_kind, uri)
    name = _attachment_display_name(block, uri, f"{inferred_kind}-{index + 1}")
    reserved = {
        "type",
        "source",
        "image_url",
        "video_url",
        "file",
        "dataUrl",
        "data_url",
        *_DIRECT_URI_KEYS,
        *_MIME_KEYS,
        *_NAME_KEYS,
        *_DATA_KEYS,
    }
    metadata = {
        **_block_extra_metadata(block, reserved),
        "acp_type": inferred_kind,
        "uri": uri,
        "url": uri if isinstance(uri, str) and _is_remote_uri(uri) else _string(block.get("url")),
        "path": uri if isinstance(uri, str) and not _is_remote_uri(uri) else _string(block.get("path")),
        "title": _string(block.get("title")),
        "description": _string(block.get("description")),
        "size": block.get("size") if isinstance(block.get("size"), int) else None,
    }
    return Attachment(
        name=name,
        path=uri,
        mime_type=mime_type,
        data_base64=data,
        metadata={key: value for key, value in metadata.items() if value not in (None, "", [], {})},
    )


def _block_extra_metadata(block: dict[str, Any], reserved_keys: set[str]) -> dict[str, Any]:
    metadata = block.get("_meta") if isinstance(block.get("_meta"), dict) else {}
    extra = {
        key: value
        for key, value in block.items()
        if key not in reserved_keys and key != "_meta" and value not in (None, "", [], {})
    }
    return {**metadata, **extra}


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
            metadata={"acp_type": "embedded_text_resource", "uri": uri, "text": text},
        )
    blob = _string(resource.get("blob"))
    if blob is None:
        return "", None
    return "", Attachment(
        name=_attachment_name(uri, f"resource-{index + 1}"),
        path=uri,
        mime_type=mime_type,
        data_base64=blob,
        metadata={"acp_type": "embedded_blob_resource", "uri": uri},
    )


def _attachment_name(uri: str | None, fallback: str) -> str:
    if not uri:
        return fallback
    parsed = urlparse(uri)
    raw_name = parsed.path if parsed.scheme else uri
    name = PurePosixPath(unquote(raw_name)).name
    return name or fallback


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _attachments_from_dict_collection(
    value: object,
    *,
    default_kind: str | None,
    index_prefix: str,
) -> list[Attachment]:
    if isinstance(value, str):
        attachment = _string_attachment(value, default_kind=default_kind, index_prefix=index_prefix, index=0)
        return [attachment] if attachment is not None else []
    if isinstance(value, list):
        attachments: list[Attachment] = []
        for index, item in enumerate(value):
            if isinstance(item, str):
                attachment = _string_attachment(
                    item,
                    default_kind=default_kind,
                    index_prefix=index_prefix,
                    index=index,
                )
            elif isinstance(item, dict):
                attachment = attachment_from_dict_block(item, index, default_kind=default_kind)
            else:
                attachment = None
            if attachment is not None:
                attachments.append(attachment)
                attachments.extend(_related_video_attachments_from_dict_block(item, index, primary=attachment))
        return attachments
    if not isinstance(value, dict):
        return []

    direct = attachment_from_dict_block(value, 0, default_kind=default_kind)
    if direct is not None:
        return [direct, *_related_video_attachments_from_dict_block(value, 0, primary=direct)]

    attachments: list[Attachment] = []
    for key, nested_kind in _ATTACHMENT_COLLECTION_KEYS:
        if key not in value:
            continue
        attachments.extend(
            _attachments_from_dict_collection(
                value[key],
                default_kind=nested_kind or default_kind,
                index_prefix=nested_kind or key,
            )
        )
    return attachments


def _related_video_attachments_from_dict_block(
    block: dict[str, Any],
    index: int,
    *,
    primary: Attachment,
) -> list[Attachment]:
    attachments: list[Attachment] = []
    seen: set[str] = {str(primary.path or "").strip()}
    for key_path, uri, explicit_mime_type, explicit_name in _iter_related_video_uris(block):
        clean_uri = uri.strip()
        if not clean_uri or clean_uri in seen:
            continue
        seen.add(clean_uri)
        mime_type = explicit_mime_type or _guess_mime_type(clean_uri) or "video/mp4"
        metadata = dict(primary.metadata) if isinstance(primary.metadata, dict) else {}
        metadata.update(
            {
                "acp_type": "video",
                "source": "record",
                "record_key": key_path,
                "record_for": primary.name,
                "uri": clean_uri,
                "url": clean_uri if _is_remote_uri(clean_uri) else None,
                "path": clean_uri if not _is_remote_uri(clean_uri) else None,
            }
        )
        attachments.append(
            Attachment(
                name=explicit_name or _attachment_name(clean_uri, f"record-{index + 1}.mp4"),
                path=clean_uri,
                mime_type=mime_type,
                metadata={key: value for key, value in metadata.items() if value not in (None, "", [], {})},
            )
        )
    return attachments


def _iter_related_video_uris(
    value: Any,
    *,
    prefix: str = "",
    depth: int = 0,
) -> list[tuple[str, str, str | None, str | None]]:
    if depth > 8:
        return []
    found: list[tuple[str, str, str | None, str | None]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            key_path = f"{prefix}.{key}" if prefix else key
            if isinstance(item, str) and key in _RELATED_VIDEO_URI_KEYS and _looks_like_video_uri(item, key):
                found.append((key_path, item, None, None))
            elif isinstance(item, dict):
                record_parts = _related_video_parts_from_mapping(key, item)
                if record_parts is not None:
                    uri, mime_type, name = record_parts
                    found.append((key_path, uri, mime_type, name))
                    continue
                found.extend(_iter_related_video_uris(item, prefix=key_path, depth=depth + 1))
            elif isinstance(item, list):
                found.extend(_iter_related_video_uris(item, prefix=key_path, depth=depth + 1))
    elif isinstance(value, list):
        for item_index, item in enumerate(value):
            if isinstance(item, dict | list):
                found.extend(_iter_related_video_uris(item, prefix=f"{prefix}[{item_index}]", depth=depth + 1))
            elif isinstance(item, str) and _looks_like_video_uri(item, prefix.rsplit(".", 1)[-1]):
                found.append((f"{prefix}[{item_index}]", item, None, None))
    return found


def _related_video_parts_from_mapping(
    key: str,
    value: dict[str, Any],
) -> tuple[str, str | None, str | None] | None:
    if key not in _RELATED_VIDEO_URI_KEYS:
        return None
    uri = _nested_url(value) or _first_string(value, _DIRECT_URI_KEYS)
    if uri is None:
        return None
    mime_type = _first_string(value, _MIME_KEYS)
    if not _is_video_mime_type(mime_type) and not _looks_like_video_uri(uri, key):
        return None
    return uri, mime_type, _first_string(value, _NAME_KEYS)


def _looks_like_video_uri(value: str, key: str) -> bool:
    uri = value.strip()
    if not uri:
        return False
    lowered = uri.lower()
    if lowered.startswith("data:"):
        return lowered.startswith("data:video/")
    guessed = _guess_mime_type(uri)
    if guessed and guessed.lower().startswith("video/"):
        return True
    parsed = urlparse(uri)
    path = unquote(parsed.path if parsed.scheme else uri).lower()
    if any(path.endswith(extension) for extension in _VIDEO_EXTENSIONS):
        return True
    if _normalize_related_key(key) in {"record", "mp4", "video", "videourl", "recordurl"}:
        return bool(parsed.scheme and parsed.netloc) or path.startswith("/")
    return False


def _is_video_mime_type(mime_type: str | None) -> bool:
    return (mime_type or "").split(";", 1)[0].strip().lower().startswith("video/")


def _normalize_related_key(key: str) -> str:
    return key.replace("_", "").replace("-", "").lower()


def _string_attachment(
    value: str,
    *,
    default_kind: str | None,
    index_prefix: str,
    index: int,
) -> Attachment | None:
    uri = value.strip()
    if not uri:
        return None
    if uri.lower().startswith("data:"):
        mime_type, data = _parse_data_url(uri)
        if data is None:
            return None
        kind = _attachment_kind(default_kind, mime_type, default_kind, None) or "resource_link"
        return Attachment(
            name=f"{index_prefix}-{index + 1}",
            mime_type=mime_type or _default_mime_type(kind, None),
            data_base64=data,
            metadata={"acp_type": kind},
        )
    kind = default_kind or _kind_from_mime_type(_guess_mime_type(uri)) or "resource_link"
    return Attachment(
        name=_attachment_name(uri, f"{index_prefix}-{index + 1}"),
        path=uri,
        mime_type=_default_mime_type(kind, uri),
        metadata={
            "acp_type": kind,
            "uri": uri,
            "url": uri if _is_remote_uri(uri) else None,
            "path": uri if not _is_remote_uri(uri) else None,
        },
    )


def _attachment_display_name(block: dict[str, Any], uri: str | None, fallback: str) -> str:
    return _first_string(block, _NAME_KEYS) or _attachment_name(uri, fallback)


def _attachment_kind(
    block_type: str | None,
    mime_type: str | None,
    default_kind: str | None,
    uri: str | None,
) -> str | None:
    clean_type = (block_type or "").replace("-", "_").lower()
    if clean_type in {"image", "input_image", "image_url"}:
        return "image"
    if clean_type in {"video", "input_video", "video_url"}:
        return "video"
    if clean_type in {"audio", "input_audio", "audio_url"}:
        return "audio"
    if clean_type in {"resource_link", "resource", "file", "attachment", "media"}:
        return default_kind or _kind_from_mime_type(mime_type) or _kind_from_mime_type(_guess_mime_type(uri)) or "resource_link"
    if default_kind is not None:
        return default_kind
    return _kind_from_mime_type(mime_type) or _kind_from_mime_type(_guess_mime_type(uri))


def _kind_from_mime_type(mime_type: str | None) -> str | None:
    clean = (mime_type or "").split(";", 1)[0].strip().lower()
    if clean.startswith("image/"):
        return "image"
    if clean.startswith("video/"):
        return "video"
    if clean.startswith("audio/"):
        return "audio"
    return None


def _default_mime_type(kind: str, uri: str | None) -> str:
    guessed = _guess_mime_type(uri)
    if guessed:
        return guessed
    if kind == "image":
        return "image/*"
    if kind == "video":
        return "video/*"
    if kind == "audio":
        return "audio/*"
    return "application/octet-stream"


def _guess_mime_type(uri: str | None) -> str | None:
    if not uri:
        return None
    parsed = urlparse(uri)
    path = unquote(parsed.path if parsed.scheme else uri)
    mime_type, _encoding = mimetypes.guess_type(path)
    return mime_type


def _parse_data_url(value: str) -> tuple[str | None, str | None]:
    match = _DATA_URL_RE.match(value.strip())
    if match is None:
        return None, None
    mime_type = match.group(1) or None
    data = match.group(2).strip()
    return mime_type, data or None


def _first_string(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _string(source.get(key))
        if value is not None:
            return value
    return None


def _nested_url(value: object) -> str | None:
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, dict):
        return _first_string(value, _DIRECT_URI_KEYS)
    return None


def _normalized_type(value: object) -> str | None:
    text = _string(value)
    return text.replace("-", "_").lower() if text is not None else None


def _is_remote_uri(value: str) -> bool:
    scheme = urlparse(value).scheme.lower()
    return scheme in {"http", "https"}
