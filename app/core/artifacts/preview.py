from __future__ import annotations

import mimetypes
from pathlib import Path


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type:
        return mime_type
    suffix = path.suffix.lower()
    if suffix == ".drawio":
        return "application/vnd.jgraph.mxfile"
    if suffix == ".xmind":
        return "application/vnd.xmind.workbook"
    return "application/octet-stream"


def artifact_kind(path: Path, mime_type: str) -> str:
    suffix = path.suffix.lower()
    if mime_type.startswith("image/"):
        return "image"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".txt", ".json", ".csv", ".xml", ".drawio"}:
        return "text"
    if suffix in {".xlsx", ".xls"}:
        return "spreadsheet"
    if suffix in {".pptx", ".ppt"}:
        return "presentation"
    if suffix == ".xmind":
        return "mindmap"
    return "file"

