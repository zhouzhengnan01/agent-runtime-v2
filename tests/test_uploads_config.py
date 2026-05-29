from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def test_default_upload_limit_allows_large_training_dataset(monkeypatch) -> None:
    module = _load_uploads_module_without_fastapi()
    monkeypatch.delenv("JETLINKS_MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("JETLINKS_MAX_UPLOAD_MB", raising=False)
    monkeypatch.setattr(module, "MAX_UPLOAD_BYTES", None)

    assert module._max_upload_bytes() == 4096 * 1024 * 1024
    assert module._max_upload_bytes() > int(3342.9 * 1024 * 1024)


def test_upload_limit_can_be_overridden_by_megabytes(monkeypatch) -> None:
    module = _load_uploads_module_without_fastapi()
    monkeypatch.delenv("JETLINKS_MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.setenv("JETLINKS_MAX_UPLOAD_MB", "8192")
    monkeypatch.setattr(module, "MAX_UPLOAD_BYTES", None)

    assert module._max_upload_bytes() == 8192 * 1024 * 1024


def test_upload_limit_can_be_overridden_by_bytes(monkeypatch) -> None:
    module = _load_uploads_module_without_fastapi()
    monkeypatch.setenv("JETLINKS_MAX_UPLOAD_BYTES", "123456789")
    monkeypatch.setenv("JETLINKS_MAX_UPLOAD_MB", "8192")
    monkeypatch.setattr(module, "MAX_UPLOAD_BYTES", None)

    assert module._max_upload_bytes() == 123456789


def _load_uploads_module_without_fastapi() -> object:
    project_root = Path(__file__).resolve().parents[1]
    _install_uploads_import_stubs()
    path = project_root / "app" / "api" / "uploads.py"
    spec = importlib.util.spec_from_file_location("uploads_config_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_uploads_import_stubs() -> None:
    fastapi = types.ModuleType("fastapi")
    fastapi.APIRouter = _Router
    fastapi.Depends = lambda dependency=None, **kwargs: dependency
    fastapi.File = lambda default=None, **kwargs: default
    fastapi.Form = lambda default=None, **kwargs: default
    fastapi.HTTPException = type("HTTPException", (Exception,), {})
    fastapi.UploadFile = type("UploadFile", (), {})
    sys.modules["fastapi"] = fastapi

    auth = types.ModuleType("app.api.auth")
    auth.require_runtime_token = lambda: None
    sys.modules["app.api.auth"] = auth

    artifacts = types.ModuleType("app.core.artifacts")
    artifacts.ArtifactStore = lambda *args, **kwargs: object()
    sys.modules["app.core.artifacts"] = artifacts

    preview = types.ModuleType("app.core.artifacts.preview")
    preview.guess_mime_type = lambda path: "application/octet-stream"
    sys.modules["app.core.artifacts.preview"] = preview


class _Router:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def post(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator
