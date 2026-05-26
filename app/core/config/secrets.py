from __future__ import annotations

import base64
import hashlib
import hmac
import os
from pathlib import Path
from typing import Protocol, cast


SECRET_ENV = "JETLINKS_AGENT_SECRET_KEY"


class FernetLike(Protocol):
    @staticmethod
    def generate_key() -> bytes: ...

    def __init__(self, key: bytes) -> None: ...

    def encrypt(self, data: bytes) -> bytes: ...

    def decrypt(self, token: bytes) -> bytes: ...


class SecretCodec:
    """Encrypt local secrets stored in ignored agent override files."""

    def __init__(self, root_dir: Path | None = None, env_name: str = SECRET_ENV) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.root_dir = root_dir or project_root
        self.env_name = env_name
        self.key_path = self.root_dir / ".runtime" / "secrets" / "master.key"

    def encrypt(self, plaintext: str, *, purpose: str) -> str:
        clean_text = plaintext.strip()
        if not clean_text:
            raise ValueError("secret plaintext is required")
        payload = purpose.encode("utf-8") + b"\0" + clean_text.encode("utf-8")
        encrypted = self._fernet().encrypt(payload).decode("ascii")
        return "enc.fernet.v1." + encrypted

    def decrypt(self, token: str, *, purpose: str) -> str:
        if not token.startswith("enc.fernet.v1."):
            raise ValueError("encrypted secret must start with enc.fernet.v1.")
        raw_token = token.removeprefix("enc.fernet.v1.").encode("ascii")
        invalid_token_error = self._invalid_token_error()
        try:
            payload = self._fernet().decrypt(raw_token)
        except invalid_token_error as exc:
            raise ValueError("encrypted secret authentication failed") from exc
        raw_purpose, separator, raw_plaintext = payload.partition(b"\0")
        if separator != b"\0" or not hmac.compare_digest(raw_purpose, purpose.encode("utf-8")):
            raise ValueError("encrypted secret purpose mismatch")
        return raw_plaintext.decode("utf-8")

    def get_or_create_master_key(self) -> str:
        raw_env = os.getenv(self.env_name)
        if raw_env and raw_env.strip():
            return raw_env.strip()
        if self.key_path.is_file():
            return self.key_path.read_text(encoding="utf-8").strip()
        fernet = self._fernet_class()
        key = fernet.generate_key().decode("ascii")
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_text(key + "\n", encoding="utf-8")
        self.key_path.chmod(0o600)
        return key

    def _fernet(self) -> FernetLike:
        return self._fernet_class()(self._fernet_key())

    def _fernet_key(self) -> bytes:
        raw_key = self.get_or_create_master_key()
        fernet = self._fernet_class()
        try:
            key_bytes = raw_key.encode("ascii")
            fernet(key_bytes)
            return key_bytes
        except (ValueError, TypeError):
            digest = hashlib.sha256(raw_key.encode("utf-8")).digest()
            return base64.urlsafe_b64encode(digest)

    @staticmethod
    def _fernet_class() -> type[FernetLike]:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise RuntimeError(
                "Encrypted secrets require the cryptography package. "
                "Install dependencies with uv sync or run through the project .venv."
            ) from exc
        return cast("type[FernetLike]", Fernet)

    @staticmethod
    def _invalid_token_error() -> type[Exception]:
        try:
            from cryptography.fernet import InvalidToken
        except ImportError:
            return ValueError
        return InvalidToken
