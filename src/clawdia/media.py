"""Download Telegram media (photo, document, voice, audio) into the media directory."""

from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


def _ext(file_name: str | None, mime: str | None, default: str) -> str:
    """A safe extension: from the file name, else the mime type, else `default`."""
    suffix = Path(file_name).suffix if file_name else ""
    if suffix and _EXT_RE.match(suffix):
        return suffix.lower()
    guessed = mimetypes.guess_extension(mime) if mime else None
    if guessed and _EXT_RE.match(guessed):
        return guessed
    return default


def _pick(message: Any, max_document_mb: int) -> tuple[str, str, str, str] | None:
    """Return (file_id, file_unique_id, extension, mime) for the message media, or None."""
    photo = getattr(message, "photo", None)
    if photo:
        best = max(photo, key=lambda p: (getattr(p, "width", 0) or 0) * (getattr(p, "height", 0) or 0)
                   or getattr(p, "file_size", 0) or 0)
        return best.file_id, getattr(best, "file_unique_id", None) or best.file_id, ".jpg", "image/jpeg"
    document = getattr(message, "document", None)
    if document:
        size = getattr(document, "file_size", None)
        if size is not None and size > max_document_mb * 1024 * 1024:
            return None
        mime = getattr(document, "mime_type", None) or "application/octet-stream"
        ext = _ext(getattr(document, "file_name", None), mime, ".bin")
        return document.file_id, getattr(document, "file_unique_id", None) or document.file_id, ext, mime
    voice = getattr(message, "voice", None)
    if voice:
        mime = getattr(voice, "mime_type", None) or "audio/ogg"
        return voice.file_id, getattr(voice, "file_unique_id", None) or voice.file_id, ".ogg", mime
    audio = getattr(message, "audio", None)
    if audio:
        mime = getattr(audio, "mime_type", None) or "audio/mpeg"
        ext = _ext(getattr(audio, "file_name", None), mime, ".mp3")
        return audio.file_id, getattr(audio, "file_unique_id", None) or audio.file_id, ext, mime
    return None


UNSUPPORTED_ATTRS = ("video", "video_note", "animation", "sticker")


def skip_reason(message: Any, max_document_mb: int) -> str | None:
    """Why `save` returns None for a message that does carry media; None when there is nothing to report."""
    if _pick(message, max_document_mb) is not None:
        return None
    if getattr(message, "document", None):
        return "too large"
    if any(getattr(message, attr, None) for attr in UNSUPPORTED_ATTRS):
        return "unsupported type"
    return None


async def save(bot: Any, message: Any, media_dir: Path, max_document_mb: int,
               stem: str | None = None) -> tuple[Path, str] | None:
    """Download the message media to `media_dir/<stem><ext>`. None when there is none or it is too big."""
    picked = _pick(message, max_document_mb)
    if picked is None:
        return None
    file_id, unique_id, ext, mime = picked
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    media_dir.chmod(0o700)
    dest = media_dir / f"{stem or unique_id}{ext}"
    await bot.download(file_id, destination=dest)
    return dest, mime
