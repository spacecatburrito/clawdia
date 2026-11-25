"""Voice and audio transcription through OpenAI (whisper-1)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from clawdia.config import Settings

log = logging.getLogger(__name__)

MODEL = "whisper-1"
REQUEST_TIMEOUT_S = 30.0


async def transcribe(path: Path, duration_s: int | None, settings: Settings, client: Any = None) -> str | None:
    """Return the transcript, or None when the audio is too long, empty or any step fails."""
    if duration_s is not None and duration_s > settings.max_audio_seconds:
        return None
    own_client = client is None
    try:
        if own_client:
            import openai
            client = openai.AsyncOpenAI(api_key=settings.openai_api_key, timeout=REQUEST_TIMEOUT_S, max_retries=0)
        try:
            with open(path, "rb") as fh:
                result = await client.audio.transcriptions.create(model=MODEL, file=fh)
        finally:
            if own_client:
                await client.close()
        text = (getattr(result, "text", None) or "").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 - transcription is best effort, the item is already saved
        log.warning("transcription failed: %s", type(exc).__name__)
        return None
