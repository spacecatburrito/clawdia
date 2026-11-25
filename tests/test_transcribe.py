import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from clawdia.config import Settings
from clawdia.transcribe import transcribe


def settings(tmp_path):
    return Settings(bot_token="t", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k", state_dir=tmp_path)


def fake_client(result=None, error=None):
    create = AsyncMock(side_effect=error) if error else AsyncMock(return_value=result)
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


@pytest.mark.asyncio
async def test_transcribes_with_whisper(tmp_path):
    f = tmp_path / "a.ogg"; f.write_bytes(b"x")
    c = fake_client(SimpleNamespace(text=" hello there "))
    assert await transcribe(f, 10, settings(tmp_path), client=c) == "hello there"
    assert c.audio.transcriptions.create.await_args.kwargs["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_too_long_or_error_or_empty_returns_none(tmp_path):
    f = tmp_path / "a.ogg"; f.write_bytes(b"x")
    c = fake_client(SimpleNamespace(text="hi"))
    assert await transcribe(f, 601, settings(tmp_path), client=c) is None
    c.audio.transcriptions.create.assert_not_awaited()
    assert await transcribe(f, 5, settings(tmp_path), client=fake_client(error=RuntimeError("x"))) is None
    assert await transcribe(f, 5, settings(tmp_path), client=fake_client(SimpleNamespace(text="  "))) is None
    assert await transcribe(tmp_path / "missing.ogg", 5, settings(tmp_path), client=c) is None


@pytest.mark.asyncio
async def test_own_client_uses_30s_timeout_and_no_retries(tmp_path, monkeypatch):
    import openai
    seen = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            seen.update(kw)
            self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(text="ok"))))

        async def close(self):
            seen["closed"] = True

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAI)
    f = tmp_path / "a.ogg"; f.write_bytes(b"x")
    assert await transcribe(f, 5, settings(tmp_path)) == "ok"
    assert seen["timeout"] == 30.0 and seen["max_retries"] == 0 and seen["closed"]
