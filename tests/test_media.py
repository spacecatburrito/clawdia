import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from clawdia.media import save, skip_reason


def m(**kw):
    base = dict(photo=None, document=None, voice=None, audio=None)
    base.update(kw)
    return SimpleNamespace(**base)


def bot_with_download():
    async def download(file_id, destination):
        Path(destination).write_bytes(b"x")
    return SimpleNamespace(download=AsyncMock(side_effect=download))


@pytest.mark.asyncio
async def test_photo_uses_largest_size(tmp_path):
    bot = bot_with_download()
    photo = [SimpleNamespace(file_id="s", width=90, height=90, file_size=1), SimpleNamespace(file_id="l", width=1280, height=960, file_size=9)]
    path, mime = await save(bot, m(photo=photo), tmp_path / "media", 20, stem="7")
    assert path == tmp_path / "media" / "7.jpg" and mime == "image/jpeg" and path.exists()
    assert bot.download.await_args.args[0] == "l"
    assert (tmp_path / "media").stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_voice_is_ogg_and_document_keeps_extension(tmp_path):
    bot = bot_with_download()
    path, mime = await save(bot, m(voice=SimpleNamespace(file_id="v", mime_type="audio/ogg", file_size=10)), tmp_path, 20, stem="1")
    assert path.name == "1.ogg" and mime == "audio/ogg"
    doc = SimpleNamespace(file_id="d", file_name="CV final.PDF", mime_type="application/pdf", file_size=1000)
    path, mime = await save(bot, m(document=doc), tmp_path, 20, stem="2")
    assert path.name == "2.pdf" and mime == "application/pdf"


@pytest.mark.asyncio
async def test_document_ext_cannot_escape_media_dir(tmp_path):
    bot = bot_with_download()
    doc = SimpleNamespace(file_id="d", file_name="x./../../evil", mime_type=None, file_size=1)
    path, _ = await save(bot, m(document=doc), tmp_path / "media", 20, stem="3")
    assert path.parent == tmp_path / "media"


@pytest.mark.asyncio
async def test_oversize_document_and_no_media_return_none(tmp_path):
    bot = bot_with_download()
    big = SimpleNamespace(file_id="d", file_name="a.zip", mime_type="application/zip", file_size=21 * 1024 * 1024)
    assert await save(bot, m(document=big), tmp_path, 20, stem="1") is None
    assert await save(bot, m(), tmp_path, 20, stem="1") is None
    bot.download.assert_not_awaited()


def test_skip_reason():
    big = SimpleNamespace(file_id="d", file_name="a.zip", mime_type="application/zip", file_size=21 * 1024 * 1024)
    small = SimpleNamespace(file_id="d", file_name="a.zip", mime_type="application/zip", file_size=5)
    assert skip_reason(m(document=big), 20) == "too large"
    assert skip_reason(m(document=small), 20) is None
    assert skip_reason(m(), 20) is None
    for attr in ("video", "video_note", "animation", "sticker"):
        assert skip_reason(m(**{attr: SimpleNamespace(file_id="x")}), 20) == "unsupported type"
