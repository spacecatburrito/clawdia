import pytest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from clawdia.capture import process, extract, Deps
from clawdia.tagger import TagResult
from clawdia.storage import Store
from clawdia.schedule import WorkCalendar
from clawdia.context import ContextCache
from clawdia.second_client import SecondUnavailable, SecondRejected
from tests.conftest import OPEN_STORES


def msg(text="", caption=None, photo=None, voice=None, forward=None, mgid=None, mid=1, entities=None):
    return SimpleNamespace(text=text or None, caption=caption, photo=photo, document=None, voice=voice, audio=None,
        entities=entities or [], caption_entities=[], forward_origin=forward, media_group_id=mgid, message_id=mid,
        date=datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc), from_user=SimpleNamespace(id=1), chat=SimpleNamespace(id=1))


async def capture_only(text, **_kw):
    """Fake understanding call: every typed message is a capture without a model capture, so the
    phase 1 tagger (`tag_fn`) tags it and the reply is the deterministic one."""
    from clawdia.understand import Understanding, guess_language
    return Understanding(intent="capture", language=guess_language(text))


async def fallback_reply(facts, *, settings, client=None):
    """Fake reply call: the deterministic fallback."""
    return facts.fallback


def deps(tmp_path, tag_result, now=datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc), save_media=None, transcribe=None):
    from clawdia.config import Settings
    s = Store(tmp_path / "c.sqlite"); s.ensure_schema()
    OPEN_STORES.append(s)  # closed by the autouse fixture in conftest.py
    second = SimpleNamespace(post_task=AsyncMock(return_value={"id": 2601, "project": "Acme"}),
                             post_job=AsyncMock(return_value={"id": 77, "status": "queued"}))
    return Deps(store=s, settings=Settings(bot_token="t", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k", state_dir=tmp_path, brain="classic"),
                context_cache=ContextCache(tmp_path / "context.json"), second=second, calendar=WorkCalendar("mon-fri", "Europe/Lisbon"),
                bot=SimpleNamespace(), tag_fn=AsyncMock(return_value=tag_result), transcribe_fn=transcribe or AsyncMock(return_value=None),
                save_media_fn=save_media or AsyncMock(return_value=None), now_fn=lambda: now, group_wait_s=0,
                understand_fn=AsyncMock(side_effect=capture_only), reply_fn=AsyncMock(side_effect=fallback_reply))


@pytest.mark.asyncio
async def test_buy_item_saved_and_reply_has_keyboard(tmp_path):
    d = deps(tmp_path, TagResult("buy", "Olive oil", None, {}, 0.95, None, "claude-haiku-5-5"))
    c = await process(msg("купить оливковое масло"), d)
    assert c.kind == "buy" and c.reply_text.startswith(f"Saved #{c.item_id} · buy · Olive oil")
    assert c.keyboard == [[("Undo", f"undo:{c.item_id}")]] and d.store.get_item(c.item_id)["status"] == "new"
    assert d.store.get_item(c.item_id)["chat_id"] == 1


@pytest.mark.asyncio
async def test_task_routes_immediately_on_workday(tmp_path):
    d = deps(tmp_path, TagResult("task", "Ask Bruno about the invoice", "Acme", {}, 0.9, None, "claude-haiku-5-5"))
    c = await process(msg("напомнить Bruno про инвойс по Acme"), d)
    it = d.store.get_item(c.item_id)
    assert it["status"] == "filed" and it["filed_ref"] == "2601" and "-> Second #2601" in c.reply_text


@pytest.mark.asyncio
async def test_task_captured_on_saturday_is_posted_at_once(tmp_path):
    d = deps(tmp_path, TagResult("task", "x", None, {}, 0.9, None, "m"), now=datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc))
    c = await process(msg("сделать x"), d)
    assert d.store.get_item(c.item_id)["status"] == "filed" and "-> Second #2601" in c.reply_text and d.second.post_task.await_count == 1


@pytest.mark.asyncio
async def test_task_captured_before_0700_is_posted_at_once(tmp_path):
    d = deps(tmp_path, TagResult("task", "x", None, {}, 0.9, None, "m"), now=datetime(2026, 10, 29, 22, 0, tzinfo=timezone.utc))
    # 22:00 UTC Thursday = 22:00 Lisbon, past 07:00, so it routes now
    c = await process(msg("x"), d)
    assert d.store.get_item(c.item_id)["status"] == "filed"
    d2 = deps(tmp_path / "b", TagResult("task", "x", None, {}, 0.9, None, "m"), now=datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc))
    c2 = await process(msg("x"), d2)  # Tuesday 05:00 Lisbon, before 07:00
    assert d2.store.get_item(c2.item_id)["status"] == "filed" and "-> Second #2601" in c2.reply_text


@pytest.mark.asyncio
async def test_reply_has_no_deferred_second_line(tmp_path):
    from clawdia.capture import build_reply
    sunday = datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc)
    task = {"id": 7, "kind": "task", "title": "x", "status": "routing", "route_error": None, "fields": {}}
    for lang in ("en", "ru"):
        reply = build_reply(task, WorkCalendar("mon-fri", "Europe/Lisbon"), sunday, lang)
        assert "07:00" not in reply and "Monday" not in reply and "понедельник" not in reply
    d = deps(tmp_path, TagResult("task", "x", None, {}, 0.9, None, "m"), now=sunday)
    c = await process(msg("x"), d)
    assert "07:00" not in c.reply_text


@pytest.mark.asyncio
async def test_event_is_routed_to_second_and_job_is_not(tmp_path):
    d = deps(tmp_path, TagResult("event", "Dinner", None, {"date": "2026-11-12"}, 0.9, None, "m"),
             now=datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc))
    c = await process(msg("dinner on the 12th"), d)
    assert d.store.get_item(c.item_id)["status"] == "filed" and "-> Second #2601" in c.reply_text
    d2 = deps(tmp_path / "b", TagResult("job", "Role", None, {}, 0.9, None, "m"))
    c2 = await process(msg("https://example.com/job"), d2)
    assert d2.store.get_item(c2.item_id)["status"] == "routing" and d2.second.post_task.await_count == 0
    assert d2.second.post_job.await_count == 1 and "Sent for review" in c2.reply_text


@pytest.mark.asyncio
async def test_second_unreachable_and_rejected_reply_lines(tmp_path):
    d = deps(tmp_path, TagResult("task", "x", None, {}, 0.9, None, "m"))
    d.second.post_task = AsyncMock(side_effect=SecondUnavailable("down"))
    c = await process(msg("x"), d)
    assert d.store.get_item(c.item_id)["status"] == "routing" and "-> Second queued (Second unreachable)" in c.reply_text
    d2 = deps(tmp_path / "b", TagResult("task", "x", None, {}, 0.9, None, "m"))
    d2.second.post_task = AsyncMock(side_effect=SecondRejected("project_unknown"))
    c2 = await process(msg("x"), d2)
    assert d2.store.get_item(c2.item_id)["status"] == "new" and "-> Second rejected: project_unknown" in c2.reply_text


@pytest.mark.asyncio
async def test_other_filed_items_are_announced_through_notify(tmp_path):
    d = deps(tmp_path, TagResult("task", "new one", None, {}, 0.9, None, "m"))
    old = d.store.create_item(kind="task", status="routing", title="old", text="old")
    d.notify = AsyncMock()
    c = await process(msg("new one"), d)
    notified = [call.args[0] for call in d.notify.await_args_list]
    assert notified == [old]  # the item being captured is announced by the reply itself
    assert d.store.get_item(old)["status"] == "filed" and d.store.get_item(c.item_id)["status"] == "filed"


@pytest.mark.asyncio
async def test_voice_transcript_in_reply_and_forward_kept(tmp_path):
    fwd = SimpleNamespace(type="user", sender_user=SimpleNamespace(first_name="Bruno", last_name=None, username=None), date=None)
    d = deps(tmp_path, TagResult("task", "Invoice", "Acme", {}, 0.9, None, "m"),
             save_media=AsyncMock(return_value=(tmp_path / "v.ogg", "audio/ogg")), transcribe=AsyncMock(return_value="please send the invoice"))
    c = await process(msg(voice=SimpleNamespace(duration=12, file_id="f"), forward=fwd), d)
    it = d.store.get_item(c.item_id)
    assert it["transcript"] == "please send the invoice" and it["forward_from"]["name"] == "Bruno" and "please send the invoice" in c.reply_text
    assert d.transcribe_fn.await_args.args[1] == 12
    item_passed = d.tag_fn.await_args.args[0]
    assert item_passed["transcript"] == "please send the invoice" and d.tag_fn.await_args.kwargs["image_path"] is None


@pytest.mark.asyncio
async def test_image_passed_to_tagger_only_for_image_mime(tmp_path):
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"), save_media=AsyncMock(return_value=(tmp_path / "a.jpg", "image/jpeg")))
    await process(msg(photo=[SimpleNamespace(file_id="a")]), d)
    assert d.tag_fn.await_args.kwargs["image_path"] == tmp_path / "a.jpg"


def test_extract_hidden_user_forward_does_not_crash():
    fwd = SimpleNamespace(type="hidden_user", sender_user_name="Someone", date=None)
    assert extract(msg("hi", forward=fwd))["forward_from"]["name"] == "Someone"


def test_extract_channel_forward_links_only_public_channels():
    pub = SimpleNamespace(type="channel", chat=SimpleNamespace(title="News", username="newsch"), message_id=77, author_signature=None, date=None)
    priv = SimpleNamespace(type="channel", chat=SimpleNamespace(title="Secret", username=None), message_id=78, author_signature="Ann", date=None)
    assert extract(msg("x", forward=pub))["forward_from"] == {"name": "News", "chat_title": "News", "link": "https://t.me/newsch/77"}
    f = extract(msg("x", forward=priv))["forward_from"]
    assert f["link"] is None and f["chat_title"] == "Secret" and f["name"] == "Ann"


def test_extract_chat_origin_and_missing_attrs():
    chat = SimpleNamespace(type="chat", sender_chat=SimpleNamespace(title="Group"), author_signature=None, date=None)
    assert extract(msg("x", forward=chat))["forward_from"]["chat_title"] == "Group"
    assert extract(msg("x", forward=SimpleNamespace(type="channel")))["forward_from"]["link"] is None


def test_extract_urls_from_entities_and_text_with_utf16_offsets():
    text = "😀 see https://a.example/x and also http://b.example"
    ents = [SimpleNamespace(type="url", offset=7, length=19, url=None),
            SimpleNamespace(type="text_link", offset=0, length=2, url="https://c.example/l")]
    data = extract(msg(text, entities=ents))
    assert data["urls"] == ["https://a.example/x", "https://c.example/l", "http://b.example"]
    assert extract(msg("plain"))["urls"] == []


@pytest.mark.asyncio
async def test_media_group_makes_one_item(tmp_path):
    d = deps(tmp_path, TagResult("job", "Role", None, {}, 0.8, None, "m"), save_media=AsyncMock(side_effect=[(tmp_path / "a.jpg", "image/jpeg"), (tmp_path / "b.jpg", "image/jpeg")]))
    c1 = await process(msg(photo=[SimpleNamespace(file_id="a")], mgid="g1", mid=1), d)
    c2 = await process(msg(photo=[SimpleNamespace(file_id="b")], mgid="g1", mid=2), d)
    assert c1.item_id == c2.item_id and len(d.store.get_item(c1.item_id)["media_paths"]) == 2
    assert c2.reply_text == c1.reply_text and c2.duplicate and not c1.duplicate
    assert d.tag_fn.await_count == 1 and len(d.store.list_items()) == 1


@pytest.mark.asyncio
async def test_media_group_concurrent_updates_share_one_item(tmp_path):
    import asyncio
    d = deps(tmp_path, TagResult("job", "Role", None, {}, 0.8, None, "m"), save_media=AsyncMock(side_effect=[(tmp_path / "a.jpg", "image/jpeg"), (tmp_path / "b.jpg", "image/jpeg")]))
    d.group_wait_s = 0.05
    c1, c2 = await asyncio.gather(process(msg(photo=[SimpleNamespace(file_id="a")], mgid="g", mid=1), d),
                                  process(msg(photo=[SimpleNamespace(file_id="b")], mgid="g", mid=2), d))
    assert c1.item_id == c2.item_id and len(d.store.get_item(c1.item_id)["media_paths"]) == 2 and d.tag_fn.await_count == 1


@pytest.mark.asyncio
async def test_media_group_key_expires(tmp_path):
    d = deps(tmp_path, TagResult("job", "Role", None, {}, 0.8, None, "m"))
    c1 = await process(msg("a", mgid="g", mid=1), d)
    d.groups["g"].started -= 120
    c2 = await process(msg("b", mgid="g", mid=2), d)
    assert c1.item_id != c2.item_id


@pytest.mark.asyncio
async def test_model_failure_still_saves(tmp_path):
    d = deps(tmp_path, TagResult("unsorted", "something", None, {}, 0.0, None, "error"))
    c = await process(msg("something"), d)
    assert d.store.get_item(c.item_id)["kind"] == "unsorted" and "unsorted" in c.reply_text


@pytest.mark.asyncio
async def test_tagger_exception_still_returns_captured_with_error_event(tmp_path):
    d = deps(tmp_path, None)
    d.tag_fn = AsyncMock(side_effect=RuntimeError("boom secret text"))
    c = await process(msg("hello"), d)
    it = d.store.get_item(c.item_id)
    assert it["kind"] == "unsorted" and it["text"] == "hello" and c.reply_text.startswith(f"Saved #{c.item_id} · unsorted")
    events = d.store._conn.execute("SELECT type, payload FROM events WHERE item_id=?", (c.item_id,)).fetchall()
    assert any(e["type"] == "error" for e in events) and not any("secret" in (e["payload"] or "") for e in events)


@pytest.mark.asyncio
async def test_media_and_transcribe_failures_do_not_block_tagging(tmp_path):
    d = deps(tmp_path, TagResult("idea", "Idea", None, {}, 0.8, None, "m"),
             save_media=AsyncMock(return_value=(tmp_path / "v.ogg", "audio/ogg")), transcribe=AsyncMock(side_effect=RuntimeError("x")))
    c = await process(msg(voice=SimpleNamespace(duration=3, file_id="f")), d)
    assert c.kind == "idea"
    d2 = deps(tmp_path / "b", TagResult("idea", "Idea", None, {}, 0.8, None, "m"), save_media=AsyncMock(side_effect=OSError("disk")))
    c2 = await process(msg("hello", photo=[SimpleNamespace(file_id="a")]), d2)
    assert c2.kind == "idea" and d2.store.get_item(c2.item_id)["media_path"] is None
    assert "Media not saved (download failed), send again" in c2.reply_text


@pytest.mark.asyncio
async def test_invalid_kind_becomes_unsorted(tmp_path):
    d = deps(tmp_path, TagResult("bogus", "T", None, {}, 0.5, None, "m"))
    c = await process(msg("t"), d)
    assert c.kind == "unsorted" and d.store.get_item(c.item_id)["kind"] == "unsorted"


@pytest.mark.asyncio
async def test_reminder_and_fields_stored_and_fields_columns_updated(tmp_path):
    d = deps(tmp_path, TagResult("event", "Dinner", "Acme", {"place": "Lisbon"}, 0.8, "2026-11-02T19:00:00+00:00", "claude-haiku-5-5"))
    c = await process(msg("напомни про dinner monday"), d)
    it = d.store.get_item(c.item_id)
    assert it["fields"] == {"place": "Lisbon", "reminder_at": "2026-11-02T19:00:00+00:00",
                            "reminders": ["2026-11-02T19:00:00.000000+00:00"]}  # scheduled: now is 2026-10-29
    assert it["project"] == "Acme" and it["confidence"] == 0.8 and it["model"] == "claude-haiku-5-5" and it["status"] == "filed"
    assert it["tg_message_id"] == 1 and it["tg_date"].startswith("2026-10-30T10:00:00")


def _media_msg(**kw):
    m = msg("", **{k: v for k, v in kw.items() if k in ("photo", "voice")})
    for k in ("document", "video", "video_note", "animation", "sticker"):
        setattr(m, k, kw.get(k))
    return m


def _error_events(d, item_id):
    import json
    rows = d.store._conn.execute("SELECT payload FROM events WHERE item_id=? AND type='error'", (item_id,)).fetchall()
    return [json.loads(r["payload"]) for r in rows]


@pytest.mark.asyncio
async def test_download_failure_is_reported_and_item_kept(tmp_path):
    d = deps(tmp_path, TagResult("idea", "Pic", None, {"a": 1}, 0.8, None, "m"), save_media=AsyncMock(side_effect=OSError("disk")))
    c = await process(_media_msg(photo=[SimpleNamespace(file_id="a")]), d)
    assert c.reply_text.splitlines()[0].startswith(f"Saved #{c.item_id} · idea")
    assert "Media not saved (download failed), send again" in c.reply_text
    it = d.store.get_item(c.item_id)
    assert it["fields"] == {"a": 1, "media_error": "download failed"}
    assert {"step": "media", "type": "OSError"} in _error_events(d, c.item_id)


@pytest.mark.asyncio
async def test_oversize_document_is_reported(tmp_path):
    big = SimpleNamespace(file_id="d", file_name="a.zip", mime_type="application/zip", file_size=21 * 1024 * 1024)
    d = deps(tmp_path, TagResult("idea", "Zip", None, {}, 0.8, None, "m"))
    c = await process(_media_msg(document=big), d)
    assert "Media not saved (too large), send again" in c.reply_text
    assert {"step": "media", "type": "too_large"} in _error_events(d, c.item_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("attr", ["video", "video_note", "animation", "sticker"])
async def test_unsupported_media_type_is_reported(tmp_path, attr):
    d = deps(tmp_path, TagResult("idea", "Clip", None, {}, 0.8, None, "m"))
    c = await process(_media_msg(**{attr: SimpleNamespace(file_id="x")}), d)
    assert "Media not saved (unsupported type), send again" in c.reply_text
    assert d.store.get_item(c.item_id)["fields"]["media_error"] == "unsupported type"


@pytest.mark.asyncio
async def test_plain_text_and_saved_media_have_no_media_warning(tmp_path):
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"), save_media=AsyncMock(return_value=(tmp_path / "a.jpg", "image/jpeg")))
    assert "Media not saved" not in (await process(msg("hello"), d)).reply_text
    assert "Media not saved" not in (await process(_media_msg(photo=[SimpleNamespace(file_id="a")]), d)).reply_text


# -- forward sender in the reply ------------------------------------------------

@pytest.mark.asyncio
async def test_reply_shows_the_forward_sender(tmp_path):
    user = SimpleNamespace(type="user", sender_user=SimpleNamespace(first_name="Bruno", last_name="K", username=None), date=None)
    d = deps(tmp_path, TagResult("idea", "Pricing idea", None, {}, 0.8, None, "m"))
    c = await process(msg("pricing thought", forward=user), d)
    assert c.reply_text.splitlines() == [f"Saved #{c.item_id} · idea · Pricing idea", "From: Bruno K"]
    assert d.store.get_item(c.item_id)["forward_from"]["name"] == "Bruno K"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin,line", [
    (SimpleNamespace(type="channel", chat=SimpleNamespace(title="News", username="newsch"), message_id=7, author_signature="Ann", date=None),
     "From: Ann (News)"),
    (SimpleNamespace(type="channel", chat=SimpleNamespace(title="News", username=None), message_id=7, author_signature=None, date=None),
     "From: News"),  # no signature: the name is the chat title, shown once
    (SimpleNamespace(type="hidden_user", sender_user_name="Someone", date=None), "From: Someone"),
    (SimpleNamespace(type="chat", sender_chat=SimpleNamespace(title="Team"), author_signature=None, date=None), "From: Team"),
])
async def test_reply_from_line_variants(tmp_path, origin, line):
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    c = await process(msg("x", forward=origin), d)
    assert c.reply_text.splitlines()[1] == line


@pytest.mark.asyncio
async def test_reply_has_no_from_line_without_a_named_forward(tmp_path):
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    c = await process(msg("x"), d)
    c2 = await process(msg("y", forward=SimpleNamespace(type="channel")), d)  # nameless origin
    assert not any(l.startswith("From:") for l in c.reply_text.splitlines() + c2.reply_text.splitlines())


@pytest.mark.asyncio
async def test_voice_task_reaches_second_with_its_transcript(tmp_path):
    fwd = SimpleNamespace(type="user", sender_user=SimpleNamespace(first_name="Bruno", last_name=None, username=None), date=None)
    d = deps(tmp_path, TagResult("task", "Invoice", "Acme", {}, 0.9, None, "m"),
             save_media=AsyncMock(return_value=(tmp_path / "v.ogg", "audio/ogg")), transcribe=AsyncMock(return_value="please send the invoice"))
    from clawdia.second_client import source_text
    c = await process(msg(voice=SimpleNamespace(duration=12, file_id="f"), forward=fwd), d)
    posted = d.second.post_task.await_args.args[0]
    assert source_text(posted) == "Transcript: please send the invoice\nForwarded from: Bruno"
    assert "From: Bruno" in c.reply_text.splitlines()


def test_build_reply_in_russian():
    from clawdia.capture import build_reply
    cal = WorkCalendar("mon-fri", "Europe/Lisbon")
    sat = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
    task = {"id": 7, "kind": "task", "title": "Счёт", "status": "routing", "route_error": None, "fields": {}}
    assert build_reply(task, cal, sat, "ru") == "Сохранила #7 · задача · Счёт\n-> Second в очереди"
    assert build_reply(task, cal, sat) == "Saved #7 · task · Счёт\n-> Second queued"
    filed = {**task, "status": "filed", "filed_ref": "2601"}
    assert build_reply(filed, cal, sat, "ru", lead="Обновила") == "Обновила #7 · задача · Счёт\n-> Second #2601"
    down = {**task, "route_error": "unavailable"}
    assert build_reply(down, cal, sat, "ru").endswith("-> Second в очереди (Second недоступен)")
    event = {"id": 8, "kind": "event", "title": "Концерт", "status": "new",
             "fields": {"reminders": ["2026-11-06T19:00:00+00:00"]}}
    assert build_reply(event, cal, sat, "ru") == "Сохранила #8 · событие · Концерт\nНапоминание: пт 19:00"
    asks = {"id": 9, "kind": "unsorted", "title": None, "status": "new", "fields": {"ask_when": True}}
    assert build_reply(asks, cal, sat, "ru") == "Сохранила #9 · без категории · (без названия)\nКогда напомнить?"
    off = {**event, "fields": {"reminders_off": True}}
    assert build_reply(off, cal, sat, "ru").endswith("Напоминания выключены")
    from clawdia.capture import KIND_RU
    from clawdia.kinds import KINDS
    assert set(KIND_RU) == set(KINDS)
