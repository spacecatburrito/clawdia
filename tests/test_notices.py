import pytest
from unittest.mock import AsyncMock

from clawdia.notices import CURSOR_KEY, send_resolution_notices
from clawdia.cli import main
from tests.test_loops import MON_9, mk


def sent(d):
    return [(c.args[0], c.kwargs["text"]) for c in d.bot.send_message.await_args_list]


def item(d, title="Scrapely scraper", chat_id=1, status="new", **kw):
    return d.store.create_item(kind="idea", status=status, title=title, chat_id=chat_id, **kw)


async def start(d):
    """First run: the cursor is set to the current max event id."""
    await send_resolution_notices(d)
    assert d.store.meta_get(CURSOR_KEY) is not None


@pytest.mark.asyncio
async def test_cursor_init_sends_nothing_for_old_events(tmp_path):
    d = mk(tmp_path)
    i = item(d)
    d.store.add_event(i, "dropped", {"via": "cli"})
    await send_resolution_notices(d)
    assert sent(d) == []
    await send_resolution_notices(d)
    assert sent(d) == []


@pytest.mark.asyncio
async def test_notice_after_cli_dismiss_with_note(tmp_path):
    d = mk(tmp_path)
    await start(d)
    i = item(d, title="Quarterly mood board")
    d.store.add_event(i, "dropped", {"via": "cli", "note": "дубль #12"})
    await send_resolution_notices(d)
    assert sent(d) == [(1, f"#{i} · Quarterly mood board · отклонено · дубль #12")]
    assert d.store.recent_turns(1)[-1]["item_id"] == i and d.store.recent_turns(1)[-1]["role"] == "bot"
    await send_resolution_notices(d)
    assert len(sent(d)) == 1  # never twice


@pytest.mark.asyncio
async def test_notice_after_cli_file_with_link_and_second_id(tmp_path):
    d = mk(tmp_path)
    await start(d)
    a, b, c = item(d, title="Job A"), item(d, title="Job B"), item(d, title="Job C")
    d.store.add_event(a, "filed", {"ref": "https://notion.so/x", "via": "cli", "note": "applied"})
    d.store.add_event(b, "filed", {"ref": "Second #2753", "via": "cli", "note": None})
    d.store.add_event(c, "filed", {"ref": "some local note", "via": "cli"})
    await send_resolution_notices(d)
    assert [t for _, t in sent(d)] == [
        f"#{a} · Job A · в работе · applied · https://notion.so/x",
        f"#{b} · Job B · в работе · Second #2753",
        f"#{c} · Job C · в работе",
    ]


@pytest.mark.asyncio
async def test_only_cli_filed_and_dropped_events_count(tmp_path):
    d = mk(tmp_path)
    await start(d)
    i = item(d)
    d.store.add_event(i, "dropped", {})                      # in-bot /drop
    d.store.add_event(i, "undone", {})                       # Undo button
    d.store.add_event(i, "filed", {"second_id": 5})          # background drain
    d.store.add_event(i, "done", {"via": "cli"})             # cli done: out of scope
    await send_resolution_notices(d)
    assert sent(d) == []


@pytest.mark.asyncio
async def test_failed_send_is_retried_then_sent_once(tmp_path):
    d = mk(tmp_path)
    await start(d)
    i = item(d)
    d.store.add_event(i, "dropped", {"via": "cli", "note": "n"})
    d.bot.send_message = AsyncMock(side_effect=[RuntimeError("down"), None])
    await send_resolution_notices(d)
    assert d.store.recent_turns(1) == []
    await send_resolution_notices(d)
    await send_resolution_notices(d)
    assert d.bot.send_message.await_count == 2 and len(d.store.recent_turns(1)) == 1


@pytest.mark.asyncio
async def test_goes_to_item_chat_and_falls_back_to_owner(tmp_path):
    d = mk(tmp_path)
    await start(d)
    other = item(d, title="Other chat", chat_id=777)
    nochat = item(d, title="No chat", chat_id=None)
    d.store.add_event(other, "dropped", {"via": "cli"})
    d.store.add_event(nochat, "dropped", {"via": "cli"})
    await send_resolution_notices(d)
    assert [chat for chat, _ in sent(d)] == [777, d.settings.owner_id]


@pytest.mark.asyncio
async def test_english_chat_gets_english_words(tmp_path):
    d = mk(tmp_path)
    await start(d)
    d.store.add_turn(1, "user", "please remind me to call the bank tomorrow morning")
    i, j = item(d, title="A"), item(d, title="B")
    d.store.add_event(i, "dropped", {"via": "cli", "note": "dup"})
    d.store.add_event(j, "filed", {"ref": "Second #9", "via": "cli"})
    await send_resolution_notices(d)
    assert [t for _, t in sent(d)] == [f"#{i} · A · declined · dup", f"#{j} · B · in progress · Second #9"]


@pytest.mark.asyncio
async def test_long_title_and_multiline_note_are_flattened(tmp_path):
    d = mk(tmp_path)
    await start(d)
    i = item(d, title="T" * 300)
    d.store.add_event(i, "dropped", {"via": "cli", "note": "line one\nline two"})
    await send_resolution_notices(d)
    text = sent(d)[0][1]
    assert "\n" not in text and "line one line two" in text and len(text) < 200 and chr(0x2014) not in text


def test_cli_events_carry_only_the_note_added_by_that_command(tmp_path, monkeypatch, capsys):
    from clawdia.storage import Store
    monkeypatch.setenv("CLAWDIA_STATE_DIR", str(tmp_path))
    s = Store(tmp_path / "clawdia.sqlite"); s.ensure_schema()
    a = s.create_item(kind="job", status="new", title="x", notes="old note")
    b = s.create_item(kind="job", status="new", title="y")
    s.close()
    assert main(["dismiss", str(a), "--note", "fresh"]) == 0
    assert main(["file", str(b), "--ref", "https://x.y", "--note", "applied"]) == 0
    s = Store(tmp_path / "clawdia.sqlite"); s.ensure_schema()
    rows = s.events_after(0, ("filed", "dropped"))
    s.close()
    assert [(r["type"], r["payload"].get("note"), r["payload"].get("via")) for r in rows] == [
        ("dropped", "fresh", "cli"), ("filed", "applied", "cli")]


@pytest.mark.asyncio
async def test_tick_runs_the_notices_step(tmp_path):
    from clawdia.loops import tick
    d = mk(tmp_path)
    await tick(d, MON_9)
    i = item(d)
    d.store.add_event(i, "dropped", {"via": "cli", "note": "n"})
    await tick(d, MON_9)
    assert any("отклонено" in t for _, t in sent(d))


@pytest.mark.asyncio
async def test_permanent_telegram_error_skips_the_event_but_network_error_retries(tmp_path):
    from aiogram.exceptions import TelegramForbiddenError
    from aiogram.methods import SendMessage
    d = mk(tmp_path)
    await start(d)
    a, b = item(d, title="A", chat_id=5), item(d, title="B")
    d.store.add_event(a, "dropped", {"via": "cli"})
    d.store.add_event(b, "dropped", {"via": "cli"})
    forbidden = TelegramForbiddenError(method=SendMessage(chat_id=5, text="x"), message="bot was blocked")
    d.bot.send_message = AsyncMock(side_effect=[forbidden, None])
    await send_resolution_notices(d)  # the blocked chat is skipped, the next notice still goes out
    assert d.bot.send_message.await_count == 2 and len(d.store.recent_turns(1)) == 1
    await send_resolution_notices(d)
    assert d.bot.send_message.await_count == 2  # nothing is retried
