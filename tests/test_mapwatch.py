from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawdia.mapwatch import check_map
from clawdia.storage import Store
from clawdia.tagger import TagResult

NOW = datetime(2026, 11, 21, 9, 0, tzinfo=timezone.utc)  # Saturday, Lisbon = UTC in November
MAP = """Built automatically on the laptop.

## 🔴 Decisions for you
- Energy set to normal <b>&

## 🎯 Top-4 (4 open)

1. **One**
2. **Two**
3. **Three**
4. **Four**
5. **Five**
"""


def mk(tmp_path, now=NOW):
    from tests.test_capture import deps as mkdeps

    d = mkdeps(tmp_path, TagResult("idea", "x", None, {}, 0.9, None, "m"), now=now)
    d.bot = SimpleNamespace(send_message=AsyncMock())
    d.settings.focus_dir = tmp_path / "focus"
    d.settings.focus_dir.mkdir()
    return d


def put_map(d, day="2026-11-21", text=MAP):
    (d.settings.focus_dir / f"{day}.md").write_text(text, encoding="utf-8")


async def test_nothing_without_map_or_requests(tmp_path):
    d = mk(tmp_path)
    await check_map(d, NOW)
    d.bot.send_message.assert_not_awaited()


async def test_ready_sent_once_across_ticks_and_restart(tmp_path):
    d = mk(tmp_path)
    put_map(d)
    await check_map(d, NOW)
    await check_map(d, NOW + timedelta(minutes=1))
    assert d.bot.send_message.await_count == 1
    chat, = d.bot.send_message.await_args.args
    text = d.bot.send_message.await_args.kwargs["text"]
    assert chat == 1 and ("Day map ready" in text or "Карта дня готова" in text)
    assert "One" in text and "Four" in text and "Five" not in text
    assert "🔴 Energy set to normal <b>&" in text  # plain text: no parse mode, nothing to escape
    assert "parse_mode" not in d.bot.send_message.await_args.kwargs
    assert d.store.recent_turns(1, 5)[-1]["source"] == "agent_data"
    # restart: a fresh Store on the same database
    d.store.close()
    d.store = Store(tmp_path / "c.sqlite")
    await check_map(d, NOW + timedelta(minutes=2))
    assert d.bot.send_message.await_count == 1


async def test_yesterdays_map_is_not_announced(tmp_path):
    d = mk(tmp_path)
    put_map(d, "2026-11-20")
    await check_map(d, NOW)
    d.bot.send_message.assert_not_awaited()


async def test_failed_send_is_retried_next_tick(tmp_path):
    d = mk(tmp_path)
    put_map(d)
    d.bot.send_message.side_effect = [RuntimeError("down"), None]
    with pytest.raises(RuntimeError):  # the loop step catches it; the guard stays unset
        await check_map(d, NOW)
    await check_map(d, NOW + timedelta(minutes=1))
    assert d.bot.send_message.await_count == 2


async def test_failure_report_sent_once(tmp_path):
    d = mk(tmp_path)
    d.store.focus_report("failed", "claude exited 1", NOW)
    await check_map(d, NOW)
    await check_map(d, NOW + timedelta(minutes=1))
    assert d.bot.send_message.await_count == 1
    assert "claude exited 1" in d.bot.send_message.await_args.kwargs["text"]
    d.store.focus_report("failed", "second failure", NOW + timedelta(minutes=5))
    await check_map(d, NOW + timedelta(minutes=6))
    assert d.bot.send_message.await_count == 2


async def test_ok_report_sends_nothing(tmp_path):
    d = mk(tmp_path)
    d.store.focus_report("ok", "", NOW)
    await check_map(d, NOW)
    d.bot.send_message.assert_not_awaited()


async def test_one_reminder_after_an_hour_without_a_map(tmp_path):
    d = mk(tmp_path)
    d.store.focus_request_open(NOW)
    await check_map(d, NOW + timedelta(minutes=59))
    d.bot.send_message.assert_not_awaited()
    await check_map(d, NOW + timedelta(minutes=61))
    await check_map(d, NOW + timedelta(minutes=90))
    assert d.bot.send_message.await_count == 1
    # still one reminder after the laptop took the request
    d.store.focus_request_take()
    await check_map(d, NOW + timedelta(minutes=120))
    assert d.bot.send_message.await_count == 1


async def test_no_reminder_when_map_exists(tmp_path):
    d = mk(tmp_path)
    d.store.focus_request_open(NOW)
    put_map(d)
    await check_map(d, NOW + timedelta(minutes=61))
    assert d.bot.send_message.await_count == 1  # only the ready notice
    assert "reminder" not in d.bot.send_message.await_args.kwargs["text"].lower()


async def test_fulfilled_request_does_not_remind_next_day(tmp_path):
    t18 = datetime(2026, 11, 21, 18, 0, tzinfo=timezone.utc)
    d = mk(tmp_path, now=t18)
    d.store.focus_request_open(t18)
    put_map(d)
    await check_map(d, t18 + timedelta(minutes=10))
    assert d.bot.send_message.await_count == 1  # the ready notice
    await check_map(d, datetime(2026, 11, 22, 0, 30, tzinfo=timezone.utc))
    assert d.bot.send_message.await_count == 1


async def test_stale_request_from_yesterday_does_not_remind(tmp_path):
    d = mk(tmp_path)
    d.store.focus_request_open(NOW - timedelta(days=1))
    await check_map(d, NOW)
    d.bot.send_message.assert_not_awaited()


async def test_map_clears_a_pending_request(tmp_path):
    d = mk(tmp_path)
    d.store.focus_request_open(NOW)
    put_map(d)
    await check_map(d, NOW + timedelta(minutes=5))
    assert not d.store.meta_get("focus_request")  # the laptop must not take it and rebuild
    assert d.store.focus_request_take() is None


async def test_pending_request_stays_without_a_map(tmp_path):
    d = mk(tmp_path)
    d.store.focus_request_open(NOW)
    await check_map(d, NOW + timedelta(minutes=5))
    assert d.store.meta_get("focus_request")
