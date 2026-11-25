from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import MessageOriginHiddenUser

from clawdia.bot import on_rdone, on_rsnooze
from clawdia.tagger import TagResult
from tests.test_capture import deps
from tests.test_commands import NOW, ent, feed  # NOW = Fri 2026-10-30 12:00Z (Lisbon is UTC+0)


def make(tmp_path):
    return deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"), now=NOW)


def asking_item(d, title="Time Out Market"):
    return d.store.create_item(kind="place", status="new", title=title, text=f"напомни {title}", fields={"ask_when": True})


def cb(data):
    msg = SimpleNamespace(edit_text=AsyncMock(), answer=AsyncMock())
    return SimpleNamespace(data=data, message=msg, answer=AsyncMock(), from_user=SimpleNamespace(id=1))


def times(d):
    return [r["at"] for r in d.store.due_reminders(NOW + timedelta(days=60))]


async def test_no_date_wait_a_date_after_an_ask_when_capture_is_a_normal_message(tmp_path):
    d = make(tmp_path); i = asking_item(d)
    sent = await feed(d, "7 Nov", [])
    assert d.understand_fn.await_count == 1 and sent[0].startswith("Saved #")
    assert d.store.get_item(i)["fields"] == {"ask_when": True} and times(d) == []
    assert d.store.meta_get("awaiting_date_for") is None
    d.store.close()


async def test_rdone_marks_done_and_disables(tmp_path):
    d = make(tmp_path)
    i = d.store.create_item(kind="place", status="new", title="Market")
    d.store.add_reminder(i, NOW + timedelta(days=1)); d.store.add_reminder(i, NOW + timedelta(days=2))
    q = cb(f"rdone:{i}"); await on_rdone(q, d)
    assert d.store.get_item(i)["status"] == "done" and d.store.due_reminders(NOW + timedelta(days=60)) == []
    assert q.message.edit_text.await_args.args[0] == f"Done #{i} · Market"
    q2 = cb(f"rdone:{i}"); await on_rdone(q2, d)  # second tap: harmless
    assert d.store.get_item(i)["status"] == "done" and q2.answer.await_count == 1
    for bad in ("rdone:abc", "rdone:999", "rdone", f"rdone:{'9' * 30}"):
        qq = cb(bad); await on_rdone(qq, d); qq.message.edit_text.assert_not_awaited(); qq.answer.assert_awaited()
    d.store.close()


async def test_rdone_waits_for_the_drain_lock(tmp_path):
    import asyncio
    d = make(tmp_path)
    i = d.store.create_item(kind="place", status="new", title="Market")
    await d.drain_lock.acquire()
    task = asyncio.create_task(on_rdone(cb(f"rdone:{i}"), d))
    await asyncio.sleep(0.05)
    assert d.store.get_item(i)["status"] == "new"
    d.drain_lock.release(); await task
    assert d.store.get_item(i)["status"] == "done"
    d.store.close()


async def test_rsnooze_moves_the_reminder(tmp_path):
    d = make(tmp_path)
    i = d.store.create_item(kind="place", status="new", title="Market")
    rid = d.store.add_reminder(i, NOW - timedelta(hours=1)); d.store.mark_reminder_sent(rid, NOW)
    q = cb(f"rsnooze:{rid}:1d"); await on_rsnooze(q, d)
    r = d.store.get_reminder(rid)
    assert r["snoozed_until"].startswith("2026-10-31T12:00:00") and r["sent_at"] is None
    assert d.store.due_reminders(NOW + timedelta(hours=23)) == [] and len(d.store.due_reminders(NOW + timedelta(days=1))) == 1
    assert q.message.edit_text.await_args.args[0] == f"Snoozed #{i} · Market until Sat 31 Oct 12:00"
    await on_rsnooze(cb(f"rsnooze:{rid}:7d"), d)
    assert d.store.get_reminder(rid)["snoozed_until"].startswith("2026-11-06T12:00:00")
    for bad in ("rsnooze:abc:1d", "rsnooze:999:1d", f"rsnooze:{rid}:2d", f"rsnooze:{rid}", f"rsnooze:{'9' * 30}:1d"):
        qq = cb(bad); await on_rsnooze(qq, d); qq.message.edit_text.assert_not_awaited(); qq.answer.assert_awaited()
    d.store.close()


async def test_rsnooze_on_a_closed_item_does_nothing(tmp_path):
    d = make(tmp_path)
    i = d.store.create_item(kind="place", status="done", title="Market")
    rid = d.store.add_reminder(i, NOW)
    q = cb(f"rsnooze:{rid}:1d"); await on_rsnooze(q, d)
    assert d.store.get_reminder(rid)["snoozed_until"] is None
    q.message.edit_text.assert_not_awaited()
    d.store.close()


# -- forwarded messages are always captures ---------------------------------

def hidden(name="Ana"):
    return dict(forward_origin=MessageOriginHiddenUser(type="hidden_user", date=NOW, sender_user_name=name))


async def test_forwarded_message_starting_with_unknown_command_is_captured(tmp_path):
    d = make(tmp_path)
    sent = await feed(d, "/lsit films", ent("/lsit films"), **hidden())
    assert sent[0].startswith("Saved #")
    assert len(d.store.list_items(limit=10)) == 1 and d.store.list_items(limit=10)[0]["forward_from"]["name"] == "Ana"
    d.store.close()


async def test_forwarded_known_command_text_is_captured_not_run(tmp_path):
    d = make(tmp_path)
    sent = await feed(d, "/list films", ent("/list films"), **hidden())
    assert sent[0].startswith("Saved #") and len(d.store.list_items(limit=10)) == 1
    d.store.close()


def test_unknown_command_helper_ignores_forwards():
    from clawdia.bot import unknown_command
    m = SimpleNamespace(text="/lsit", entities=[SimpleNamespace(type="bot_command", offset=0, length=5)], forward_origin=object())
    assert unknown_command(m) is None
    m.forward_origin = None
    assert unknown_command(m) == "/lsit"


async def test_callbacks_from_non_owners_are_dropped(tmp_path):
    from aiogram import Bot, Dispatcher
    from aiogram.types import CallbackQuery, Chat, Message, Update, User
    from clawdia.bot import build_router
    d = make(tmp_path); i = d.store.create_item(kind="place", status="new", title="M")
    bot = Bot("1:AAA"); bot.session.make_request = AsyncMock(return_value=True)
    dp = Dispatcher(); dp.include_router(build_router(d))
    try:
        for uid in (2, 1):
            m = Message(message_id=1, date=NOW, chat=Chat(id=uid, type="private"), from_user=User(id=99, is_bot=True, first_name="b"), text="x")
            q = CallbackQuery(id="1", from_user=User(id=uid, is_bot=False, first_name="x"), chat_instance="c",
                              data=f"rdone:{i}", message=m)
            await dp.feed_update(bot, Update(update_id=uid, callback_query=q))
            assert (d.store.get_item(i)["status"] == "done") is (uid == 1)
    finally:
        await bot.session.close()
    d.store.close()


# -- a reminder request naming only a day -----------------------------------

async def test_request_naming_only_a_day_gets_nine_oclock_not_a_question(tmp_path):
    import json
    from clawdia.tagger import tag
    from tests.test_tagger import good
    from tests.test_capture import msg
    from clawdia.capture import process
    d = make(tmp_path)
    d.tag_fn = tag  # the real tagger, with a fake model returning a bare date
    payload = good(kind="place", title="Time Out Market", project=None, reminder_at="2026-10-31")
    d.ai_client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))], stop_reason="end_turn"))))
    c = await process(msg("напомни сходить в Time Out Market в субботу"), d)
    assert c.reply_text.splitlines() == [f"Saved #{c.item_id} · place · Time Out Market", "Reminder: Sat 09:00"]
    assert "ask_when" not in d.store.get_item(c.item_id)["fields"]
    assert [t[:16] for t in times(d)] == ["2026-10-31T09:00"]
    assert c.keyboard == [[("Undo", f"undo:{c.item_id}")]]


# -- snooze after Off ----------------------------------------------------------

async def test_snooze_after_off_reenables_that_reminder(tmp_path):
    d = make(tmp_path)
    i = d.store.create_item(kind="event", status="new", title="Party", fields={"reminders": ["2026-10-30T11:00:00.000000+00:00"]})
    a = d.store.add_reminder(i, NOW - timedelta(hours=1)); b = d.store.add_reminder(i, NOW + timedelta(days=2))
    d.store.mark_reminder_sent(a, NOW - timedelta(minutes=30))
    d.store.disable_reminders(i)  # "не напоминай" (converse_actions._reminders_off)
    d.store.update_item(i, fields={"reminders_off": True})
    assert d.store.get_item(i)["fields"]["reminders_off"] is True
    q = cb(f"rsnooze:{a}:1d"); await on_rsnooze(q, d)
    assert q.message.edit_text.await_args.args[0] == f"Snoozed #{i} · Party until Sat 31 Oct 12:00"
    due = d.store.due_reminders(NOW + timedelta(days=1))
    assert [r["id"] for r in due] == [a]  # the snoozed one fires; the other stays off
    assert d.store.get_reminder(b)["disabled"] == 1
    fields = d.store.get_item(i)["fields"]
    assert "reminders_off" not in fields and fields["reminders"][-1].startswith("2026-10-31T12:00:00")
    d.store.close()


async def test_snooze_of_an_enabled_reminder_leaves_the_fields_alone(tmp_path):
    d = make(tmp_path)
    i = d.store.create_item(kind="place", status="new", title="Market", fields={"city": "Lisbon"})
    rid = d.store.add_reminder(i, NOW - timedelta(hours=1)); d.store.mark_reminder_sent(rid, NOW)
    await on_rsnooze(cb(f"rsnooze:{rid}:1d"), d)
    assert d.store.get_item(i)["fields"] == {"city": "Lisbon"}
    d.store.close()


# -- a requested reminder whose time is already past asks instead -------------

def _real_tagger(d, payload):
    import json
    from clawdia.tagger import tag
    d.tag_fn = tag
    d.ai_client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))], stop_reason="end_turn"))))


@pytest.mark.parametrize("reminder_at", ["2026-10-30", "2026-10-30T09:00:00"])  # today bare, today 09:00 explicit
async def test_remind_today_after_nine_asks_when(tmp_path, reminder_at):
    from tests.test_tagger import good
    from tests.test_capture import msg
    from clawdia.capture import process
    d = make(tmp_path)  # NOW = Fri 2026-10-30 12:00 Lisbon
    _real_tagger(d, good(kind="task", title="Call mum", project=None, reminder_at=reminder_at))
    c = await process(msg("напомни сегодня позвонить маме"), d)
    assert c.reply_text.splitlines()[-1] == "When should I remind you?"
    assert c.keyboard == [[("Undo", f"undo:{c.item_id}")]]
    assert d.store.get_item(c.item_id)["fields"]["ask_when"] is True and times(d) == []


async def test_remind_with_a_future_explicit_time_schedules_exactly_it(tmp_path):
    from tests.test_tagger import good
    from tests.test_capture import msg
    from clawdia.capture import process
    d = make(tmp_path)
    _real_tagger(d, good(kind="idea", title="Call mum", project=None, reminder_at="2026-10-30T18:30:00"))
    c = await process(msg("напомни сегодня в 18:30 позвонить маме"), d)
    assert c.reply_text.splitlines()[-1] == "Reminder: Fri 18:30"
    assert [t[:16] for t in times(d)] == ["2026-10-30T18:30"]
    assert "ask_when" not in d.store.get_item(c.item_id)["fields"]
