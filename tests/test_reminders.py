import pytest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock
from types import SimpleNamespace

from clawdia.reminders import render_reminder, schedule_reminders, send_due, wants_reminder
from clawdia.tagger import TagResult
from clawdia.schedule import WorkCalendar
from clawdia.storage import Store

LX = ZoneInfo("Europe/Lisbon")
cal = WorkCalendar("mon-fri", "Europe/Lisbon")
NOW = datetime(2026, 11, 1, 10, 0, tzinfo=LX)  # a Sunday


def store(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    s.ensure_schema()
    return s


def test_wants_reminder():
    assert wants_reminder("напомни сходить в Time Out Market") and wants_reminder("Remind me later")
    assert not wants_reminder("рецепт пасты") and not wants_reminder("")


def test_event_with_date_schedules_two_defaults(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="event", status="new", title="Party")
    r = TagResult("event", "Party", None, {"date": "2026-11-07", "time": None}, 0.9, None, "m")
    ts = schedule_reminders(s, s.get_item(i), r, cal, NOW)
    assert [t.astimezone(LX).strftime("%d %H:%M") for t in ts] == ["06 19:00", "07 09:00"]
    assert len(s.due_reminders(datetime(2026, 11, 7, 9, 1, tzinfo=LX))) == 2
    assert len(s.get_item(i)["fields"]["reminders"]) == 2
    s.close()


def test_event_with_time_adds_two_hours_before(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="event", status="new", title="Dinner")
    r = TagResult("event", "Dinner", None, {"date": "2026-11-07", "time": "20:30"}, 0.9, None, "m")
    ts = schedule_reminders(s, s.get_item(i), r, cal, NOW)
    assert [t.astimezone(LX).strftime("%d %H:%M") for t in ts] == ["06 19:00", "07 09:00", "07 18:30"]
    s.close()


def test_explicit_reminder_at_beats_event_defaults(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="event", status="new", title="Party", text="напомни про Party 5 ноября")
    r = TagResult("event", "Party", None, {"date": "2026-11-07"}, 0.9, "2026-11-05T08:00:00+00:00", "m")
    ts = schedule_reminders(s, s.get_item(i), r, cal, NOW)
    assert [t.isoformat() for t in ts] == ["2026-11-05T08:00:00+00:00"]
    assert len(s.due_reminders(datetime(2026, 11, 7, tzinfo=timezone.utc))) == 1
    s.close()


def test_model_reminder_at_without_a_request_word_is_ignored(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="event", status="new", title="Dinner", text="dinner Sat 7 Nov at 20:30")
    r = TagResult("event", "Dinner", None, {"date": "2026-11-07", "time": "20:30"}, 0.9, "2026-11-07T20:30:00+00:00", "m")
    ts = schedule_reminders(s, s.get_item(i), r, cal, NOW)  # model echoed the event instant
    assert [t.astimezone(LX).strftime("%d %H:%M") for t in ts] == ["06 19:00", "07 09:00", "07 18:30"]
    j = s.create_item(kind="idea", status="new", title="x", text="some idea")
    r2 = TagResult("idea", "x", None, {}, 0.9, "2026-11-05T08:00:00+00:00", "m")
    assert schedule_reminders(s, s.get_item(j), r2, cal, NOW) == [] and not s.get_item(j)["fields"]
    s.close()


def test_event_with_request_and_time_gets_only_that_time(tmp_path):
    s = store(tmp_path)
    for field in ("text", "caption", "transcript"):
        i = s.create_item(kind="event", status="new", title="Dinner", **{field: "напомни про ужин в 18:00"})
        r = TagResult("event", "Dinner", None, {"date": "2026-11-07", "time": "20:30"}, 0.9, "2026-11-07T18:00:00+00:00", "m")
        ts = schedule_reminders(s, s.get_item(i), r, cal, NOW)
        assert [t.astimezone(LX).strftime("%d %H:%M") for t in ts] == ["07 18:00"]
    s.close()


def test_bad_event_date_or_time_is_ignored(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="event", status="new", title="x")
    for f in ({"date": "next friday"}, {"date": "2026-11-07", "time": "late"}, {"date": 5}):
        schedule_reminders(s, s.get_item(i), TagResult("event", "x", None, f, 0.9, None, "m"), cal, NOW)
    # a valid date with a junk time still gets the two date-level defaults
    assert len(s.due_reminders(datetime(2030, 1, 1, tzinfo=timezone.utc))) == 2
    s.close()


def test_non_event_date_does_not_schedule(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="job", status="new", title="Role", text="apply")
    r = TagResult("job", "Role", None, {"date": "2026-11-07"}, 0.9, None, "m")
    assert schedule_reminders(s, s.get_item(i), r, cal, NOW) == []
    assert "ask_when" not in (s.get_item(i)["fields"] or {})
    s.close()


def test_past_only_event_schedules_nothing_and_does_not_ask(tmp_path):
    s = store(tmp_path)
    j = s.create_item(kind="event", status="new", title="old")
    r2 = TagResult("event", "old", None, {"date": "2026-10-01", "time": "10:00"}, 0.9, None, "m")
    assert schedule_reminders(s, s.get_item(j), r2, cal, NOW) == []
    assert "ask_when" not in (s.get_item(j)["fields"] or {})
    s.close()


def test_requested_reminder_with_only_a_past_time_asks_instead_of_vanishing(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="place", status="new", title="x", text="remind me")
    r = TagResult("place", "x", None, {}, 0.9, "2026-10-01T09:00:00+00:00", "m")
    assert schedule_reminders(s, s.get_item(i), r, cal, NOW) == []
    assert s.get_item(i)["fields"]["ask_when"] is True
    j = s.create_item(kind="event", status="new", title="Concert", text="напомни про концерт")
    r2 = TagResult("event", "Concert", None, {"date": "2026-11-20"}, 0.9, "2026-10-01T09:00:00+00:00", "m")
    assert schedule_reminders(s, s.get_item(j), r2, cal, NOW) == [] and s.get_item(j)["fields"]["ask_when"] is True
    assert s.due_reminders(datetime(2030, 1, 1, tzinfo=timezone.utc)) == []
    s.close()


def test_request_without_time_asks(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="place", status="new", title="Time Out Market", text="напомни сходить в Time Out Market")
    assert schedule_reminders(s, s.get_item(i), TagResult("place", "Time Out Market", None, {}, 0.9, None, "m"), cal, NOW) == []
    assert s.get_item(i)["fields"]["ask_when"] is True
    s.close()


def test_request_in_caption_or_transcript_asks_and_keeps_fields(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="place", status="new", title="x", transcript="remind me about this", fields={"city": "Porto"})
    schedule_reminders(s, s.get_item(i), TagResult("place", "x", None, {"city": "Porto"}, 0.9, None, "m"), cal, NOW)
    assert s.get_item(i)["fields"] == {"city": "Porto", "ask_when": True}
    s.close()


def test_plain_item_without_request_or_date_is_left_alone(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="recipe", status="new", title="Pasta", text="рецепт пасты")
    assert schedule_reminders(s, s.get_item(i), TagResult("recipe", "Pasta", None, {}, 0.9, None, "m"), cal, NOW) == []
    assert not s.get_item(i)["fields"]
    s.close()


def test_render_reminder():
    assert render_reminder({"id": 4, "title": "  Time  Out\nMarket "}) == "Reminder · #4 · Time Out Market"
    assert render_reminder({"id": 4, "title": None}) == "Reminder · #4 · (no title)"


async def test_send_due_marks_sent(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="place", status="new", title="Market")
    rid = s.add_reminder(i, NOW)
    bot = SimpleNamespace(send_message=AsyncMock())
    await send_due(s, bot, 1, NOW)
    assert bot.send_message.await_count == 1 and "Reminder · #%d · Market" % i in bot.send_message.await_args.kwargs["text"]
    kb = bot.send_message.await_args.kwargs["reply_markup"].inline_keyboard
    assert [(b.text, b.callback_data) for b in kb[0]] == [
        ("Done", f"rdone:{i}"), ("Snooze 1d", f"rsnooze:{rid}:1d"), ("Snooze 1w", f"rsnooze:{rid}:7d")]
    assert s.due_reminders(NOW) == []
    await send_due(s, bot, 1, NOW)
    assert bot.send_message.await_count == 1  # not sent twice
    s.close()


async def test_send_due_error_leaves_it_due_and_logs_ids_only(tmp_path, caplog):
    s = store(tmp_path)
    i = s.create_item(kind="place", status="new", title="SECRET TITLE")
    rid = s.add_reminder(i, NOW)
    bot = SimpleNamespace(send_message=AsyncMock(side_effect=RuntimeError("boom with SECRET TITLE")))
    with caplog.at_level("WARNING"):
        await send_due(s, bot, 1, NOW)
    assert [r["id"] for r in s.due_reminders(NOW)] == [rid]
    assert str(rid) in caplog.text and "SECRET TITLE" not in caplog.text and "RuntimeError" in caplog.text
    bot.send_message = AsyncMock()
    await send_due(s, bot, 1, NOW)
    assert bot.send_message.await_count == 1 and s.due_reminders(NOW) == []
    s.close()


async def test_send_due_continues_after_one_failure(tmp_path):
    s = store(tmp_path)
    a = s.create_item(kind="place", status="new", title="A")
    b = s.create_item(kind="place", status="new", title="B")
    ra, rb = s.add_reminder(a, NOW - timedelta(minutes=5)), s.add_reminder(b, NOW)
    bot = SimpleNamespace(send_message=AsyncMock(side_effect=[RuntimeError("x"), None]))
    await send_due(s, bot, 1, NOW)
    assert [r["id"] for r in s.due_reminders(NOW)] == [ra]
    s.close()


async def test_send_due_skips_and_disables_closed_items(tmp_path):
    s = store(tmp_path)
    done = s.create_item(kind="place", status="done", title="D")
    gone = s.create_item(kind="place", status="dismissed", title="G")
    s.add_reminder(done, NOW); s.add_reminder(gone, NOW)
    bot = SimpleNamespace(send_message=AsyncMock())
    await send_due(s, bot, 1, NOW)
    assert bot.send_message.await_count == 0 and s.due_reminders(NOW + timedelta(days=30)) == []
    s.close()


async def test_snoozed_reminder_is_sent_again_after_snooze(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="place", status="new", title="M")
    rid = s.add_reminder(i, NOW)
    bot = SimpleNamespace(send_message=AsyncMock())
    await send_due(s, bot, 1, NOW)
    s.snooze_reminder(rid, NOW + timedelta(days=1))
    await send_due(s, bot, 1, NOW + timedelta(hours=1))
    assert bot.send_message.await_count == 1
    await send_due(s, bot, 1, NOW + timedelta(days=1, minutes=1))
    assert bot.send_message.await_count == 2
    s.close()


# -- capture integration -------------------------------------------------

from clawdia.capture import process  # noqa: E402
from tests.test_capture import deps, msg  # noqa: E402


async def test_capture_event_reply_lists_reminders_and_has_only_undo(tmp_path):
    d = deps(tmp_path, TagResult("event", "Party", None, {"date": "2026-10-30"}, 0.9, None, "m"))  # now = Thu 29 Oct 10:00Z
    c = await process(msg("party on Friday"), d)
    assert c.reply_text.splitlines() == [f"Saved #{c.item_id} · event · Party", "-> Second #2601", "Reminder: Thu 19:00, Fri 09:00"]
    assert c.keyboard == [[("Undo", f"undo:{c.item_id}")]]
    assert len(d.store.due_reminders(datetime(2026, 11, 1, tzinfo=timezone.utc))) == 2
    d.store.close()


async def test_capture_explicit_request_asks_when(tmp_path):
    d = deps(tmp_path, TagResult("place", "Time Out Market", None, {}, 0.9, None, "m"))
    c = await process(msg("напомни сходить в Time Out Market"), d)
    assert c.reply_text.splitlines()[-1] == "When should I remind you?"
    assert c.keyboard == [[("Undo", f"undo:{c.item_id}")]] and d.store.get_item(c.item_id)["fields"]["ask_when"] is True
    assert d.store.due_reminders(datetime(2030, 1, 1, tzinfo=timezone.utc)) == []


async def test_capture_plain_item_has_no_reminder_line_or_row(tmp_path):
    d = deps(tmp_path, TagResult("recipe", "Pasta", None, {}, 0.9, None, "m"))
    c = await process(msg("рецепт пасты"), d)
    assert c.reply_text == f"Saved #{c.item_id} · recipe · Pasta" and c.keyboard == [[("Undo", f"undo:{c.item_id}")]]


async def test_capture_survives_scheduling_failure(tmp_path, monkeypatch):
    import clawdia.capture as capture
    def boom(*a, **k):
        raise RuntimeError("nope")
    monkeypatch.setattr(capture, "schedule_reminders", boom)
    d = deps(tmp_path, TagResult("event", "Party", None, {"date": "2026-10-30"}, 0.9, None, "m"))
    c = await process(msg("party"), d)
    assert c.reply_text == f"Saved #{c.item_id} · event · Party\n-> Second #2601"
    assert d.store._conn.execute("SELECT count(*) FROM events WHERE type='error'").fetchone()[0] == 1


# -- event date/time normalisation ----------------------------------------------

from datetime import date as _date  # noqa: E402
from clawdia.reminders import normalise_date, normalise_time, normalise_when  # noqa: E402

TODAY = _date(2026, 10, 30)  # a Friday


@pytest.mark.parametrize("raw,expected", [
    ("2026-11-07", "2026-11-07"),
    ("Saturday, 7 November", "2026-11-07"),
    ("Sat 7 Nov", "2026-11-07"),
    ("7 November 2026", "2026-11-07"),
    ("07/11/2026", "2026-11-07"),        # day first
    ("2026-11-7", "2026-11-07"),         # year first when it starts with a year
    ("2026-11-07T19:00:00", "2026-11-07"),
    ("Saturday", "2026-10-31"),
    ("next friday", None), ("soon", None), ("", None), (None, None), (5, None),
])
def test_normalise_date(raw, expected):
    assert normalise_date(raw, TODAY) == expected


@pytest.mark.parametrize("raw,expected", [
    ("19:00", "19:00"), ("7pm", "19:00"), ("7.30 pm", "19:30"), ("19.30", "19:30"), ("7 pm", "19:00"), ("7:30pm", "19:30"), ("12am", "00:00"),
    ("19:00:00", "19:00"), ("09:15", "09:15"),
    ("5", None), ("late", None), ("2026-11-07", None), ("", None), (None, None),
])
def test_normalise_time(raw, expected):
    assert normalise_time(raw) == expected


def test_normalise_when_keeps_the_raw_string_only_when_changed():
    out = normalise_when({"date": "Saturday, 7 November", "time": "7pm", "place": "Lux"}, NOW, LX)
    assert out == {"date": "2026-11-07", "date_raw": "Saturday, 7 November", "time": "19:00", "time_raw": "7pm", "place": "Lux"}
    assert normalise_when({"date": "2026-11-07", "time": "19:00"}, NOW, LX) == {"date": "2026-11-07", "time": "19:00"}
    assert normalise_when({"date": "someday", "time": "late"}, NOW, LX) == {"date": "someday", "time": "late"}
    assert normalise_when(None, NOW, LX) == {}


async def test_capture_normalises_a_non_iso_event_date_and_time(tmp_path):
    d = deps(tmp_path, TagResult("event", "Concert", None, {"date": "Saturday, 31 October", "time": "7pm"}, 0.9, None, "m"))
    c = await process(msg("concert Saturday 7pm"), d)  # now = Thu 29 Oct 10:00Z
    f = d.store.get_item(c.item_id)["fields"]
    assert (f["date"], f["time"], f["date_raw"], f["time_raw"]) == ("2026-10-31", "19:00", "Saturday, 31 October", "7pm")
    assert c.reply_text.splitlines()[-1] == "Reminder: Fri 19:00, Sat 09:00, Sat 17:00"
    d.store.close()


async def test_reminder_ping_is_stored_as_a_reminder_turn(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="place", status="new", title="Market", chat_id=1)
    s.add_reminder(i, NOW)
    await send_due(s, SimpleNamespace(send_message=AsyncMock()), 1, NOW)
    assert [(t["role"], t["source"]) for t in s.recent_turns(1, 10)] == [("bot", "reminder")]
    s.close()
