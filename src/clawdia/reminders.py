"""Opt-in reminders: only for dated events and explicit requests ("напомни", "remind me").

Clawdia never nudges on her own. `schedule_reminders` runs once after tagging; `send_due`
is called by the background loop (B7) and sends whatever is due.

Item fields written here: `reminders` (list of UTC ISO strings, for the capture reply),
`ask_when` (an explicit request came without a time) and `reminders_off`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from dateutil import parser as dateparser

from clawdia.keyboards import markup
from clawdia.storage import Store, utc_iso
from clawdia.tagger import REMINDER_RE, TagResult

log = logging.getLogger(__name__)

SNOOZES = {"1d": timedelta(days=1), "7d": timedelta(days=7)}
CLOSED = ("done", "dismissed")
ASK_WHEN = "When should I remind you?"
TURNS_KEPT = 20  # same as converse.TURNS_KEPT: turns kept per chat
ASK_WHEN_RU = "Когда напомнить?"
_DAYS_RU = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")

Keyboard = list[list[tuple[str, str]]]


def wants_reminder(text: str | None) -> bool:
    return bool(REMINDER_RE.search(text or ""))


# -- formatting ---------------------------------------------------------


def fmt_slot(moment: datetime, tz: Any, lang: str = "en") -> str:
    """Local `Fri 19:00` (`пт 19:00` in Russian), used in the capture reply."""
    local = moment.astimezone(tz)
    if lang == "ru":
        return f"{_DAYS_RU[local.weekday()]} {local:%H:%M}"
    return local.strftime("%a %H:%M")


def fmt_full(moment: datetime, tz: Any) -> str:
    """Local `Sat 7 Nov 09:00`."""
    local = moment.astimezone(tz)
    return f"{local:%a} {local.day} {local:%b %H:%M}"


def reminder_line(fields: dict | None, tz: Any, lang: str = "en") -> str | None:
    """The reminder line of a capture reply, derived from the item's fields."""
    fields = fields or {}
    ru = lang == "ru"
    if fields.get("reminders_off"):
        return "Напоминания выключены" if ru else "Reminders off"
    stamps = fields.get("reminders")
    if stamps:
        slots = ", ".join(fmt_slot(datetime.fromisoformat(s), tz, lang) for s in stamps)
        return ("Напоминание: " if ru else "Reminder: ") + slots
    if fields.get("ask_when"):
        return ASK_WHEN_RU if ru else ASK_WHEN
    return None


def reminder_keyboard(item_id: int, reminder_id: int) -> Keyboard:
    return [[("Done", f"rdone:{item_id}"), ("Snooze 1d", f"rsnooze:{reminder_id}:1d"),
             ("Snooze 1w", f"rsnooze:{reminder_id}:7d")]]


def render_reminder(item: dict) -> str:
    title = " ".join((item.get("title") or "").split()) or "(no title)"
    return f"Reminder · #{item['id']} · {title}"


# -- event date/time normalisation ----------------------------------------


def _dateutil(text: str, default: datetime) -> datetime | None:
    # Day-first for forms like 07/11/2026, year-first when the text starts with a year (2026-11-7).
    year_first = len(text) >= 4 and text[:4].isdigit()
    try:
        return dateparser.parse(text, dayfirst=not year_first, yearfirst=year_first, default=default)
    except (ValueError, OverflowError):  # ParserError is a ValueError
        return None


def normalise_date(raw: Any, today: date) -> str | None:
    """`YYYY-MM-DD` from a model date value, or None when unreadable. ISO first, then dateutil."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        pass
    parsed = _dateutil(text, datetime.combine(today, time(0, 0)))
    return parsed.date().isoformat() if parsed is not None else None


def normalise_time(raw: Any) -> str | None:
    """`HH:MM` (24 h) from a model time value such as "19:00", "7pm" or "7.30 pm", or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace(".", ":")  # "19.30" and "7.30 pm" mean 19:30
    try:
        return time.fromisoformat(text).strftime("%H:%M")
    except ValueError:
        pass
    # Parse against two different default times: when both agree, the text itself named the time
    # (a bare "5" or a date would take the defaults' hours and disagree).
    base = date(2000, 1, 3)
    first = _dateutil(text, datetime.combine(base, time(0, 0)))
    second = _dateutil(text, datetime.combine(base, time(13, 0)))
    if first is None or second is None or first.hour != second.hour:
        return None
    return first.strftime("%H:%M")


def normalise_when(fields: dict | None, now: datetime, tz: Any) -> dict:
    """Fields with `date` as YYYY-MM-DD and `time` as HH:MM. A value that normalisation changed
    keeps its original string under `date_raw` / `time_raw`; an unreadable value is left as is."""
    out = dict(fields or {})
    today = now.astimezone(tz).date()
    for key, norm in (("date", lambda v: normalise_date(v, today)), ("time", normalise_time)):
        raw = out.get(key)
        value = norm(raw)
        if value is not None and value != raw:
            out[key] = value
            out[f"{key}_raw"] = raw
    return out


# -- scheduling ---------------------------------------------------------


def _aware(value: str | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _event_at(fields: dict | None, tz: Any) -> tuple[datetime, bool] | None:
    """Event start from `fields.date` (+ `fields.time`), localised; 00:00 and has_time False without a time."""
    fields = fields or {}
    raw_date = fields.get("date")
    if not isinstance(raw_date, str):
        return None
    try:
        day = date.fromisoformat(raw_date.strip())
    except ValueError:
        return None
    at, has_time = time(0, 0), False
    raw_time = fields.get("time")
    if isinstance(raw_time, str):
        try:
            at, has_time = time.fromisoformat(raw_time.strip()).replace(tzinfo=None), True
        except ValueError:
            pass
    return datetime.combine(day, at, tzinfo=tz), has_time


def add_reminders(store: Store, item: dict, times: list[datetime]) -> None:
    """Store reminder rows and record them in the item's fields (replacing the whole dict)."""
    if not times:
        return
    for moment in times:
        store.add_reminder(item["id"], moment)
    fields = dict(item.get("fields") or {})
    fields["reminders"] = list(fields.get("reminders") or []) + [utc_iso(t) for t in times]
    fields.pop("ask_when", None)
    fields.pop("reminders_off", None)
    store.update_item(item["id"], fields=fields)
    store.add_event(item["id"], "reminders", {"count": len(times)})


def schedule_reminders(store: Store, item: dict, result: TagResult, calendar: Any, now: datetime) -> list[datetime]:
    """Schedule reminders for a freshly tagged item.

    A model `reminder_at` is honoured only when the text carries an explicit request and then
    beats the event defaults; a dated event without a request gets the calendar defaults; an
    explicit request without a usable time asks "when?" (also when the model's time is not in
    the future, e.g. "напомни сегодня" after 09:00: a requested reminder never vanishes);
    anything else only in the past schedules nothing and does not ask.
    """
    text = " ".join(str(item.get(k) or "") for k in ("text", "caption", "transcript"))
    requested = wants_reminder(text)
    # The model's reminder_at counts only when the owner actually asked to be reminded.
    explicit = _aware(result.reminder_at) if requested else None
    event = _event_at(result.fields, calendar.tz) if result.kind == "event" else None
    if explicit is not None and explicit > now:
        times = [explicit]
    elif event is not None and explicit is None:
        times = calendar.reminder_times(event[0], event[1], now)
    else:
        if requested:  # no time, or only a past one: the capture reply asks "when?" once
            fields = dict(item.get("fields") or {})
            fields["ask_when"] = True
            store.update_item(item["id"], fields=fields)
        return []
    add_reminders(store, item, times)
    return times


# -- sending ------------------------------------------------------------


async def send_due(store: Store, bot: Any, owner_id: int, now: datetime) -> int:
    """Send every due reminder; returns how many went out. A failed send stays due for the next tick."""
    sent = 0
    for rem in store.due_reminders(now):
        item = store.get_item(rem["item_id"])
        if item is None or item["status"] in CLOSED:
            store.disable_reminders(rem["item_id"])
            continue
        text = render_reminder(item)
        try:
            await bot.send_message(owner_id, text=text,  # primary owner only
                                   reply_markup=markup(reminder_keyboard(item["id"], rem["id"])))
        except Exception as exc:  # noqa: BLE001 - retried on the next tick
            log.warning("reminder %s for item %s not sent: %s", rem["id"], item["id"], type(exc).__name__)
            continue
        store.mark_reminder_sent(rem["id"], now)
        sent += 1
        record_bot_turn(store, owner_id, text, item["id"], now)
    return sent


def record_bot_turn(store: Store, chat_id: int, text: str, item_id: int | None, at: datetime | None = None,
                    source: str = "reminder") -> None:
    """Write a message Clawdia sent on her own (a reminder ping, a drain notice) as a bot turn
    naming its item, so a typed "готово" / "snooze a week" right after it is a follow-up on that
    item. Bookkeeping only: a failure is logged (ids only) and never stops the sender."""
    try:
        store.add_turn(chat_id, "bot", text, item_id=item_id, at=at, source=source)
        store.prune_turns(chat_id, TURNS_KEPT)
    except Exception as exc:  # noqa: BLE001
        log.warning("bot turn for item %s not written: %s", item_id, type(exc).__name__)
