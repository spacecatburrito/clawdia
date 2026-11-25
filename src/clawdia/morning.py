"""Morning digest: one Telegram message on weekday mornings, format A.

Second does the thinking (`morning_digest` read tool: prepared blocks, importance by code), Clawdia
renders and delivers. The message is Telegram HTML: a 🔴 block of at most three lines, then one headline
line per non-empty topic with its details in an expandable blockquote, then "Ask me about any item.".
Every third-party string (chat, event, task, company, role) and every map line is `html.escape`d; text
from Second's code-built headline/details is escaped too. The message is stored as an "agent_data" turn.

Budget by code: visible text (outside blockquotes) at most 900 characters, the whole message at most
3,500; headlines shrink first, then details are trimmed with "+N more". The 🔴 lines are never dropped.
"""

from __future__ import annotations

import asyncio
import html
import logging
import math
import re
from datetime import datetime, time as dtime
from typing import TYPE_CHECKING, Any

from clawdia.focus import today_map, top_items
from clawdia.keyboards import focus_request_keyboard, markup
from clawdia.reminders import TURNS_KEPT

if TYPE_CHECKING:
    from clawdia.capture import Deps

log = logging.getLogger(__name__)

VISIBLE_MAX = 900  # characters outside the blockquotes
TOTAL_MAX = 3500  # the whole message (Telegram allows 4096)
RED_MAX = 150  # one 🔴 line
RED_LINES = 3
DETAIL_MAX = 200  # one details line, before escaping
LINK_MAX = 300
HEADLINE_CAPS = (90, 60, 40, 25)
REFRESH_TIMEOUT_S = 75.0  # a little over the client's 70 s, so the client reports its own timeout
READ_TIMEOUT_S = 45.0
SINCE_HOURS = 24
SINCE_MAX_HOURS = 72
STALE_MINUTES = 60

DEFAULT_TIME = "09:30"
EARLIEST = dtime(5, 0)
END = dtime(12, 0)  # a digest that has not gone out by noon is skipped for the day
DAYS = ("mon", "tue", "wed", "thu", "fri")
DAY_NAMES = {"mon": "mon", "monday": "mon", "tue": "tue", "tues": "tue", "tuesday": "tue", "wed": "wed",
             "wednesday": "wed", "thu": "thu", "thur": "thu", "thurs": "thu", "thursday": "thu", "fri": "fri",
             "friday": "fri"}

NO_MAP = {"en": "No day map yet", "ru": "Карты дня нет"}
TOP_LABEL = {"en": "Top 1", "ru": "Топ 1"}
CLOSING = {"en": "Ask me about any item.", "ru": "Спроси меня про любой пункт."}
UNAVAILABLE = {"en": "{topic}: not available right now", "ru": "{topic}: сейчас недоступно"}
MORE = {"en": "+{n} more", "ru": "+{n} ещё"}
LINK = {"en": "link", "ru": "ссылка"}
OLD_DATA = {"en": "Messages are about {h} h old.", "ru": "Сообщения старше примерно {h} ч."}
OVERDUE_LABEL = {"en": "{n} overdue", "ru": "Просрочено: {n}"}
UNAVAILABLE_DIGEST = {"en": "Digest unavailable: Second did not answer.",
                      "ru": "Сводка недоступна: Second не ответил."}
TOPICS = (
    ("calls", "📅", {"en": "Calls today", "ru": "Звонки сегодня"}),
    ("silent", "💬", {"en": "No reply 3+ days", "ru": "Без ответа 3+ дня"}),
    ("promises", "🤝", {"en": "Your promises", "ru": "Твои обещания"}),
    ("vacancies", "💼", {"en": "Vacancies", "ru": "Вакансии"}),
)
_TIME_RE = re.compile(r"\d{1,2}:\d{2}")


# -- settings (meta) ------------------------------------------------------------------------------


def parse_time(raw: Any) -> dtime | None:
    """"9:30" or "09:30" within the send window, else None."""
    if not isinstance(raw, str) or not re.fullmatch(r"\d{1,2}:\d{2}", raw.strip()):
        return None
    hour, minute = (int(part) for part in raw.strip().split(":"))
    if hour > 23 or minute > 59:
        return None
    parsed = dtime(hour, minute)
    return parsed if EARLIEST <= parsed < END else None


def settings_view(store) -> dict:
    """The effective settings: a missing or broken stored value means the default."""
    parsed = parse_time(store.meta_get("digest_time"))
    stored_days = [d for d in (store.meta_get("digest_days") or "").split(",") if d in DAYS]
    return {"enabled": store.meta_get("digest_enabled") != "0",
            "time": (parsed or parse_time(DEFAULT_TIME)).strftime("%H:%M"),
            "days": [d for d in DAYS if d in stored_days] or list(DAYS)}


def restore_digest(store, before: dict) -> None:
    """Undo of set_digest: put the raw meta values back (None means it was never set)."""
    for key in ("enabled", "time", "days"):
        if key in before:
            store.meta_set(f"digest_{key}", before[key] if isinstance(before[key], str) else "")


def due(deps: "Deps", now: datetime) -> bool:
    store = deps.store
    local = now.astimezone(deps.calendar.tz)
    if local.weekday() > 4 or store.meta_get(f"digest_sent:{local.date().isoformat()}"):
        return False
    cfg = settings_view(store)
    if not cfg["enabled"] or DAYS[local.weekday()] not in cfg["days"]:
        return False
    return parse_time(cfg["time"]) <= local.time().replace(tzinfo=None) < END


# -- rendering ------------------------------------------------------------------------------------


def _s(value: Any) -> str:
    """A third-party field as one line of text; anything but a string is empty."""
    return " ".join(value.split()) if isinstance(value, str) else ""


def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[: max(n - 1, 0)].rstrip() + "…"


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _items(blocks: dict, key: str) -> list[dict]:
    block = blocks.get(key) if isinstance(blocks, dict) else None
    items = block.get("items") if isinstance(block, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _count(blocks: dict, key: str) -> int:
    block = blocks.get(key) if isinstance(blocks, dict) else None
    total = block.get("total") if isinstance(block, dict) else None
    items = _items(blocks, key)
    return total if isinstance(total, int) and not isinstance(total, bool) and total >= len(items) else len(items)


def _failed(blocks: dict, key: str) -> bool:
    block = blocks.get(key) if isinstance(blocks, dict) else None
    return isinstance(block, dict) and block.get("error") is True


def _untrusted(item: dict, key: str) -> str:
    raw = item.get("untrusted")
    return _s(raw.get(key)) if isinstance(raw, dict) else ""


def _projects(item: dict) -> str:
    raw = item.get("untrusted")
    projects = raw.get("projects") if isinstance(raw, dict) else None
    return ", ".join(p for p in (_s(x) for x in projects) if p) if isinstance(projects, list) else ""


def _headline(item: dict) -> str:
    return _s(item.get("headline"))


def _time_of(item: dict) -> str:
    found = _TIME_RE.search(_headline(item))
    return found.group(0) if found else ""


def _name(topic: str, item: dict) -> str:
    """The third-party name shown in a topic headline."""
    if topic == "calls":
        return f"{_time_of(item)} {_untrusted(item, 'title')}".strip()
    if topic == "silent":
        return _untrusted(item, "chat_title")
    if topic == "promises":
        return _untrusted(item, "title")
    role, company = _untrusted(item, "role"), _untrusted(item, "company")
    return f"{role} ({company})" if role and company else role or company


def _detail(topic: str, item: dict) -> str:
    head, code = _headline(item), _s(item.get("details"))
    if topic == "calls":
        place = _untrusted(item, "location")
        title = _untrusted(item, "title")
        line = f"{_time_of(item)} {title}".strip() if _time_of(item) else f"{code}: {title}".strip(": ")
        return f"{line} ({place})" if place else line
    if topic == "silent":
        where = _projects(item)
        chat = _untrusted(item, "chat_title")
        return f"{chat} ({where}) - {head}" if where else f"{chat} - {head}"
    if topic == "promises":
        return f"{_untrusted(item, 'title')} - {head} ({code})" if code else f"{_untrusted(item, 'title')} - {head}"
    role, company = _untrusted(item, "role"), _untrusted(item, "company")
    who = f"{role}, {company}" if role and company else role or company
    return f"{who} - {code}" if who else code


def _link(item: dict, lang: str) -> str:
    url = item.get("link")
    if not isinstance(url, str) or not re.match(r"https?://", url) or len(url) > LINK_MAX:
        return ""
    return f' <a href="{html.escape(url, quote=True)}">{LINK[lang]}</a>'


def _red_lines(blocks: dict, map_text: str | None, lang: str) -> list[str]:
    """Raw (unescaped) 🔴 lines, at most RED_LINES: the map's Top-1 (or "No day map yet"), one aggregated
    overdue-promise line, the soonest must-see call, then "Decisions for you" lines fill what is left."""
    from clawdia.mapwatch import _decision_lines  # lazy: mapwatch imports the capture module

    lines: list[str] = []
    if map_text is None:
        lines.append(NO_MAP[lang])
    else:
        top = top_items(map_text, 1)
        if top:
            lines.append(f"{TOP_LABEL[lang]}: {_s(top[0])}")
    overdue = [i for i in _items(blocks, "promises") if i.get("importance") == "must_see"]
    if overdue:
        titles = ", ".join(t for t in (_untrusted(i, "title") for i in overdue) if t)
        count = max(len(overdue), _count(blocks, "promises") if len(overdue) == len(_items(blocks, "promises")) else 0)
        lines.append(f"{OVERDUE_LABEL[lang].format(n=count)}: {titles}".rstrip(": "))
    for item in _items(blocks, "calls"):
        if item.get("importance") == "must_see":
            lines.append(f"{_headline(item)}: {_untrusted(item, 'title')}".strip(": "))
            break
    if map_text is not None:
        lines.extend(_decision_lines(map_text))
    return [_cut(line, RED_MAX) for line in lines if line][:RED_LINES]


def _topic_headline(blocks: dict, topic: str, emoji: str, label: str, lang: str, cap: int) -> str:
    items = _items(blocks, topic)
    count = _count(blocks, topic)
    head = f"{emoji} {label}: {count}"
    if topic == "promises":
        overdue = sum(1 for i in items if i.get("importance") == "must_see")
        if overdue:
            head += f" ({OVERDUE_LABEL[lang].format(n=overdue)})"
    names = [n for n in (_name(topic, i) for i in items[:2]) if n]
    if names:
        head += " - " + ", ".join(names)
    return _cut(head, cap)


def _freshness_line(blocks: dict, lang: str) -> str:
    fresh = blocks.get("freshness") if isinstance(blocks, dict) else None
    age = fresh.get("latest_message_age_minutes") if isinstance(fresh, dict) else None
    if isinstance(age, (int, float)) and not isinstance(age, bool) and age > STALE_MINUTES:
        return OLD_DATA[lang].format(h=round(age / 60))
    return ""


def render(blocks: dict, map_text: str | None, lang: str, now: datetime) -> tuple[str, list[list[tuple[str, str]]] | None]:
    """The digest as Telegram HTML and the keyboard to attach ("" and None when there is nothing to send).
    `map_text` is today's map or None when it does not exist yet."""
    lang = "ru" if lang == "ru" else "en"
    blocks = blocks if isinstance(blocks, dict) else {}
    reds = _red_lines(blocks, map_text, lang)
    present = [(key, emoji, label) for key, emoji, label in TOPICS if _items(blocks, key)]
    failed = [(key, label) for key, _, label in TOPICS if _failed(blocks, key) and not _items(blocks, key)]
    if not reds and not present:
        if all(_failed(blocks, key) for key, _, _ in TOPICS):  # Second did not answer at all: say so once
            return _esc(UNAVAILABLE_DIGEST[lang]), None
        return "", None
    keyboard = focus_request_keyboard(lang) if map_text is None else None
    red_html = "\n".join(f"🔴 {_esc(line)}" for line in reds)
    unavailable = [_esc(UNAVAILABLE[lang].format(topic=label[lang])) for _, label in failed]
    closing = [CLOSING[lang]]
    stale = _freshness_line(blocks, lang)
    if stale:
        closing.insert(0, stale)

    def skeleton(cap: int) -> tuple[list[tuple[str, str]], int, int]:
        heads = [(key, _topic_headline(blocks, key, emoji, label[lang], lang, cap)) for key, emoji, label in present]
        visible = len("\n".join(["🔴 " + line for line in reds] + [h for _, h in heads] + [
            UNAVAILABLE[lang].format(topic=label[lang]) for _, label in failed] + closing)) + 6
        shell = (len(red_html) + sum(len(_esc(h)) + len("\n<blockquote expandable></blockquote>") for _, h in heads)
                 + sum(len(u) for u in unavailable) + len(_esc("\n".join(closing))) + 40 + 14 * len(heads))
        return [(k, _esc(h)) for k, h in heads], visible, shell

    for cap in HEADLINE_CAPS:
        heads, visible, shell = skeleton(cap)
        if visible <= VISIBLE_MAX and shell <= TOTAL_MAX:
            break

    details: dict[str, list[str]] = {}
    for key, _, _ in present:
        rows = []
        for item in _items(blocks, key):
            line = _cut(_detail(key, item), DETAIL_MAX)
            if line:
                rows.append(_esc(line) + _link(item, lang))
        details[key] = rows
    shown: dict[str, list[str]] = {key: [] for key, _ in heads}
    more: dict[str, int] = {key: 0 for key, _ in heads}

    def assemble() -> str:
        parts = [red_html] if red_html else []
        for key, head in heads:
            body = "\n".join(shown[key] + ([MORE[lang].format(n=more[key])] if more[key] else []))
            parts.append(f"{head}\n<blockquote expandable>{body}</blockquote>" if body else head)
        parts.extend(unavailable)
        parts.append(_esc("\n".join(closing)))
        return "\n\n".join(parts)

    exhausted = False  # once one row does not fit, every later row only counts toward "+N more"
    for key, _ in heads:
        rows = details[key]
        for index, row in enumerate(rows):
            if not exhausted:
                shown[key].append(row)
                if len(assemble()) > TOTAL_MAX - 40:  # keep room for the "+N more" lines
                    shown[key].pop()
                    exhausted = True
            if exhausted:
                more[key] = len(rows) - index
                break
    text = assemble()
    while len(text) > TOTAL_MAX:  # a new blockquote per later topic outgrew the reserve: drop rows, largest topic first
        key = max(shown, key=lambda k: len(shown[k]), default=None)
        if key is None or not shown[key]:
            break
        shown[key].pop()
        more[key] += 1
        text = assemble()
    return text, keyboard


# -- sending --------------------------------------------------------------------------------------


def _lang(deps: "Deps", chat_id: int) -> str:
    from clawdia.converse import chat_language  # lazy: converse imports the agent module
    return "ru" if chat_language(deps, chat_id) == "ru" else "en"


def _read_map(deps: "Deps", today) -> str | None:
    folder = getattr(deps.settings, "focus_dir", None)
    path = today_map(folder, today) if folder is not None else None
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("map not read for the digest: %s", type(exc).__name__)
        return None


def since_hours(store, now: datetime) -> int:
    """Hours since the digest last went out, clamped to 24..72 (a Monday digest covers the weekend)."""
    try:
        last = datetime.fromisoformat(store.meta_get("digest_last_sent_at") or "")
    except ValueError:
        return SINCE_HOURS
    if last.tzinfo is None:
        last = last.replace(tzinfo=now.tzinfo)
    hours = (now - last).total_seconds() / 3600
    return math.ceil(min(max(hours, SINCE_HOURS), SINCE_MAX_HOURS)) if hours > 0 else SINCE_HOURS


def _vacancies_read(blocks: dict) -> bool:
    block = blocks.get("vacancies") if isinstance(blocks, dict) else None
    return isinstance(block, dict) and block.get("error") is not True


async def _read_blocks(deps: "Deps", now: datetime) -> dict:
    try:
        await asyncio.wait_for(deps.second.refresh_sources(), REFRESH_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - the digest goes out anyway, its freshness line says how old the data is
        log.warning("source refresh failed: %s", type(exc).__name__)
    factory = getattr(deps, "second_reader_fn", None)
    if factory is not None:
        reader = factory()
        try:
            data = await asyncio.wait_for(reader.call("morning_digest", {"since_hours": since_hours(deps.store, now)}), READ_TIMEOUT_S)
            if isinstance(data, dict):
                return data
        except Exception as exc:  # noqa: BLE001
            log.warning("morning digest not read: %s", type(exc).__name__)
        finally:
            try:
                await reader.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("digest reader not closed: %s", type(exc).__name__)
    return {key: {"items": [], "total": 0, "error": True} for key, _, _ in TOPICS}


async def send_digest(deps: "Deps", now: datetime) -> None:
    store, chat_id = deps.store, deps.settings.owner_id
    today = now.astimezone(deps.calendar.tz).date()
    store.meta_set(f"digest_sent:{today.isoformat()}", now.isoformat())  # first: a failed send is not retried every minute
    blocks = await _read_blocks(deps, now)
    text, keyboard = render(blocks, _read_map(deps, today), _lang(deps, chat_id), now)
    if not text:
        return
    await deps.bot.send_message(chat_id, text=text, parse_mode="HTML", reply_markup=markup(keyboard) if keyboard else None)
    try:
        if _vacancies_read(blocks):  # a failed vacancies block must not move the window past unseen triage
            store.meta_set("digest_last_sent_at", now.isoformat())  # only after a successful send
        store.add_turn(chat_id, "bot", text, at=deps.now_fn(), source="agent_data")
        store.prune_turns(chat_id, TURNS_KEPT)
    except Exception as exc:  # noqa: BLE001 - the message is out; bookkeeping must not resend it
        log.warning("digest turn not stored: %s", type(exc).__name__)


async def check(deps: "Deps", now: datetime) -> None:
    """Loop step: send the digest when it is due (weekday, enabled, from the send time until noon, once a day)."""
    if due(deps, now):
        await send_digest(deps, now)
