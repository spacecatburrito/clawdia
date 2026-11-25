"""Friday review of misses: what Elena asked for and Clawdia could not do, grouped by topic.

One Telegram HTML message on Friday evening (Lisbon), one keyboard row per topic: "To backlog" or "Skip".
Dynamic text is `html.escape`d; the whole message stays under BUDGET characters.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, time as dtime, timedelta
from typing import TYPE_CHECKING

from clawdia.keyboards import markup
from clawdia.reminders import TURNS_KEPT

if TYPE_CHECKING:
    from clawdia.capture import Deps

log = logging.getLogger(__name__)

REVIEW_TIME = dtime(17, 0)
REVIEW_UNTIL = dtime(21, 0)
MAX_TOPICS = 5
MAX_EXAMPLES = 3
BUDGET = 3500
LABEL_TOPIC = 20
FRIDAY = 4

HEADER = {"ru": "Разбор недели: что я не смогла", "en": "Weekly review: what I could not do"}
CLOSING = {"ru": "«В бэклог» - Claude сделает навык или инструмент.",
           "en": '"To backlog" - Claude will build a skill or tool.'}
COUNT = {"ru": "раз", "en": "x"}
BACKLOG = {"ru": "В бэклог: ", "en": "To backlog: "}
SKIP = {"ru": "Не надо", "en": "Skip"}


def week_key(local: datetime) -> str:
    year, week, _ = local.isocalendar()
    return f"review_sent:{year}-W{week:02d}"


def due(deps: "Deps", now: datetime) -> bool:
    store = deps.store
    local = now.astimezone(deps.calendar.tz)
    if local.weekday() != FRIDAY or store.meta_get("review_enabled") == "0":
        return False
    if not REVIEW_TIME <= local.time().replace(tzinfo=None) < REVIEW_UNTIL:
        return False
    return not store.meta_get(week_key(local))


def group(misses: list[dict]) -> list[dict]:
    """Open misses by topic: most requested first, ties by the newest miss; examples newest first."""
    by_key: dict[str, dict] = {}
    for row in misses:  # newest first
        entry = by_key.get(row["topic_key"])
        if entry is None:
            entry = by_key[row["topic_key"]] = {"topic": row["topic"], "count": 0, "newest_id": row["id"], "examples": []}
        entry["count"] += 1
        entry["newest_id"] = max(entry["newest_id"], row["id"])
        if len(entry["examples"]) < MAX_EXAMPLES:
            entry["examples"].append(row["request"])
    ordered = sorted(by_key.values(), key=lambda g: (-g["count"], -g["newest_id"]))
    return ordered[:MAX_TOPICS]


def _plain(text: str) -> str:
    return " ".join(str(text).replace(chr(0x2014), " - ").split())


def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[: max(n - 1, 0)].rstrip() + "…"


def _text(groups: list[dict], lang: str, cap: int) -> str:
    parts = [html.escape(HEADER[lang], quote=False)]
    for g in groups:
        head = f"• {html.escape(_plain(g['topic']), quote=False)} - {g['count']} {COUNT[lang]}"
        lines = "\n".join(html.escape(_cut(_plain(e), cap), quote=False) for e in g["examples"] if cap > 0 and e.strip())
        parts.append(f"{head}\n<blockquote expandable>{lines}</blockquote>" if lines else head)
    parts.append(html.escape(CLOSING[lang], quote=False))
    return "\n\n".join(parts)


def render(groups: list[dict], lang: str) -> tuple[str, list[list[tuple[str, str]]]]:
    lang = "ru" if lang == "ru" else "en"
    cap = 300
    text = _text(groups, lang, cap)
    while len(text) >= BUDGET and cap > 0:  # examples shrink first, then go
        cap = cap - 20 if cap > 20 else 0
        text = _text(groups, lang, cap)
    keyboard = [[(BACKLOG[lang] + _cut(_plain(g["topic"]), LABEL_TOPIC), f"missbl:{g['newest_id']}"),
                 (SKIP[lang], f"missno:{g['newest_id']}")] for g in groups]
    return text, keyboard


async def send_review(deps: "Deps", now: datetime) -> bool:
    from clawdia.morning import _lang  # lazy: morning pulls in the agent modules

    store, chat_id = deps.store, deps.settings.owner_id
    key = week_key(now.astimezone(deps.calendar.tz))
    groups = group(store.misses_since(now - timedelta(days=7), status="open"))
    if not groups:
        store.meta_set(key, now.isoformat())  # an empty week is done
        return False
    text, keyboard = render(groups, _lang(deps, chat_id))
    try:
        await deps.bot.send_message(chat_id, text=text, parse_mode="HTML", reply_markup=markup(keyboard))
    except Exception as exc:  # noqa: BLE001 - the key stays unset: the next tick inside the window retries
        log.warning("review not sent: %s", type(exc).__name__)
        return False
    store.meta_set(key, now.isoformat())
    try:
        store.add_turn(chat_id, "bot", text, at=deps.now_fn(), source="agent_data")
        store.prune_turns(chat_id, TURNS_KEPT)
    except Exception as exc:  # noqa: BLE001 - the message is out; bookkeeping must not resend it
        log.warning("review turn not stored: %s", type(exc).__name__)
    return True


async def check(deps: "Deps", now: datetime) -> None:
    """Loop step: send the review on Friday evening, once a week."""
    if due(deps, now):
        await send_review(deps, now)
