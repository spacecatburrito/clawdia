"""Resolution notices: one line to Elena when the Claude Code session files or dismisses an item.

The CLI is a separate process and never talks to Telegram. It records `filed` / `dropped` events
with via="cli"; each loop tick sends a notice for the ones after a meta cursor and moves the cursor.
Undo, in-bot /drop and background drain events carry no via="cli", so they stay silent.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from clawdia.capture import Deps
from clawdia.converse import chat_language
from clawdia.reminders import record_bot_turn
from clawdia.render import cap
from clawdia.tagger import TITLE_MAX

log = logging.getLogger(__name__)

CURSOR_KEY = "resolution_notice_after_event_id"
NOTE_MAX = 200
_VERB = {"filed": {"ru": "в работе", "en": "in progress"}, "dropped": {"ru": "отклонено", "en": "declined"}}
_SECOND_REF = re.compile(r"Second #\d+")


def _flat(text: Any, limit: int) -> str:
    return cap(" ".join(str(text or "").split()), limit)


def _ref(payload: dict) -> str:
    ref = " ".join(str(payload.get("ref") or "").split())
    return ref if ref.startswith(("http://", "https://")) or _SECOND_REF.fullmatch(ref) else ""


def notice_text(event: dict, item: dict, lang: str) -> str:
    payload = event["payload"]
    title = _flat(item.get("title"), TITLE_MAX) or ("(без названия)" if lang == "ru" else "(no title)")
    parts = [f"#{item['id']}", title, _VERB[event["type"]][lang], _flat(payload.get("note"), NOTE_MAX)]
    if event["type"] == "filed":
        parts.append(_ref(payload))
    return " · ".join(p for p in parts if p)


async def send_resolution_notices(deps: Deps) -> None:
    """Send the notice for each new cli filed/dropped event, oldest first. A failed send stops the
    pass with the cursor before that event, so the next tick retries it and nothing is sent twice."""
    store = deps.store
    raw = store.meta_get(CURSOR_KEY)
    if raw is None:  # first run after deploy: history gets no notices
        store.meta_set(CURSOR_KEY, str(store.max_event_id()))
        return
    for event in store.events_after(int(raw), ("filed", "dropped")):
        item = store.get_item(event["item_id"]) if event["payload"].get("via") == "cli" else None
        if item is not None:
            chat_id = item.get("chat_id") or deps.settings.owner_id
            text = notice_text(event, item, "ru" if chat_language(deps, chat_id) == "ru" else "en")
            try:
                await deps.bot.send_message(chat_id, text=text)
            except (TelegramForbiddenError, TelegramBadRequest) as exc:  # Telegram will never accept it: skip, log once
                log.warning("resolution notice for item %s dropped: %s", item["id"], type(exc).__name__)
            except Exception as exc:  # noqa: BLE001 - network and other errors: retried next tick
                log.warning("resolution notice for item %s not sent: %s", item["id"], type(exc).__name__)
                return
            else:
                record_bot_turn(store, chat_id, text, item["id"])
        store.meta_set(CURSOR_KEY, str(event["id"]))
