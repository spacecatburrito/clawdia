"""Call waits: every 10 minutes while a wait is open, pull transcripts, read the list and answer
Elena's original question when a matching transcript has arrived (no model call unless it matches).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from clawdia.agent import cap_reached, run_agent
from clawdia.capture import Deps
from clawdia.keyboards import markup
from clawdia.reminders import TURNS_KEPT
from clawdia.storage import utc_iso

log = logging.getLogger(__name__)

CHECK_EVERY = timedelta(minutes=10)
TITLE_SHOWN = 120
_DRIVE_TAG_RE = re.compile(r"\s*\[drive:[^\]]*\]\s*$")

EXPIRED = {
    "en": "No transcript of that call arrived in 3 hours; it was probably not recorded.",
    "ru": "За 3 часа запись звонка так и не появилась, похоже, он не записывался.",
}
CAP_LINE = {
    "en": "Transcript #{id} ({title}) is in Second; the daily spend cap stops the summary today.",
    "ru": "Запись #{id} ({title}) уже в Second; дневной лимит расходов не даёт сделать саммари сегодня.",
}
FAILED_LINE = {
    "en": "Transcript #{id} ({title}) is in Second, but the summary did not work out; ask me again.",
    "ru": "Запись #{id} ({title}) уже в Second, но саммари не получилось; спроси меня ещё раз.",
}
NOTE = ("Delayed answer to Elena's earlier question; the matching transcript is #{id}. "
        "Start the answer with the transcript's title. Changes she asked for were handled in the earlier turn; "
        "only answer about the call.")


def _filename(item: dict) -> str:
    untrusted = item.get("untrusted")
    name = untrusted.get("filename") if isinstance(untrusted, dict) else None
    return name if isinstance(name, str) else ""


def _int_id(item: Any) -> int | None:
    value = item.get("id") if isinstance(item, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def match(wait: dict, items: list[dict]) -> dict | None:
    """The lowest-id transcript above the wait's baseline whose file name contains the hint (if any)."""
    hint = (wait.get("title_hint") or "").strip().casefold()
    best: dict | None = None
    for item in items:
        item_id = _int_id(item)
        if item_id is None or item_id <= (wait.get("baseline_id") or 0):
            continue
        if hint and hint not in _filename(item).casefold():
            continue
        if best is None or item_id < best["id"]:
            best = item
    return best


def _title(transcript: dict) -> str:
    name = _DRIVE_TAG_RE.sub("", " ".join(_filename(transcript).split()))
    return name[:TITLE_SHOWN] or "untitled"


def _parse(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value) if value else None
    except ValueError:
        return None
    return parsed if parsed is None or parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _lang(deps: Deps, chat_id: int) -> str:
    from clawdia.converse import chat_language  # lazy: converse imports the agent module
    return "ru" if chat_language(deps, chat_id) == "ru" else "en"


class _Standin:
    """Elena's stored question as a message: chat and text only, no reply target, no voice, no forward."""

    def __init__(self, chat_id: int, text: str) -> None:
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.caption = None
        self.message_id = None
        self.reply_to_message = None
        self.voice = None
        self.forward_origin = None


async def _send(deps: Deps, chat_id: int, text: str, reply_to: int | None, keyboard=None) -> None:
    markup_arg = markup(keyboard) if keyboard else None
    if reply_to is not None:
        try:
            await deps.bot.send_message(chat_id, text=text, reply_to_message_id=reply_to, reply_markup=markup_arg)
            return
        except Exception as exc:  # noqa: BLE001 - the question message may be gone: send it plainly
            log.warning("follow-up reply not sent as a reply: %s", type(exc).__name__)
    await deps.bot.send_message(chat_id, text=text, reply_markup=markup_arg)


MAX_ATTEMPTS = 3


async def answer_wait(deps: Deps, wait: dict, transcript: dict, now: datetime) -> None:
    """Answer the wait's stored question; the caller holds the claim. A failure before the send reopens
    the wait (the next check retries); the third failure, or one past the expiry, ends it with the
    failure line. Once the message is sent the wait is answered, whatever bookkeeping does."""
    store, chat_id = deps.store, wait["chat_id"]
    sent = False
    try:
        question = wait.get("question") or ""
        if not question:
            raise LookupError("question missing")
        transcript_id = transcript["id"]
        # processed from the start and source "followup": crash recovery must not re-answer it as her message
        turn_id = store.add_turn(chat_id, "user", question, processed=True, at=now, source="followup")
        outcome = await run_agent(_Standin(chat_id, question), deps, turn_id=turn_id, now=now,
                                  followup_for=wait["id"], followup_note=NOTE.format(id=transcript_id))
        lang = _lang(deps, chat_id)
        if outcome is None:
            line = (CAP_LINE if cap_reached(deps, now) else FAILED_LINE)[lang]
            text, keyboard = line.format(id=transcript_id, title=_title(transcript)), None
            source = "agent_data"  # the line carries a third-party file name
        else:
            text, keyboard, source = outcome.text, outcome.keyboard, outcome.source
        await _send(deps, chat_id, text, wait.get("message_id"), keyboard)
        sent = True
        store.wait_set_status(wait["id"], "answered")
        store.add_turn(chat_id, "bot", text, at=deps.now_fn(), source=source)
        store.prune_turns(chat_id, TURNS_KEPT)
    except Exception as exc:  # noqa: BLE001
        log.warning("wait %s not answered: %s", wait.get("id"), type(exc).__name__)
        if sent:
            return
        await _failed(deps, wait, transcript, now)


async def _failed(deps: Deps, wait: dict, transcript: dict, now: datetime) -> None:
    store = deps.store
    attempts = store.wait_fail(wait["id"])
    expires = _parse(wait.get("expires_at"))
    if attempts < MAX_ATTEMPTS and not (expires is not None and now >= expires):
        store.wait_set_status(wait["id"], "open")
        return
    store.wait_set_status(wait["id"], "answered")  # closed: no more retries; the failure line says so
    line = FAILED_LINE[_lang(deps, wait["chat_id"])].format(id=transcript["id"], title=_title(transcript))
    try:
        await _send(deps, wait["chat_id"], line, wait.get("message_id"))
        deps.store.add_turn(wait["chat_id"], "bot", line, at=deps.now_fn(), source="agent_data")
    except Exception as exc:  # noqa: BLE001
        log.warning("wait %s failure line not sent: %s", wait.get("id"), type(exc).__name__)


async def _expire(deps: Deps, wait: dict) -> None:
    store = deps.store
    current = store.wait_get(wait["id"])
    if current is None or current["status"] != "open":
        return
    store.wait_set_status(wait["id"], "expired")  # first: a failed send must not repeat every check
    try:
        await deps.bot.send_message(wait["chat_id"], text=EXPIRED[_lang(deps, wait["chat_id"])])
    except Exception as exc:  # noqa: BLE001 - one failed send must not stop the other waits of this tick
        log.warning("wait %s expiry line not sent: %s", wait.get("id"), type(exc).__name__)


def _grace_over(wait: dict, now: datetime) -> bool:
    """Past the expiry plus one check: a late match is no longer answered (an outage must not revive it)."""
    expires = _parse(wait.get("expires_at"))
    return expires is not None and now >= expires + CHECK_EVERY


async def _expire_overdue(deps: Deps, waits: list[dict], now: datetime) -> None:
    for wait in waits:
        if _grace_over(wait, now):
            await _expire(deps, wait)


async def check_waits(deps: Deps, now: datetime) -> None:
    store = deps.store
    waits = store.waits_open()
    if not waits:
        return
    last = _parse(store.meta_get("last_calls_check_at"))
    if last is not None and now - last < CHECK_EVERY:
        return
    store.meta_set("last_calls_check_at", utc_iso(now))  # first: a down Second is not retried every minute
    try:
        pulled = await deps.second.pull_transcripts()
        if not isinstance(pulled, dict) or pulled.get("status") != "ok":
            log.info("transcript pull: %s", pulled.get("status") if isinstance(pulled, dict) else "unexpected")
    except Exception as exc:  # noqa: BLE001 - the list read still shows what Second already has
        log.warning("transcript pull failed: %s", type(exc).__name__)
    factory = getattr(deps, "second_reader_fn", None)
    if factory is None:
        log.warning("transcript list not read: no Second reader")
        await _expire_overdue(deps, waits, now)
        return
    reader = factory()
    try:
        data = await reader.call("list_transcripts", {"days": 1, "limit": 50})
    except Exception as exc:  # noqa: BLE001 - waits stay open for the next check, unless long overdue
        log.warning("transcript list not read: %s", type(exc).__name__)
        await _expire_overdue(deps, waits, now)
        return
    finally:
        await reader.close()
    items = data.get("items") if isinstance(data, dict) else None
    items = items if isinstance(items, list) else []
    for wait in waits:
        found = None if _grace_over(wait, now) else match(wait, items)
        if found is not None:
            if store.wait_claim(wait["id"], found["id"], now):
                await answer_wait(deps, wait, found, now)
            continue
        expires = _parse(wait.get("expires_at"))
        if expires is not None and now >= expires:
            await _expire(deps, wait)
