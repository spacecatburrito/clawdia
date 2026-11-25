"""Day map watch: tell Elena when today's map has landed on the server, when the laptop reported a
failed build, and once when a requested map is an hour late. Runs every loop tick; each notice is
guarded by a meta key so a restart or a second tick never repeats it.

The map quotes chats and calls, so it is third-party text: sent as plain text (no parse mode) and
stored as an "agent_data" turn.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from clawdia.capture import Deps
from clawdia.focus import _clean, split_sections, today_map, top_items
from clawdia.reminders import TURNS_KEPT
from clawdia.storage import utc_iso

log = logging.getLogger(__name__)

REMINDER_AFTER = timedelta(minutes=60)
TOP_SHOWN = 4
DECISIONS_SHOWN = 4
LINE_MAX = 240

READY = {"en": "Day map ready", "ru": "Карта дня готова"}
DECISIONS = {"en": "Decisions for you", "ru": "Нужно твоё решение"}
FAILED = {
    "en": "The laptop could not build the day map: {reason}",
    "ru": "Ноутбук не смог собрать карту дня: {reason}",
}
FAILED_NO_REASON = {"en": "no reason given", "ru": "причина не указана"}
LATE = {
    "en": "You asked for a day map an hour ago and it has not arrived. The laptop is probably asleep or offline.",
    "ru": "Ты просила карту дня час назад, она так и не пришла. Ноутбук, скорее всего, спит или не в сети.",
}


def _parse(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value) if value else None
    except ValueError:
        return None
    return parsed if parsed is None or parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _lang(deps: Deps, chat_id: int) -> str:
    from clawdia.converse import chat_language  # lazy: converse imports the agent module
    return "ru" if chat_language(deps, chat_id) == "ru" else "en"


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= LINE_MAX else text[: LINE_MAX - 1].rstrip() + "…"


def _decision_lines(text: str) -> list[str]:
    heading = next((h for h, _ in split_sections(text).items() if "decisions for you" in h), None)
    if heading is None:
        return []
    lines = []
    for line in split_sections(text)[heading].splitlines():
        line = _clean(line.strip().lstrip("-*").strip())
        if line:
            lines.append(_short(line))
    return lines[:DECISIONS_SHOWN]


def ready_text(map_text: str, lang: str) -> str:
    parts = [READY[lang]]
    items = top_items(map_text, TOP_SHOWN)
    if items:
        parts.append("\n".join(f"{i}. {_short(item)}" for i, item in enumerate(items, 1)))
    decisions = _decision_lines(map_text)
    if decisions:
        parts.append("\n".join(f"🔴 {line}" for line in decisions))
    return "\n\n".join(parts)


async def _notify(deps: Deps, text: str) -> None:
    """Send to the primary owner and keep the bot turn; a failed send raises so the guard stays unset."""
    chat_id = deps.settings.owner_id
    await deps.bot.send_message(chat_id, text=text)
    try:
        deps.store.add_turn(chat_id, "bot", text, at=deps.now_fn(), source="agent_data")
        deps.store.prune_turns(chat_id, TURNS_KEPT)
    except Exception as exc:  # noqa: BLE001 - the message is out; bookkeeping must not resend it
        log.warning("map notice turn not stored: %s", type(exc).__name__)


async def check_map(deps: Deps, now: datetime) -> None:
    store, settings = deps.store, deps.settings
    chat_id = settings.owner_id
    lang = _lang(deps, chat_id)
    today = now.astimezone(deps.calendar.tz).date()
    path = today_map(settings.focus_dir, today)

    if path is not None:
        key = f"map_ready_sent:{today.isoformat()}"
        if not store.meta_get(key):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.warning("map not read: %s", type(exc).__name__)
            else:
                await _notify(deps, ready_text(text, lang))
                store.meta_set(key, utc_iso(now))

    report = store.focus_report_get()
    if report and report.get("status") == "failed":
        at = report.get("at") or ""
        if store.meta_get("focus_failed_notified_at") != at:
            reason = _short(report.get("reason") or "") or FAILED_NO_REASON[lang]
            await _notify(deps, FAILED[lang].format(reason=reason))
            store.meta_set("focus_failed_notified_at", at)

    if path is not None and store.meta_get("focus_request"):
        store.meta_set("focus_request", "")  # the map exists: a pending request must not start a second build

    requested = store.meta_get("focus_requested_at")
    asked = _parse(requested)
    if asked is None:
        return
    reported = _parse((report or {}).get("at"))
    answered = reported is not None and reported >= asked  # an ok or failed report closes the request
    if path is not None or (answered and report.get("status") == "ok"):
        store.meta_set("focus_requested_at", "")  # fulfilled: a later day must not remind about it
        return
    same_day = asked.astimezone(deps.calendar.tz).date() == today
    if same_day and now - asked >= REMINDER_AFTER and not answered:
        if store.meta_get("focus_reminder_sent_for") != requested:
            await _notify(deps, LATE[lang])
            store.meta_set("focus_reminder_sent_for", requested)
