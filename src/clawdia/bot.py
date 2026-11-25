"""Telegram handlers (aiogram 3) and the polling entry point."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command, Filter
from aiogram.types import CallbackQuery, Message

from clawdia import media, transcribe
from clawdia.capture import Captured, Deps, flush_now, process, sync_now
from clawdia.config import Settings
from clawdia.context import ContextCache
from clawdia.focus import today_map
from clawdia.converse import chat_language, handle_capture, handle_text, handle_voice, recover_unprocessed
from clawdia.jobs import reopen_if_requeued
from clawdia.keyboards import markup, undo_turn_keyboard
from clawdia.memory import (PENDING_TTL, _CONFIRM_LINE, _is_proposal_line, check_add, fit_edit, fit_reply,
                            render_list as render_memory)
from clawdia.loops import refresh_context, run_loops
from clawdia.pairing import handle_glued, hold_forward, hold_text
from clawdia.render import (  # noqa: F401 - re-exported for callers and tests
    HELP_TEXT, MAX_ID, UNKNOWN_SHELF, cap, find_item, greeting, handle_done, handle_drop, parse_id, parse_list_args, render_inbox,
    render_list, render_show, second_note, second_task_ref, ago)
from clawdia.morning import restore_digest
from clawdia.reminders import CLOSED, SNOOZES, fmt_full, record_bot_turn, send_due  # noqa: F401 - send_due is re-exported for the background loop (B7)
from clawdia.reply import fallback_undone
from clawdia.schedule import WorkCalendar
from clawdia.second_client import SecondClient
from clawdia.second_reads import SecondReader
from clawdia.storage import ITEM_COLUMNS, Store, utc_iso
from clawdia.tagger import tag
from clawdia.understand import guess_language

log = logging.getLogger(__name__)


class OwnerOnly(Filter):
    """Pass only updates from an allowed owner account in its own private chat; everything else is dropped silently.

    A message must come from an allowed id in a private chat whose id is that same id (so an owner
    writing in a group, or a chat that belongs to another account, gets nothing); a callback must
    come from an allowed id on a message in that chat.
    """

    def __init__(self, allowed_ids: frozenset[int]) -> None:
        self.allowed_ids = frozenset(allowed_ids)

    async def __call__(self, event: Any) -> bool:
        user = getattr(event, "from_user", None)
        if user is None or user.id not in self.allowed_ids:
            return False
        message = getattr(event, "message", None) if isinstance(event, CallbackQuery) else event
        chat = getattr(message, "chat", None)
        if chat is None or getattr(chat, "type", None) != "private":
            return False
        chat_id = getattr(chat, "id", None)
        return chat_id in self.allowed_ids and chat_id == user.id


# -- handlers -----------------------------------------------------------


def _touch(deps: Deps) -> None:
    try:
        deps.store.meta_set("last_update_at", utc_iso())
    except Exception as exc:  # noqa: BLE001 - a bookkeeping failure must not cost the owner a reply
        log.warning("meta last_update_at not written: %s", type(exc).__name__)


async def on_message(message: Message, deps: Deps) -> None:
    """Forwards and media: the phase 1 capture (item first), then the conversational reply."""
    _touch(deps)
    spoken = await handle_voice(message, deps)  # her own voice, agent brain: answered like typed text
    if spoken is not None:
        await message.answer(spoken.text, reply_markup=markup(spoken.keyboard) if spoken.keyboard else None)
        return
    if await hold_forward(message, deps):  # a typed comment took it: that handler sends the one reply
        return
    try:
        captured: Captured = await process(message, deps)
    except Exception as exc:  # noqa: BLE001 - only reachable when nothing could be saved
        log.error("capture failed before the item was saved: %s", type(exc).__name__)
        await message.answer("Could not save, try again")
        return
    if captured.duplicate:  # later part of an album: the first part already got the reply
        return
    outcome = await handle_capture(message, deps, captured)
    await message.answer(outcome.text, reply_markup=markup(outcome.keyboard) if outcome.keyboard else None)


async def on_text(message: Message, deps: Deps) -> None:
    """Plain typed text: the conversation engine (understand, guard, act, reply).

    The turn is written here (persist first) so that a failure told to the owner as "Could not
    save" can be marked processed: the startup recovery must not answer it a second time.
    """
    _touch(deps)
    turn_id = None
    try:
        turn_id = deps.store.add_turn(message.chat.id, "user", message.text or "", processed=False, at=deps.now_fn())
        forwards = await hold_text(message, deps)  # a forward right before or after makes one item
        outcome = await (handle_glued(message, deps, forwards, turn_id) if forwards
                         else handle_text(message, deps, turn_id=turn_id))
    except Exception as exc:  # noqa: BLE001
        log.error("text message not handled: %s", type(exc).__name__)
        if turn_id is not None:
            try:
                deps.store.mark_turn(turn_id, processed=True)
            except Exception as mark_exc:  # noqa: BLE001
                log.warning("turn %s not marked: %s", turn_id, type(mark_exc).__name__)
        await message.answer("Could not save, try again")
        return
    await message.answer(outcome.text, reply_markup=markup(outcome.keyboard) if outcome.keyboard else None)


UNSUPPORTED = {"ru": "Пока не умею сохранять такое.", "en": "I can't save this type yet."}


async def on_unsupported(message: Message, deps: Deps) -> None:
    """Location, contact, poll and other types without text or media: say so, save nothing, no model call."""
    _touch(deps)
    await message.answer(UNSUPPORTED["ru" if chat_language(deps, message.chat.id) == "ru" else "en"])


MEDIA_ATTRS = ("photo", "document", "voice", "audio", "video", "video_note", "animation", "sticker")


def is_media(message: Any) -> bool:
    return any(getattr(message, attr, None) for attr in MEDIA_ATTRS)


def _parse(data: str | None, prefix: str, parts: int) -> list[str] | None:
    pieces = (data or "").split(":")
    ok = len(pieces) == parts and pieces[0] == prefix and pieces[1].isascii() and pieces[1].isdigit()
    return pieces if ok else None


async def _edit(callback: CallbackQuery, text: str, keyboard: list[list[tuple[str, str]]] | None) -> None:
    edit = getattr(callback.message, "edit_text", None)
    if edit is None:
        return
    try:
        await edit(text, reply_markup=markup(keyboard) if keyboard else None)
    except Exception as exc:  # noqa: BLE001 - e.g. "message is not modified"
        log.info("edit_text failed: %s", type(exc).__name__)


async def _skip_job(deps: Deps, item_id: int, intake_id: int) -> None:
    """Undo on a triaged vacancy tells Second "skip"; a failure is retried by the poll loop."""
    sent = False
    try:
        await deps.second.patch_job(intake_id, "skip", "Undone in Clawdia")
        sent = True
    except Exception as exc:  # noqa: BLE001
        log.warning("job %s skip not sent: %s", item_id, type(exc).__name__)
    fields = dict((deps.store.get_item(item_id) or {}).get("fields") or {})
    fields.pop("job_decision_note", None)  # an earlier agent decision's note must not ride on the skip
    if sent:
        fields.pop("job_decision_pending", None)  # a pending "apply" must not be sent after the undo
    else:
        fields["job_decision_pending"] = "skip"
    deps.store.update_item(item_id, fields=fields)


async def on_undo(callback: CallbackQuery, deps: Deps) -> None:
    """The Undo button of a capture reply: dismiss the item and stop its reminders. No model call."""
    _touch(deps)
    parts = _parse(callback.data, "undo", 2)
    item_id = int(parts[1]) if parts else None
    item, already, changed = None, False, False
    async with deps.drain_lock:  # a status write must not interleave with a drain
        item = find_item(deps.store, item_id) if item_id is not None else None
        if item is not None:
            already = item["status"] == "dismissed"
            fields = item.get("fields") or {}
            if not already and fields.get("job_intake_id") and fields.get("job_decision"):
                changed = True  # decided since: leave the vacancy as is
            elif not already:
                deps.store.update_item(item_id, status="dismissed", route_error=None)  # also leaves the outbox
                deps.store.disable_reminders(item_id)
                deps.store.add_event(item_id, "undone", {})
                intake = (item.get("fields") or {}).get("job_intake_id")
                if intake:
                    await _skip_job(deps, item_id, int(intake))
    if item is None:
        await callback.answer()
        return
    if already:
        await callback.answer(f"Already undone #{item_id}")
        return
    if changed:
        await callback.answer(f"#{item_id} changed since, left as is")
        return
    words = " ".join(str(item.get(k) or "") for k in ("text", "caption", "transcript"))
    async with deps.drain_lock:
        kept_rows, kept_lines = _open_proposals(callback, deps.store)  # an open memory proposal is not a write: Undo keeps it
    await _edit(callback, fit_reply(fallback_undone(item_id, guess_language(words)) + second_note(item), kept_lines),
                kept_rows or None)
    await callback.answer()


SECOND_STAYS = ("save_note", "send_to_second", "update_item")  # undone writes after which a Second task remains
UNDO_TEXT = {
    "ru": {"already": "Уже отменено", "done": "Отменила.", "failed": "Часть изменений не отменилась, нажми Undo ещё раз",
           "none": "Ничего не отменилось из-за ошибки, нажми Undo ещё раз", "changed": "#{id} изменилась после, не трогала",
           "second": "Задача Second #{ref} остаётся открытой, закрой её там"},
    "en": {"already": "Already undone", "done": "Undone.", "failed": "Some changes were not undone, press Undo again",
           "none": "Nothing was undone because of an error, press Undo again", "changed": "#{id} changed since, left as is",
           "second": "Second #{ref} stays open, close it there"},
}


def _unchanged(item: dict | None, w: dict) -> bool:
    """Whether the item still looks as this write left it (compared on the item columns `after`
    names). A write with no `after` and no created item never restores blindly."""
    after = w["after"]
    if not after:
        return bool(w["before"].get("created"))
    if item is None:
        return True
    return all(item.get(k) == v for k, v in after.items() if k in ITEM_COLUMNS)


MEMORY_WRITE_TOOLS = ("remember", "forget")
NOT_AN_ANSWER_TOOLS = (*MEMORY_WRITE_TOOLS, "set_review", "set_digest")  # undoing these is not a wrong answer


def _restore_memory(store: Store, w: dict, now: datetime) -> None:
    """Revert one memory write. A fact already in the target state counts as done; a restore
    ignores the section bounds and never duplicates a live fact with the same words."""
    memory_id = w["before"].get("memory_id")
    fact = store.memory_get(memory_id) if isinstance(memory_id, int) else None
    if fact is None:
        return
    op = w["before"].get("op") or w["tool"]
    if op == "remember":
        store.memory_delete(memory_id, now)  # False when it was forgotten since: done either way
    elif fact["deleted_at"] is not None:
        live = store.memory_list()
        if not any(f["section"] == fact["section"] and f["text"] == fact["text"] for f in live):
            store.memory_restore(memory_id, now)


def _restore(store: Store, w: dict) -> None:
    """Revert one agent write from its `before`: item columns, reminders, or a created item."""
    before, item_id = w["before"], w.get("item_id")
    if item_id is None:
        return
    if before.get("created"):
        store.update_item(item_id, status="dismissed", route_error=None)  # also leaves the outbox
        store.disable_reminders(item_id)
        store.add_event(item_id, "undone", {})
        return
    if before.get("item"):
        store.update_item(item_id, **before["item"])
    if "reminders" in before:
        store.restore_reminders(item_id, before["reminders"])
    store.add_event(item_id, "undone", {"tool": w["tool"]})


async def _clear_decision(deps: Deps, item_id: int | None, intake: int, previous: str | None = None) -> None:
    """Undo of a vacancy decision: restore the previous decision (a second decision undone) or clear it
    (the first one). When Second cannot be reached the poll loop retries (pending = that decision or "clear")."""
    restore = previous if previous in ("apply", "skip") else None
    try:
        reply = await deps.second.patch_job(intake, restore, None)  # the worker restores the card
        reopen_if_requeued(deps, item_id, restore, reply)
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("job %s decision not cleared: %s", item_id, type(exc).__name__)
    if item_id is not None:
        fields = dict((deps.store.get_item(item_id) or {}).get("fields") or {})
        fields["job_decision_pending"] = restore or "clear"
        if restore:
            fields["job_decision_note"] = ""  # the restored decision carries no note (and not the Undo note)
        deps.store.update_item(item_id, fields=fields)


MEMOK_TEXT = {
    "ru": {"handled": "Уже обработано", "expired": "Предложение устарело, скажи ещё раз", "stored": "Запомнила",
           "line": "Запомнила (M{id}): {text}"},
    "en": {"handled": "Already handled", "expired": "The proposal is out of date, say it again", "stored": "Remembered",
           "line": "Remembered (M{id}): {text}"},
}


def _other_rows(callback: CallbackQuery, pressed: str) -> list[list[tuple[str, str]]]:
    """The keyboard of the message the button sits on, minus the row holding the pressed button."""
    rows = getattr(getattr(callback.message, "reply_markup", None), "inline_keyboard", None) or []
    kept = [[(b.text, b.callback_data) for b in row] for row in rows]
    return [row for row in kept if not any(data == pressed for _, data in row)]


def _proposal_open(store: Store, data: str) -> bool:
    pid = data[6:]
    row = store.pending_get(int(pid)) if pid.isdigit() and int(pid) <= MAX_ID else None
    return row is not None and row["status"] == "open"


def _open_proposals(callback: CallbackQuery, store: Store) -> tuple[list[list[tuple[str, str]]], list[str]]:
    """The memok rows still on the message whose proposal is still open, and the proposal lines they confirm.
    Proposals are not writes, so an Undo keeps them. Call under the drain lock."""
    rows = [r for r in _other_rows(callback, "")
            if any(data.startswith("memok:") and _proposal_open(store, data) for _, data in r)]
    if not rows:
        return [], []
    nums = {re.search(r"(\d+)\s*$", label).group(1) for r in rows for label, data in r
            if data.startswith("memok:") and re.search(r"(\d+)\s*$", label)}
    lines = []
    for ln in (getattr(callback.message, "text", None) or "").split("\n"):
        if not _is_proposal_line(ln) or _CONFIRM_LINE.match(ln):
            continue
        m = re.search(r"\(\s*(\d+)", ln)
        if not nums or (m and m.group(1) in nums):  # unnumbered buttons: keep every proposal line
            lines.append(ln)
    return rows, lines


async def on_memok(callback: CallbackQuery, deps: Deps) -> None:
    """The Remember button under an agent reply: store the proposed fact once, no model call.
    The bound is checked now (it may have changed since the proposal); a refusal leaves the proposal open."""
    _touch(deps)
    parts = _parse(callback.data, "memok", 2)
    pending_id = int(parts[1]) if parts and int(parts[1]) <= MAX_ID else None
    user = getattr(callback, "from_user", None)
    words = MEMOK_TEXT["ru" if (chat_language(deps, user.id) if user is not None else "ru") == "ru" else "en"]
    if pending_id is None:
        await callback.answer()
        return
    store, now = deps.store, deps.now_fn()
    rest = _other_rows(callback, callback.data)
    original = getattr(callback.message, "text", None) or ""
    edit: tuple[str, list[list[tuple[str, str]]]] | None = None
    async with deps.drain_lock:  # a status write must not interleave with a drain; no network inside
        pending = store.pending_get(pending_id)
        if pending is None or pending["status"] != "open":
            toast = words["handled"]
        elif now - datetime.fromisoformat(pending["created_at"]) > PENDING_TTL:
            store.pending_close(pending_id, "expired", None, now)
            edit, toast = (fit_edit(original, words["expired"]), rest), words["expired"]
        else:
            cleaned, reason = check_add(store, pending["section"], pending["text"])
            if cleaned is None:  # full, or the same words are already a live fact: the proposal stays open
                toast = reason or words["handled"]
            else:
                memory_id = store.memory_add(pending["section"], cleaned, source="button",
                                             turn_id=pending["turn_id"], now=now)
                if not store.pending_close(pending_id, "stored", memory_id, now):  # closed meanwhile: never keep a duplicate
                    store.memory_delete(memory_id, now)
                    toast = words["handled"]
                else:
                    store.add_agent_write(pending["turn_id"], None, "remember",
                                          {"memory_id": memory_id, "op": "remember"}, {"memory_id": memory_id})
                    undo = undo_turn_keyboard(pending["turn_id"])
                    if not any(data == undo[0][0][1] for row in rest for _, data in row):  # every memory write stays undoable
                        turn_items = {w.get("item_id") for w in store.agent_writes(pending["turn_id"]) if not w["undone"]}
                        item_undo = [r for r in rest if any(data.startswith("undo:") for _, data in r)]
                        mine = [r for r in item_undo if any(data[5:].isdigit() and int(data[5:]) in turn_items
                                                            for _, data in r if data.startswith("undo:"))]
                        if mine:  # the item Undo is a write of this turn: one Undo reverts everything
                            rest = [r for r in rest if r not in mine] + undo
                        elif item_undo:  # an item Undo from outside the turn stays; the added one says what it undoes
                            label = "Отменить запоминание" if words is MEMOK_TEXT["ru"] else "Undo memory"
                            rest = [*rest, [(label, undo[0][0][1])]]
                        else:
                            rest = [*rest, *undo]
                    edit, toast = (fit_edit(original, words["line"].format(id=memory_id, text=cleaned)), rest), words["stored"]
    if edit is not None:
        await _edit(callback, *edit)
    await callback.answer(toast)


async def on_undo_turn(callback: CallbackQuery, deps: Deps) -> None:
    """The Undo button of an agent reply: revert every write of the turn, newest first. No model call.
    Each reverted write is marked at once, so a second press retries only what failed. A Second task
    that was already created stays (the edit says so)."""
    _touch(deps)
    parts = _parse(callback.data, "undoturn", 2)
    turn_id = int(parts[1]) if parts and int(parts[1]) <= MAX_ID else None
    if turn_id is None:
        await callback.answer()
        return
    user = getattr(callback, "from_user", None)
    lang = chat_language(deps, user.id) if user is not None else "ru"
    words = UNDO_TEXT["ru" if lang == "ru" else "en"]
    ids: list[int] = []
    notes: list[str] = []
    intakes: list[tuple[int | None, int, str | None]] = []
    failed_items: set[int | None] = set()
    skips: list[tuple[int, int]] = []
    failed = False
    async with deps.drain_lock:  # a status write must not interleave with a drain; no network inside
        writes = deps.store.agent_writes(turn_id)
        if not writes:
            await callback.answer()
            return
        pending = [w for w in writes if not w["undone"]]
        if not pending:
            await callback.answer(words["already"])
            return
        for w in reversed(pending):
            if w["tool"] == "wait_for_call":  # a wait has no item: cancel it only if it is still open
                try:
                    deps.store.wait_cancel(int((w["before"] or {}).get("wait_id")))
                    deps.store.mark_write_undone(w["id"])
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    log.warning("wait write %s of turn %s not undone: %s", w["id"], turn_id, type(exc).__name__)
                continue
            if w["tool"] == "cancel_call_wait":  # reopen each cancelled wait that has not expired
                try:
                    for wid in (w["before"] or {}).get("wait_ids") or []:
                        deps.store.wait_reopen(int(wid), deps.now_fn())
                    deps.store.mark_write_undone(w["id"])
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    log.warning("wait write %s of turn %s not undone: %s", w["id"], turn_id, type(exc).__name__)
                continue
            if w["tool"] == "set_digest":  # digest settings live in meta: put the old values back
                try:
                    restore_digest(deps.store, w["before"] or {})
                    deps.store.mark_write_undone(w["id"])
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    log.warning("digest write %s of turn %s not undone: %s", w["id"], turn_id, type(exc).__name__)
                continue
            if w["tool"] == "set_review":  # the review switch lives in meta: put the old value back
                try:
                    old = (w["before"] or {}).get("review_enabled")
                    deps.store.meta_set("review_enabled", old if isinstance(old, str) else "")
                    deps.store.mark_write_undone(w["id"])
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    log.warning("review write %s of turn %s not undone: %s", w["id"], turn_id, type(exc).__name__)
                continue
            if w["tool"] in MEMORY_WRITE_TOOLS:  # a memory write has no item: nothing to compare, nothing to list
                try:
                    _restore_memory(deps.store, w, deps.now_fn())
                    deps.store.mark_write_undone(w["id"])
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    log.warning("memory write %s of turn %s not undone: %s", w["id"], turn_id, type(exc).__name__)
                continue
            if w.get("item_id") is not None and w["item_id"] in failed_items:
                continue  # an older write on an item whose newer write failed waits for the retry
            try:
                item = deps.store.get_item(w["item_id"]) if w.get("item_id") is not None else None
                later = (w["before"].get("created") and ((item or {}).get("fields") or {}).get("job_intake_id")
                         and ((item or {}).get("fields") or {}).get("job_decision"))  # decided after this turn
                if later or not _unchanged(item, w):
                    # the item moved on since this write: leave it, and do nothing else for this write
                    notes.append(words["changed"].format(id=w["item_id"]))
                    deps.store.mark_write_undone(w["id"])
                    continue
                _restore(deps.store, w)
                deps.store.mark_write_undone(w["id"])
                intake = ((item or {}).get("fields") or {}).get("job_intake_id")
                if w["before"].get("created") and intake:  # a vacancy created by save_note: Second hears "skip" too
                    skips.append((w["item_id"], int(intake)))
            except Exception as exc:  # noqa: BLE001 - revert the rest, say what failed
                failed = True
                failed_items.add(w.get("item_id"))
                log.warning("agent write %s of turn %s not undone: %s", w["id"], turn_id, type(exc).__name__)
                continue
            if w.get("item_id") is not None and w["item_id"] not in ids:
                ids.append(w["item_id"])
            if w["tool"] == "decide_vacancy" and w["before"].get("intake_id") is not None:
                prior = (((w["before"].get("item") or {}).get("fields")) or {}).get("job_decision")
                intakes.append((w.get("item_id"), int(w["before"]["intake_id"]), prior))
            ref = second_task_ref(item) if item is not None else None
            if ref and w["tool"] in SECOND_STAYS:
                notes.append(words["second"].format(ref=ref))
    for item_id, intake in skips:
        await _skip_job(deps, item_id, intake)
    skipped = {item_id for item_id, _ in skips}
    for item_id, intake, prior in intakes:
        if item_id in skipped:
            continue  # the skip is the final decision: a clear would reopen the item
        await _clear_decision(deps, item_id, intake, prior)
    if failed and not ids:
        text = words["none"]
    else:
        text = " ".join(fallback_undone(i, lang) for i in ids) or ("" if notes else words["done"])
        if failed:
            text += "\n" + words["failed"]
    if any(w["tool"] not in NOT_AN_ANSWER_TOOLS for w in pending) and not failed:
        try:  # an undone item write is a miss for the weekly review; never breaks the Undo
            turn_row = deps.store.turn_get(turn_id)
            deps.store.miss_add(turn_row["chat_id"] if turn_row else 0, turn_id, "отмена ответа",
                                ((turn_row or {}).get("text") or "").strip() or "(без текста)", "undone", deps.now_fn())
        except Exception as exc:  # noqa: BLE001
            log.warning("undo miss of turn %s not logged: %s", turn_id, type(exc).__name__)
    async with deps.drain_lock:
        kept_rows, kept_lines = _open_proposals(callback, deps.store)  # an open memory proposal is not a write: Undo keeps it
    await _edit(callback, fit_reply("\n".join(p for p in [text, *dict.fromkeys(notes)] if p), kept_lines),
                kept_rows + (undo_turn_keyboard(turn_id) if failed else []) or None)
    await callback.answer()


CALLWAIT_TEXT = {"ru": "Отменено", "en": "Cancelled"}
CALLWAIT_CLOSED_TEXT = {"ru": "Уже отвечено или закрыто", "en": "Already answered or closed"}


async def on_callwait_cancel(callback: CallbackQuery, deps: Deps) -> None:
    """The Cancel button under a reply that opened a call wait: stop waiting (only if still open). No model call."""
    _touch(deps)
    parts = (callback.data or "").split(":")
    ok = len(parts) == 3 and parts[:2] == ["callwait", "cancel"] and parts[2].isascii() and parts[2].isdigit()
    wait_id = int(parts[2]) if ok and int(parts[2]) <= MAX_ID else None
    user = getattr(callback, "from_user", None)
    lang = chat_language(deps, user.id) if user is not None else "ru"
    cancelled = False
    if wait_id is not None:
        async with deps.drain_lock:
            cancelled = deps.store.wait_cancel(wait_id)
    await callback.answer((CALLWAIT_TEXT if cancelled else CALLWAIT_CLOSED_TEXT)["ru" if lang == "ru" else "en"])


FOCUSREQ_TEXT = {"ru": "Запрос отправлен; ноутбук соберёт карту, когда проснётся.",
                 "en": "Request sent; the laptop builds the map when it wakes."}
FOCUSREQ_EXISTS = {"ru": "Карта дня на сегодня уже есть.", "en": "Today's map already exists."}


async def on_focusreq(callback: CallbackQuery, deps: Deps) -> None:
    """The "Build on laptop" button under a digest without a day map: leave one request for the laptop."""
    _touch(deps)
    if (callback.data or "") != "focusreq:open":
        await callback.answer()
        return
    user = getattr(callback, "from_user", None)
    lang = chat_language(deps, user.id) if user is not None else "ru"
    now = deps.now_fn()
    folder = getattr(deps.settings, "focus_dir", None)
    if folder is not None and today_map(folder, now.astimezone(deps.calendar.tz).date()) is not None:
        await callback.answer(FOCUSREQ_EXISTS["ru" if lang == "ru" else "en"])
        return
    async with deps.drain_lock:
        deps.store.focus_request_open(now)
    await callback.answer(FOCUSREQ_TEXT["ru" if lang == "ru" else "en"])


OLD_BUTTON = "This button is from an older version"
OLD_PREFIXES = ("retag:", "delete:", "remind:", "roff:")


async def on_old_button(callback: CallbackQuery, deps: Deps) -> None:
    """Phase 1 category, Delete and ask-when buttons: change nothing and take the keyboard away."""
    _touch(deps)
    await callback.answer(OLD_BUTTON)
    drop = getattr(callback.message, "edit_reply_markup", None)
    if drop is None:
        return
    try:
        await drop(reply_markup=None)
    except Exception as exc:  # noqa: BLE001 - e.g. the message is too old to edit
        log.info("edit_reply_markup failed: %s", type(exc).__name__)


MISS_TEXT = {"ru": {"handled": "Уже разобрано", "missbl": "В бэклоге", "missno": "Убрала"},
             "en": {"handled": "Already handled", "missbl": "In backlog", "missno": "Removed"}}


async def on_miss_button(callback: CallbackQuery, deps: Deps) -> None:
    """A topic button under the Friday review: close every open miss of that topic, drop its keyboard row."""
    _touch(deps)
    data = callback.data or ""
    kind = data.split(":")[0]
    parts = _parse(data, kind, 2) if kind in ("missbl", "missno") else None
    if parts is None or int(parts[1]) > MAX_ID:
        await callback.answer()
        return
    user = getattr(callback, "from_user", None)
    words = MISS_TEXT["ru" if (chat_language(deps, user.id) if user is not None else "ru") == "ru" else "en"]
    async with deps.drain_lock:
        closed = deps.store.misses_close_topic(int(parts[1]), "backlog" if kind == "missbl" else "dismissed", deps.now_fn(),
                                               until=getattr(callback.message, "date", None))
    await callback.answer(words[kind] if closed else words["handled"])
    if not closed:
        return
    rest = _other_rows(callback, data)
    drop = getattr(callback.message, "edit_reply_markup", None)
    if drop is None:
        return
    try:
        await drop(reply_markup=markup(rest) if rest else None)
    except Exception as exc:  # noqa: BLE001 - e.g. "message is not modified"
        log.info("edit_reply_markup failed: %s", type(exc).__name__)


# -- reminders ----------------------------------------------------------


async def on_rdone(callback: CallbackQuery, deps: Deps) -> None:
    _touch(deps)
    parts = _parse(callback.data, "rdone", 2)
    text = None
    async with deps.drain_lock:  # a status write must not interleave with a drain
        if parts and find_item(deps.store, int(parts[1])) is not None:
            text = handle_done(deps.store, parts[1])
            deps.store.disable_reminders(int(parts[1]))
    if text is not None:
        await _edit(callback, text, None)
    await callback.answer()


async def on_rsnooze(callback: CallbackQuery, deps: Deps) -> None:
    _touch(deps)
    parts = _parse(callback.data, "rsnooze", 3)
    rem = deps.store.get_reminder(int(parts[1])) if parts and parts[2] in SNOOZES and int(parts[1]) <= MAX_ID else None
    item = deps.store.get_item(rem["item_id"]) if rem else None
    if item is not None and item["status"] not in CLOSED:
        until = deps.now_fn() + SNOOZES[parts[2]]
        deps.store.snooze_reminder(rem["id"], until)  # also re-enables a reminder turned off earlier
        fields = item.get("fields") or {}
        if fields.get("reminders_off"):
            fields = {k: v for k, v in fields.items() if k != "reminders_off"}
            fields["reminders"] = list(fields.get("reminders") or []) + [utc_iso(until)]
            deps.store.update_item(item["id"], fields=fields)
            deps.store.add_event(item["id"], "reminders_on", {"snoozed": parts[2]})
        title = " ".join((item.get("title") or "").split()) or "(no title)"
        await _edit(callback, f"Snoozed #{item['id']} · {title} until {fmt_full(until, deps.calendar.tz)}", None)
    await callback.answer()


async def on_cancel(message: Message, deps: Deps) -> None:
    """Phase 1 ended a date wait with /cancel; there is no wait any more."""
    _touch(deps)
    await message.answer("Nothing to cancel")


# -- commands -----------------------------------------------------------

def _args(message: Message) -> str:
    parts = (getattr(message, "text", None) or "").split(None, 1)
    return parts[1] if len(parts) > 1 else ""


async def on_help(message: Message, deps: Deps) -> None:
    _touch(deps)
    chat = getattr(message, "chat", None) or message.from_user  # private chat: chat id == user id
    turns = deps.store.recent_turns(chat.id, limit=20)
    await message.answer(greeting([t["text"] for t in turns if t.get("role") == "user"]))


on_start = on_help


def _chunks(text: str, limit: int = 4000) -> list[str]:
    """Split on line breaks so that every part fits one Telegram message."""
    parts, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + 1 + len(line) > limit:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    return [*parts, current]


async def on_memory(message: Message, deps: Deps) -> None:
    """/memory: the live facts, no model call."""
    _touch(deps)
    lang = chat_language(deps, message.from_user.id)
    for part in _chunks(render_memory(deps.store.memory_list(), lang)):
        await message.answer(part)


async def on_list(message: Message, deps: Deps) -> None:
    _touch(deps)
    parsed = parse_list_args(_args(message))
    if parsed is None:
        await message.answer(UNKNOWN_SHELF)
        return
    kind, include_all, page = parsed
    items = deps.store.list_items(kind=kind, status=None if include_all else "new", limit=1_000_000)
    await message.answer(render_list(items, kind, page, show_status=include_all))


async def _close(message: Message, deps: Deps, handler: Any) -> None:
    _touch(deps)
    async with deps.drain_lock:  # a status write must not interleave with a drain
        text = handler(deps.store, _args(message))
    await message.answer(text)


async def on_done(message: Message, deps: Deps) -> None:
    await _close(message, deps, handle_done)


async def on_drop(message: Message, deps: Deps) -> None:
    await _close(message, deps, handle_drop)


async def on_show(message: Message, deps: Deps) -> None:
    _touch(deps)
    item_id = parse_id(_args(message))
    if item_id is None:
        await message.answer("Usage: /show <id>")
        return
    item = find_item(deps.store, item_id)
    await message.answer(render_show(item, deps.settings.timezone) if item else f"No item #{item_id}")


async def on_inbox(message: Message, deps: Deps) -> None:
    _touch(deps)
    now = deps.now_fn()
    await message.answer(render_inbox(deps.store, deps.context_cache.age_seconds(now), now, deps.settings.timezone))


def _chat_lang(deps: Deps, message: Message) -> str:
    return "ru" if chat_language(deps, message.from_user.id) == "ru" else "en"


async def on_sync(message: Message, deps: Deps) -> None:
    _touch(deps)
    await message.answer(await sync_now(deps, _chat_lang(deps, message)))


async def on_flush(message: Message, deps: Deps) -> None:
    _touch(deps)
    await message.answer(await flush_now(deps, _chat_lang(deps, message)))


def unknown_command(message: Any) -> str | None:
    """The `/cmd` text when the message starts with a bot_command entity at offset 0, else None."""
    if getattr(message, "forward_origin", None) is not None:  # a forward is always a capture
        return None
    entities = getattr(message, "entities", None) or []
    text = getattr(message, "text", None) or ""
    if not entities or getattr(entities[0], "type", None) != "bot_command" or entities[0].offset != 0:
        return None
    return text[: entities[0].length].split("@", 1)[0]


async def on_unknown(message: Message, deps: Deps) -> None:
    _touch(deps)
    await message.answer(f"Unknown command {unknown_command(message)}. /help lists them")


def _bind(handler: Any, deps: Deps) -> Any:
    async def command(message: Message) -> None:
        await handler(message, deps)

    command.__name__ = f"_{handler.__name__}"
    return command


def _bind_callback(handler: Any, deps: Deps) -> Any:
    async def callback(query: CallbackQuery) -> None:
        await handler(query, deps)

    callback.__name__ = f"_{handler.__name__}"
    return callback


def build_router(deps: Deps) -> Router:
    """Order: forwards and media (always captures), slash commands (no model call), unknown slash
    commands, plain text (conversation engine), anything else (a short "can't save this" reply)."""
    router = Router(name="clawdia")
    owner = OwnerOnly(deps.settings.allowed_owner_ids)
    router.message.filter(owner)
    router.callback_query.filter(owner)

    # Forwards and media are always captures, even when their text or caption starts with a
    # command (aiogram's Command filter also reads captions).
    router.message.register(_bind(on_message, deps), lambda m: getattr(m, "forward_origin", None) is not None)
    router.message.register(_bind(on_message, deps), is_media)

    for name, handler in (("start", on_start), ("help", on_help), ("list", on_list), ("done", on_done),
                          ("drop", on_drop), ("show", on_show), ("inbox", on_inbox), ("sync", on_sync),
                          ("flush", on_flush), ("cancel", on_cancel), ("memory", on_memory)):
        router.message.register(_bind(handler, deps), Command(name))

    router.message.register(_bind(on_unknown, deps), lambda m: unknown_command(m) is not None)
    router.message.register(_bind(on_text, deps), lambda m: bool(getattr(m, "text", None)))
    router.message.register(_bind(on_unsupported, deps))

    for prefix, handler in (("undo:", on_undo), ("undoturn:", on_undo_turn), ("memok:", on_memok), ("callwait:", on_callwait_cancel), ("focusreq:", on_focusreq), ("missbl:", on_miss_button), ("missno:", on_miss_button), ("rdone:", on_rdone), ("rsnooze:", on_rsnooze)):
        router.callback_query.register(_bind_callback(handler, deps), lambda c, p=prefix: (c.data or "").startswith(p))
    router.callback_query.register(_bind_callback(on_old_button, deps),
                                   lambda c: (c.data or "").startswith(OLD_PREFIXES))
    return router


# -- process wiring -----------------------------------------------------


def make_notify(bot: Bot, store: Store, owner_id: int):
    """Tell the primary owner about an item filed in the background (the capture reply covers its own item)."""
    async def notify(item_id: int, second_id: int, project: str) -> None:
        item = store.get_item(item_id) or {}
        title = " ".join((item.get("title") or "").split()) or "(no title)"
        where = f" ({project})" if project else ""
        text = f"#{item_id} · {title}\n-> Second #{second_id}{where}"
        await bot.send_message(owner_id, text)
        record_bot_turn(store, owner_id, text, item_id)  # the owner's private chat id is her user id
    return notify


def _ai_client(settings: Settings) -> Any:
    if settings.tag_provider == "openai":
        import openai
        return openai.AsyncOpenAI(api_key=settings.openai_api_key)
    import anthropic
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def serve(settings: Settings) -> None:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.state_dir / "clawdia.sqlite")
    store.ensure_schema()
    bot = Bot(token=settings.bot_token)
    second = SecondClient(settings.second_api_url, settings.second_api_token)
    ai_client = _ai_client(settings)
    deps = Deps(
        store=store, settings=settings, context_cache=ContextCache(settings.state_dir / "context.json"),
        second=second, calendar=WorkCalendar(settings.work_days, settings.timezone), bot=bot, tag_fn=tag,
        transcribe_fn=transcribe.transcribe, save_media_fn=media.save, now_fn=lambda: datetime.now(timezone.utc),
        group_wait_s=3.0, ai_client=ai_client, notify=make_notify(bot, store, settings.owner_id))  # primary owner only
    deps.second_reader_fn = lambda: SecondReader(second)  # agent reads: second-api /v1/read, same token
    dp = Dispatcher()
    dp.include_router(build_router(deps))
    deps.refresh_context = lambda: refresh_context(deps)  # /sync reloads the context for real
    loops = None
    try:
        try:
            await refresh_context(deps)  # a fresh snapshot before the first capture; the cache serves if this fails
        except Exception as exc:  # noqa: BLE001
            log.warning("startup context refresh failed: %s", type(exc).__name__)
        try:
            store.waits_reset_answering(deps.now_fn(), older_than=None)  # nothing is in flight at startup: every claimed wait is retried
        except Exception as exc:  # noqa: BLE001
            log.warning("startup wait reset failed: %s", type(exc).__name__)
        await recover_unprocessed(deps)  # answers turns a restart cut off; never raises
        loops = asyncio.create_task(run_loops(deps))
        await dp.start_polling(bot)
    finally:
        if loops is not None:
            loops.cancel()
            await asyncio.gather(loops, return_exceptions=True)
        await second.aclose()
        await ai_client.close()
        await bot.session.close()
        store.close()


def run(settings: Settings) -> None:
    asyncio.run(serve(settings))
