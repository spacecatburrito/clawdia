"""Executors of the conversation engine: one per intent, each returning an `Outcome`.

`converse.handle_text` builds the context (`Ctx`) and the guarded understanding; the executors
act through the phase 1 paths (item creation + `apply_tag`, `queue_and_drain`, reminders,
`handle_done` / `handle_drop`, renderers) and phrase the result with one reply call. Status
writes run under `deps.drain_lock`; the reply call never does.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

from clawdia.capture import (Deps, Keyboard, _resolve_duplicate, _step, _submit_job, _task_line, apply_tag,
                             build_reply, extract, flush_now, is_duplicate, keyboard_for, queue_and_drain, sync_now)
from clawdia.context import Snapshot
from clawdia.kinds import KINDS, SECOND_TASK_KINDS
from clawdia.render import (HELP_TEXT, cap, find_item, handle_done, handle_drop, render_inbox, render_list,
                            render_show, second_note, second_task_ref, status_line)
from clawdia.reminders import ASK_WHEN, ASK_WHEN_RU, add_reminders, fmt_full, fmt_slot
from clawdia.reply import Facts, fallback_not_saved, fallback_undone
from clawdia.second_client import apply_pending_patches
from clawdia.storage import utc_iso
from clawdia.tagger import ERROR_MODEL, TITLE_MAX, TagResult, _error_result, match_project
from clawdia.understand import Understanding

log = logging.getLogger(__name__)

ASK_KIND = {"ru": "Это задача, идея или что-то другое?", "en": "Is this a task, an idea or something else?"}
CHAT_HELP = {
    "ru": "Я сохраняю всё, что ты пишешь. Поправить можно словами («это задача», «удали»), "
          "список попросить так: «покажи фильмы».",
    "en": "I save whatever you send. Fix things in words (\"make it a task\", \"delete that\") "
          "or ask for a list (\"show films\").",
}


@dataclass
class Outcome:
    text: str
    keyboard: Keyboard | None = None
    item_id: int | None = None
    source: str = "typed"  # the bot turn's source: "agent_data" when the reply may echo read or quoted content


@dataclass
class Ctx:
    deps: Deps
    message: Any
    text: str
    chat_id: int
    turn_id: int
    now: datetime
    u: Understanding
    recent: list[dict]
    history: list[dict]  # earlier turns of this chat, oldest first
    snapshot: Snapshot
    item_id: int | None = None  # item created by this turn, so a rescue never creates a second one
    model_calls: int = 0  # understand, tag and reply calls made for this message (logged, ids only)

    @property
    def lang(self) -> str:
        return "ru" if self.u.language == "ru" else "en"


def model_client(deps: Deps) -> Any:
    """The shared client when it is an Anthropic one; None lets the call build its own."""
    return deps.ai_client if getattr(deps.settings, "tag_provider", "anthropic") == "anthropic" else None


def unsorted_result(text: str, model: str = "") -> TagResult:
    return TagResult("unsorted", text[:TITLE_MAX], None, {}, 0.0, None, model)


def _title(item: dict) -> str:
    return " ".join((item.get("title") or "").split()) or "(no title)"


def _tz(ctx: Ctx) -> ZoneInfo:
    return ZoneInfo(ctx.deps.settings.timezone)


def _aware(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value) if value else None
    except ValueError:
        return None
    return parsed if parsed is not None and parsed.tzinfo is not None else None


def _reminder_times(ctx: Ctx, item: dict) -> list[str]:
    fields = item.get("fields") or {}
    if fields.get("reminders_off"):
        return []
    times = [_aware(s) for s in fields.get("reminders") or []]
    return [fmt_slot(t, _tz(ctx)) for t in times if t is not None and t > ctx.now]


def _facts(ctx: Ctx, item: dict, action: str, fallback: str, *, question: str | None = None,
           extra: list[str] | None = None) -> Facts:
    line = _task_line(item) if item["kind"] in SECOND_TASK_KINDS else None
    return Facts(language=ctx.u.language, action=action, item_id=item["id"], kind=item["kind"],
                 title=item.get("title"), second_line=line, reminder_times=_reminder_times(ctx, item),
                 question=question, extra=list(extra or []), fallback=fallback)


async def _say(ctx: Ctx, facts: Facts) -> str:
    """The reply call; skipped when the understanding call already failed (the model is likely
    down, so a second timeout would only delay the deterministic reply)."""
    if ctx.u.model == ERROR_MODEL:
        return facts.fallback
    ctx.model_calls += 1
    return await ctx.deps.reply_fn(facts, settings=ctx.deps.settings, client=model_client(ctx.deps))


def not_saved(ctx: Ctx, text: str) -> str:
    """A reply that saved nothing ends with the not-saved marker, so "save that" can rescue a misread note."""
    marker = fallback_not_saved(ctx.lang)
    if text.rstrip().rstrip(".!").lower().endswith(marker.rstrip(".").lower()):
        return text
    return f"{text.rstrip()} {marker}".strip()


def _which(ctx: Ctx, verb: str | None = None) -> Outcome:
    """No usable target: name the missing explicit id, or ask which item (recent ones as hints)."""
    u = ctx.u
    if u.explicit_id is not None:
        text = f"Нет записи #{u.explicit_id}." if ctx.lang == "ru" else f"No item #{u.explicit_id}."
    else:
        hints = ", ".join(f"#{i['id']} {_title(i)}" for i in ctx.recent[:3])
        text = "Какую запись?" if ctx.lang == "ru" else "Which item?"
        if verb:
            text += f" Send {verb} #id." if ctx.lang == "en" else f" Напиши {verb} #id."
        if hints:
            text += f" {hints}"
    return Outcome(not_saved(ctx, text), None, None)


# -- capture ------------------------------------------------------------


def _create(ctx: Ctx, text: str, message: Any) -> int:
    data = extract(message)
    item_id = ctx.deps.store.create_item(
        status="new", kind="unsorted", text=text, urls=data["urls"] or None, chat_id=ctx.chat_id,
        tg_message_id=data["tg_message_id"], tg_date=data["tg_date"])
    ctx.item_id = item_id
    return item_id


def _question(ctx: Ctx, item_id: int, result: TagResult | None, ask: str | None) -> str | None:
    """At most one question per item: a kind question below the threshold (the model's, else a
    fixed one; never after a failed model call), else "when?" for a timeless request. Without
    `result` the item's stored confidence and model are used (a media or forward capture)."""
    store = ctx.deps.store
    item = store.get_item(item_id)
    fields = dict(item.get("fields") or {})
    if fields.get("asked"):
        return None
    confidence = result.confidence if result is not None else item.get("confidence")
    model = result.model if result is not None else item.get("model")
    low = confidence is not None and confidence < ctx.deps.settings.ask_below and model != ERROR_MODEL
    question = None
    if low:
        question = ask or ASK_KIND[ctx.lang]
    elif fields.get("ask_when"):
        question = ASK_WHEN_RU if ctx.lang == "ru" else ASK_WHEN
    if question:
        fields["asked"] = True
        store.update_item(item_id, fields=fields)
    return question


async def saved_outcome(ctx: Ctx, item_id: int, question: str | None = None, *, use_model: bool = True) -> Outcome:
    item = ctx.deps.store.get_item(item_id)
    if is_duplicate(item):  # merged into an earlier copy: a fixed reply, no question, no model
        question, use_model = None, False
    fallback = build_reply(item, ctx.deps.calendar, ctx.now, ctx.lang)
    if question:  # the reply's "when?" line gives way to the one question
        fallback = "\n".join(line for line in fallback.splitlines() if line not in (ASK_WHEN, ASK_WHEN_RU))
        if question not in fallback:
            fallback += f"\n{question}"
    text = await _say(ctx, _facts(ctx, item, "saved", fallback, question=question)) if use_model else fallback
    return Outcome(text, keyboard_for(item), item_id)


async def _tag_and_save(ctx: Ctx, text: str, message: Any, *, on_created: Callable[[int], None] | None = None,
                        bound: Callable[[], float | None] | None = None) -> int:
    """Create the item and tag it with `deps.tag_fn`: the message's second model call, so the
    caller replies deterministically. `on_created(item_id)` runs right after the insert (before any
    await), so a caller can record the write even if the rest is cancelled. `bound()` gives the
    seconds a network step may still take (None: unbounded); a step that overruns is a failed step."""
    deps = ctx.deps
    item_id = _create(ctx, text, message)
    if on_created is not None:
        on_created(item_id)

    async def limited(awaitable: Any) -> Any:
        limit = bound() if bound is not None else None
        return await (awaitable if limit is None else asyncio.wait_for(awaitable, limit))

    item = deps.store.get_item(item_id)
    ctx.model_calls += 1
    result = await _step(deps, item_id, "tag", limited(deps.tag_fn(
        item, ctx.snapshot, deps.settings, image_path=None, now=ctx.now, client=deps.ai_client)))
    await _step(deps, item_id, "tag", limited(apply_tag(deps, item_id, result or _error_result(item), ctx.now)))
    return item_id


async def do_capture(ctx: Ctx) -> Outcome:
    u = ctx.u
    if u.capture is None:  # a downgraded intent: the understanding call gave no capture to apply
        item_id = await _tag_and_save(ctx, ctx.text, ctx.message)
        # the stored tag confidence and model decide the question (a failed tag asks nothing)
        return await saved_outcome(ctx, item_id, _question(ctx, item_id, None, None), use_model=False)
    item_id = _create(ctx, ctx.text, ctx.message)
    await _step(ctx.deps, item_id, "tag", apply_tag(ctx.deps, item_id, u.capture, ctx.now))
    return await saved_outcome(ctx, item_id, _question(ctx, item_id, u.capture, u.ask))


MEDIA_MARKERS = ("[photo]", "[voice]", "[document]", "[media]", "[forward]")


def _captured_by(item: dict, turn_text: str) -> bool:
    """Whether a user turn is the message that created `item`: its text or caption, or the marker
    a media or forward turn without words carries (see `converse.handle_capture`)."""
    return turn_text in (item.get("text"), item.get("caption")) or turn_text in MEDIA_MARKERS


async def _save_previous(ctx: Ctx) -> Outcome:
    """Elena's previous message in this chat becomes a capture. The tag call is this message's second
    model call, so the reply is the deterministic one."""
    deps, store = ctx.deps, ctx.deps.store
    prev = next((t for t in reversed(ctx.history) if t["role"] == "user"), None)
    if prev is None or not (prev.get("text") or "").strip():
        return Outcome(not_saved(ctx, "Нечего сохранить." if ctx.lang == "ru" else "Nothing to save."), None, None)
    done = find_item(store, prev["item_id"]) if prev.get("item_id") else None
    if done is not None and _captured_by(done, prev["text"]) and done["status"] != "dismissed":
        text = f"Уже сохранила #{done['id']}." if ctx.lang == "ru" else f"Already saved #{done['id']}."
        return Outcome(text, None, done["id"])  # Undo stays on the capture reply itself (spec 6.6)
    item_id = await _tag_and_save(ctx, prev["text"], SimpleNamespace(text=prev["text"]))
    store.mark_turn(prev["id"], processed=True, item_id=item_id)
    return await saved_outcome(ctx, item_id, use_model=False)


# -- shared item edits (caller holds the drain lock) ----------------------


def _clear_reminder_fields(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if k not in ("reminders", "ask_when", "reminders_off")}


def _reminders_off(ctx: Ctx, item_id: int) -> None:
    store = ctx.deps.store
    store.disable_reminders(item_id)
    fields = _clear_reminder_fields(store.get_item(item_id).get("fields") or {})
    fields["reminders_off"] = True
    store.update_item(item_id, fields=fields)
    store.add_event(item_id, "reminders_off", {})


def _remind_at(ctx: Ctx, item_id: int, raw: str) -> datetime | None:
    """Exactly this reminder: earlier ones are disabled. None when the time is unusable or past."""
    at = _aware(raw)
    if at is None or at <= ctx.now:
        return None
    store = ctx.deps.store
    store.disable_reminders(item_id)
    store.update_item(item_id, fields=_clear_reminder_fields(store.get_item(item_id).get("fields") or {}))
    add_reminders(store, store.get_item(item_id), [at])
    return at


def _snooze(ctx: Ctx, item_id: int, days: int) -> datetime:
    store = ctx.deps.store
    until = ctx.now + timedelta(days=days)
    rem = store.latest_reminder(item_id)
    if rem is None:
        add_reminders(store, store.get_item(item_id), [until])
        return until
    store.snooze_reminder(rem["id"], until)  # also re-enables it
    fields = dict(store.get_item(item_id).get("fields") or {})
    fields.pop("reminders_off", None)
    fields["reminders"] = [s for s in fields.get("reminders") or [] if s != rem["at"]] + [utc_iso(until)]
    store.update_item(item_id, fields=fields)
    store.add_event(item_id, "snoozed", {"days": days})
    return until


def _deadline_note(value: str) -> str:
    try:
        return f"Deadline: {date.fromisoformat(value):%a} {value}"
    except ValueError:
        return f"Deadline: {value}"


# -- correct ------------------------------------------------------------


def _patch_note(outcome: str, second_id: str, lang: str) -> str:
    """What Second did with a correction, from `apply_pending_patches` outcomes (never assumed)."""
    if outcome == "patched":
        return f"Обновила Second #{second_id}" if lang == "ru" else f"Second #{second_id} updated"
    if outcome.startswith("rejected:"):
        code = outcome.split(":", 1)[1]
        return f"Second не принял изменение ({code})" if lang == "ru" else f"Second did not accept the change ({code})"
    return ("Second получит изменение, когда будет доступен" if lang == "ru"
            else "Second will get it when it is reachable")


async def _unchanged(ctx: Ctx, item: dict, reason: str | None = None) -> Outcome:
    """Nothing was changed: say so, no Undo, and end with the not-saved marker so "save that" can rescue a note."""
    line = f"Nothing changed for #{item['id']} · {_title(item)}"
    if reason:
        line = f"{reason}. {line}"
    text = await _say(ctx, _facts(ctx, item, "unchanged", line, extra=[line]))
    return Outcome(not_saved(ctx, text), None, None)


def unknown_project(u: Understanding, snapshot: Snapshot) -> str | None:
    """The project name the owner gave that Second does not have (Second never creates projects), else None."""
    if u.new_project is not None and match_project(u.new_project, snapshot) is None:
        return u.new_project
    if u.new_project is None and u.new_project_raw:
        return u.new_project_raw
    return None


def _project_refusal(ctx: Ctx) -> Outcome | None:
    """Second never creates projects: an unknown name changes nothing and is named in the reply."""
    name = unknown_project(ctx.u, ctx.snapshot)
    if name is None:
        return None
    msg = f"Нет проекта {name}." if ctx.lang == "ru" else f"No project {name}."
    return Outcome(not_saved(ctx, msg), None, None)


async def correct_item(deps: Deps, item: dict, *, new_kind: str | None, deadline: str | None, project: str | None,
                       title: str | None = None, route: bool = False, now: datetime) -> tuple[bool, str | None]:
    """Apply a correction to one open item: retag, deadline, project (canonical name) and title; send a
    new work item to Second; queue a patch for a task Second already has and send it. The caller holds
    `deps.drain_lock`. Returns (changed, patch outcome from `apply_pending_patches` or None)."""
    store = deps.store
    item_id, old_kind, status = item["id"], item["kind"], item["status"]
    fields = dict(item.get("fields") or {})
    patch = dict(fields.get("second_patch") or {})
    filed_ref = second_task_ref(item) if status == "filed" else None
    cols: dict[str, Any] = {}
    if new_kind and new_kind != old_kind:
        cols["kind"] = new_kind
        store.add_event(item_id, "retag", {"from": old_kind, "to": new_kind})
        if old_kind in SECOND_TASK_KINDS and status == "routing" and new_kind not in SECOND_TASK_KINDS:
            cols.update(status="new", route_error=None)  # leave the Second outbox
    if deadline and fields.get("deadline") != deadline:
        fields["deadline"] = deadline
        if filed_ref:
            patch["deadline"] = deadline
    if project and project != item.get("project"):
        cols.update(project=project, confidence=1.0)  # the owner named it: route it as is
        if filed_ref:
            patch["project"] = project
    if title and title != item.get("title"):
        cols["title"] = title
    if patch:
        fields["second_patch"] = patch
    changed = False
    if cols or fields != (item.get("fields") or {}):
        store.update_item(item_id, fields=fields, **cols)
        changed = True
    fresh = store.get_item(item_id)
    if fresh["kind"] in SECOND_TASK_KINDS and fresh["status"] == "new" and (new_kind in SECOND_TASK_KINDS or route):
        await _step(deps, item_id, "route", queue_and_drain(deps, item_id, now))
        changed = True
    if new_kind == "job" and old_kind != "job" and fresh["kind"] == "job" and fresh["status"] in ("new", "routing"):
        # a retag to job submits the vacancy to triage like a fresh capture (R17: "same" duplicates are not sent)
        if _resolve_duplicate(deps, item_id, now) != "same":
            await _step(deps, item_id, "job", _submit_job(deps, item_id, now))
        changed = True
    patch_outcome = None
    if patch and filed_ref and changed:
        outcomes: dict[int, str] = {}
        try:
            await apply_pending_patches(store, deps.second, outcomes)
        except Exception as exc:  # noqa: BLE001 - the patch stays queued for the outbox tick
            log.warning("patch for item %s failed: %s", item_id, type(exc).__name__)
        patch_outcome = outcomes.get(item_id, "pending")
    return changed, patch_outcome


async def do_correct(ctx: Ctx) -> Outcome:
    u, deps, store = ctx.u, ctx.deps, ctx.deps.store
    if u.target_item_id is None:
        return _which(ctx)
    refusal = _project_refusal(ctx)
    if refusal is not None:
        return refusal
    project = match_project(u.new_project, ctx.snapshot) if u.new_project else None
    notes: list[str] = []
    action, patch_outcome, deleted, changed, reason = "updated", None, None, False, None
    async with deps.drain_lock:
        item = store.get_item(u.target_item_id)
        if item is None:
            return _which(ctx)
        item_id, old_kind, status = item["id"], item["kind"], item["status"]
        if u.delete:
            store.update_item(item_id, status="dismissed", route_error=None)
            store.add_event(item_id, "deleted", {})
            store.disable_reminders(item_id)
            deleted = item
        else:
            filed_ref = second_task_ref(item) if status == "filed" else None
            new_kind = "task" if u.send_to_second else u.new_kind
            if new_kind and new_kind != old_kind:
                action = "retagged"
            if u.deadline and (item.get("fields") or {}).get("deadline") != u.deadline:
                notes.append(_deadline_note(u.deadline))
            if project and project != item.get("project"):
                notes.append(f"Project: {project}")
            changed, patch_outcome = await correct_item(
                deps, item, new_kind=new_kind, deadline=u.deadline, project=project, now=ctx.now)
            if u.reminders_off:
                if not (store.get_item(item_id).get("fields") or {}).get("reminders_off"):
                    _reminders_off(ctx, item_id)
                    changed = True
            elif u.reminder_at:
                if _remind_at(ctx, item_id, u.reminder_at) is None:
                    reason = "That time has passed, no reminder set"
                else:
                    changed = True
    if deleted is not None:
        text = fallback_undone(deleted["id"], ctx.lang) + second_note(deleted)
        return Outcome(text, None, None)
    item = store.get_item(item_id)
    if not changed:
        return await _unchanged(ctx, item, reason)
    if patch_outcome is not None:
        notes.append(_patch_note(patch_outcome, filed_ref, ctx.lang))
    if reason:
        notes.append(reason)
    if old_kind in SECOND_TASK_KINDS and item["kind"] not in SECOND_TASK_KINDS and item["status"] == "filed":
        notes.append(second_note(item).strip())
    lead = "Обновила" if ctx.lang == "ru" else "Updated"
    fallback = "\n".join([build_reply(item, deps.calendar, ctx.now, ctx.lang, lead=lead), *notes])
    text = await _say(ctx, _facts(ctx, item, action, fallback, extra=notes))
    return Outcome(text, None, item_id)  # Undo only under capture replies (spec 6.6)


# -- followup -----------------------------------------------------------


async def do_followup(ctx: Ctx) -> Outcome:
    u, deps, store = ctx.u, ctx.deps, ctx.deps.store
    if u.save_previous:
        return await _save_previous(ctx)
    if u.target_item_id is None:
        return _which(ctx, "done" if u.mark_done else None)
    tz = _tz(ctx)
    lines: list[str] = []
    action, reason = "unchanged", None
    async with deps.drain_lock:
        item = store.get_item(u.target_item_id)
        if item is None:
            return _which(ctx)
        item_id = item["id"]
        if u.mark_done:
            lines.append(handle_done(store, str(item_id)))
            action = "done"
        else:
            if u.reminders_off:
                if not (item.get("fields") or {}).get("reminders_off"):
                    _reminders_off(ctx, item_id)
                    lines.append(f"Reminders off for #{item_id} · {_title(item)}")
                    action = "reminders off"
            elif u.reminder_at:
                at = _remind_at(ctx, item_id, u.reminder_at)
                if at is None:
                    reason = "That time has passed, no reminder set"
                else:
                    lines.append(f"Reminder for #{item_id} · {_title(item)}: {fmt_full(at, tz)}")
                    action = "reminder set"
            if u.snooze_days and not u.reminders_off:
                until = _snooze(ctx, item_id, u.snooze_days)
                lines.append(f"Snoozed #{item_id} · {_title(item)} until {fmt_full(until, tz)}")
                action = "snoozed"
    item = store.get_item(item_id)
    if not lines:
        return await _unchanged(ctx, item, reason)
    if reason:
        lines.append(reason)
    fallback = "\n".join(lines)
    text = await _say(ctx, _facts(ctx, item, action, fallback, extra=lines))
    return Outcome(text, None, item_id)


# -- command ------------------------------------------------------------


async def _command_block(ctx: Ctx) -> tuple[str, int | None, bool]:
    """(deterministic text, item acted on, whether a model lead may go above it)."""
    u, deps, store = ctx.u, ctx.deps, ctx.deps.store
    name, tz = u.command, deps.settings.timezone
    if name == "list":
        if u.shelf is None:
            ask = ("Какую полку? " if ctx.lang == "ru" else "Which shelf? ") + " ".join(KINDS) + "."
            return not_saved(ctx, ask), None, False
        found = store.list_items(kind=u.shelf, status=None if u.include_all else "new", limit=1_000_000)
        return render_list(found, u.shelf, 1, show_status=u.include_all), None, True
    if name in ("show", "done", "drop"):
        if u.target_item_id is None:
            return _which(ctx, name).text, None, False
        if name == "show":
            item = find_item(store, u.target_item_id)
            return (render_show(item, tz), item["id"], True) if item else (_which(ctx).text, None, False)
        async with deps.drain_lock:  # a status write must not interleave with a drain
            text = (handle_done if name == "done" else handle_drop)(store, str(u.target_item_id))
        return text, u.target_item_id, True
    if name == "inbox":
        return render_inbox(store, deps.context_cache.age_seconds(ctx.now), ctx.now, tz), None, True
    if name == "sync":
        return await sync_now(deps, ctx.lang), None, True
    if name == "flush":
        return await flush_now(deps, ctx.lang), None, True
    if name == "status":
        return status_line(store, ctx.now, tz), None, True
    return HELP_TEXT, None, True


async def do_command(ctx: Ctx) -> Outcome:
    block, item_id, lead_ok = await _command_block(ctx)
    lead = ""
    if lead_ok:
        facts = Facts(language=ctx.u.language, action="listed", item_id=item_id,
                      extra=[f"command {ctx.u.command}", block.splitlines()[0] if block else ""], fallback="")
        lead = (await _say(ctx, facts)).strip()
    return Outcome(cap(f"{lead}\n{block}" if lead else block), None, item_id)


# -- chat ---------------------------------------------------------------


async def do_chat(ctx: Ctx) -> Outcome:
    help_line = CHAT_HELP[ctx.lang]
    facts = Facts(language=ctx.u.language, action="not saved", extra=[help_line],
                  fallback=f"{help_line} {fallback_not_saved(ctx.lang)}")
    return Outcome(not_saved(ctx, await _say(ctx, facts)), None, None)


EXECUTORS = {"capture": do_capture, "correct": do_correct, "followup": do_followup,
             "command": do_command, "chat": do_chat}
