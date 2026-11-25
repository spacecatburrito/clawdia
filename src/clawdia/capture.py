"""Capture pipeline: one Telegram message in, one saved, tagged and routed item out.

The item is persisted before any network call. Every later step is wrapped, so a
failure still yields a `Captured` for the saved id plus an "error" event (item id
and exception type only, never message text).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from clawdia import media
from clawdia.dedupe import resolve
from clawdia.jobs import AUTH_LINE, DOWN_LINE, SENT_LINE, SKIPPED_LINE, submit_job
from clawdia.keyboards import undo_keyboard
from clawdia.kinds import KINDS, SECOND_TASK_KINDS
from clawdia.render import ago
from clawdia.reminders import normalise_when, reminder_line, schedule_reminders
from clawdia.reply import write_reply
from clawdia.second_client import SecondUnavailable, drain_outbox
from clawdia.storage import Store
from clawdia.understand import understand

log = logging.getLogger(__name__)

GROUP_KEY_TTL_S = 60.0
SNIPPET_CHARS = 120
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_ROUTE_ERRORS = {"unavailable": "Second unreachable", "unauthorized": "auth error"}
_ROUTE_ERRORS_RU = {"unavailable": "Second недоступен", "unauthorized": "ошибка доступа"}
KIND_RU = {"task": "задача", "job": "вакансия", "event": "событие", "recipe": "рецепт", "film": "фильм",
           "place": "место", "buy": "покупка", "idea": "идея", "unsorted": "без категории"}

Keyboard = list[list[tuple[str, str]]]


@dataclass
class Captured:
    item_id: int
    kind: str
    title: str
    reply_text: str
    keyboard: Keyboard
    duplicate: bool = False  # a later part of a media group: nothing new to reply


@dataclass
class _Group:
    item_id: int
    started: float
    done: asyncio.Future


@dataclass
class Deps:
    store: Store
    settings: Any
    context_cache: Any
    second: Any
    calendar: Any
    bot: Any
    tag_fn: Callable[..., Awaitable[Any]]
    transcribe_fn: Callable[..., Awaitable[str | None]]
    save_media_fn: Callable[..., Awaitable[tuple[Path, str] | None]]
    now_fn: Callable[[], datetime]
    group_wait_s: float
    drain_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ai_client: Any = None  # shared Anthropic client, so no HTTP pool is built per message
    refresh_context: Callable[[], Awaitable[None]] | None = None
    notify: Callable[[int, int, str], Awaitable[None]] | None = None  # announces other items filed by a drain
    groups: dict[str, _Group] = field(default_factory=dict)
    pair_slots: dict[int, list] = field(default_factory=dict)  # comment/forward pairing, see pairing.py
    understand_fn: Callable[..., Awaitable[Any]] = understand  # conversation engine: intent call
    reply_fn: Callable[..., Awaitable[str]] = write_reply  # conversation engine: reply call
    second_reader_fn: Callable[[], Any] | None = None  # agent: a new Second read session (second_reads)


# -- extraction ---------------------------------------------------------


def _utf16_slice(text: str, offset: int, length: int) -> str:
    raw = text.encode("utf-16-le")
    return raw[offset * 2:(offset + length) * 2].decode("utf-16-le", errors="ignore")


def _urls(text: str | None, entities: Any) -> list[str]:
    found: list[str] = []
    for ent in entities or []:
        kind = getattr(ent, "type", None)
        if kind == "text_link" and getattr(ent, "url", None):
            found.append(ent.url)
        elif kind == "url" and text:
            found.append(_utf16_slice(text, ent.offset, ent.length))
    found.extend(m.rstrip(".,;:!?)") for m in _URL_RE.findall(text or ""))
    return list(dict.fromkeys(u for u in found if u))


def _forward_from(origin: Any) -> dict[str, Any] | None:
    if origin is None:
        return None
    kind = getattr(origin, "type", None)
    name = chat_title = link = None
    if kind == "user":
        user = getattr(origin, "sender_user", None)
        name = " ".join(p for p in (getattr(user, "first_name", None), getattr(user, "last_name", None)) if p) \
            or getattr(user, "username", None)
    elif kind == "hidden_user":
        name = getattr(origin, "sender_user_name", None)
    elif kind == "chat":
        chat_title = getattr(getattr(origin, "sender_chat", None), "title", None)
        name = getattr(origin, "author_signature", None) or chat_title
    elif kind == "channel":
        chat = getattr(origin, "chat", None)
        chat_title = getattr(chat, "title", None)
        name = getattr(origin, "author_signature", None) or chat_title
        username, message_id = getattr(chat, "username", None), getattr(origin, "message_id", None)
        if username and message_id:
            link = f"https://t.me/{username}/{message_id}"
    return {"name": name, "chat_title": chat_title, "link": link}


def extract(message: Any) -> dict[str, Any]:
    text = getattr(message, "text", None)
    caption = getattr(message, "caption", None)
    urls = _urls(text, getattr(message, "entities", None)) + _urls(caption, getattr(message, "caption_entities", None))
    date = getattr(message, "date", None)
    if isinstance(date, datetime) and date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return {
        "text": text, "caption": caption, "urls": list(dict.fromkeys(urls)),
        "forward_from": _forward_from(getattr(message, "forward_origin", None)),
        "tg_message_id": getattr(message, "message_id", None), "tg_date": date,
        "media_group_id": getattr(message, "media_group_id", None),
    }


# -- reply --------------------------------------------------------------


def _task_line(item: dict, lang: str = "en") -> str | None:
    status, err = item["status"], item.get("route_error")
    ru = lang == "ru"
    if status == "filed":
        if item["kind"] == "job":
            return f"-> Notion: {item['filed_ref']}"
        return f"-> Second #{item['filed_ref']}"
    if status == "new" and err:
        return f"-> Second отклонил: {err}" if ru else f"-> Second rejected: {err}"
    if status != "routing":
        return None
    if item["kind"] == "job":
        if (item.get("fields") or {}).get("job_intake_id"):
            return SENT_LINE["ru" if ru else "en"]
        return (AUTH_LINE if err == "unauthorized" else DOWN_LINE)["ru" if ru else "en"]
    if err in _ROUTE_ERRORS:
        return f"-> Second в очереди ({_ROUTE_ERRORS_RU[err]})" if ru else f"-> Second queued ({_ROUTE_ERRORS[err]})"
    return "-> Second в очереди" if ru else "-> Second queued"


_DUP_STATUS = {
    "en": {"new": "waiting for review", "routing": "in triage", "filed": "already filed", "done": "closed", "dismissed": "dropped"},
    "ru": {"new": "ждёт разбора", "routing": "на разборе", "filed": "уже заведена", "done": "закрыта", "dismissed": "отклонена"},
}


def _when(stamp: Any, calendar: Any) -> str:
    try:
        return datetime.fromisoformat(stamp).astimezone(calendar.tz).strftime("%d.%m")
    except (TypeError, ValueError):
        return ""


def _earlier(fields: dict, prefix: str, calendar: Any, ru: bool, with_status: bool) -> str:
    """"#10 · title (05.11, waiting for review)": the earlier vacancy a capture duplicates."""
    title = " ".join(str(fields.get(f"{prefix}_title") or "").split()) or ("(без названия)" if ru else "(no title)")
    meta = [m for m in (_when(fields.get(f"{prefix}_at"), calendar),
                        _DUP_STATUS["ru" if ru else "en"].get(fields.get(f"{prefix}_status")) if with_status else None) if m]
    return f"#{fields[prefix]} · {title}" + (f" ({', '.join(meta)})" if meta else "")


def is_duplicate(item: dict) -> bool:
    """A vacancy merged into an earlier one: nothing new was saved, so no Undo button and no model reply."""
    return bool((item.get("fields") or {}).get("duplicate_of"))


def keyboard_for(item: dict) -> Keyboard:
    return [] if is_duplicate(item) else undo_keyboard(item["id"])


def build_reply(item: dict, calendar: Any, now: datetime, lang: str = "en", lead: str | None = None) -> str:
    """The deterministic reply about an item: "Saved #N · kind · title" plus its Second and
    reminder lines; Russian ("Сохранила #N · задача · ...") when `lang` is "ru". `lead`
    replaces the leading verb (e.g. "Updated" / "Обновила" for a correction)."""
    ru = lang == "ru"
    fields = item.get("fields") or {}
    if is_duplicate(item):
        kind = KIND_RU["job"] if ru else "job"
        return ("Уже есть " if ru else "Already have ") + _earlier(fields, "duplicate_of", calendar, ru, True).replace(" · ", f" · {kind} · ", 1)
    title = " ".join((item.get("title") or "").split()) or ("(без названия)" if ru else "(no title)")
    kind = KIND_RU.get(item["kind"], item["kind"]) if ru else item["kind"]
    verb = lead or ("Сохранила" if ru else "Saved")
    lines = [f"{verb} #{item['id']} · {kind} · {title}"]
    origin = item.get("forward_from")
    name = origin.get("name") if isinstance(origin, dict) else None
    if name:
        chat = origin.get("chat_title")
        lines.append(("От: " if ru else "From: ") + " ".join(name.split())
                     + (f" ({' '.join(chat.split())})" if chat and chat != name else ""))
    media_error = (item.get("fields") or {}).get("media_error")
    if media_error:
        lines.append(f"Файл не сохранён ({media_error}), отправь ещё раз" if ru
                     else f"Media not saved ({media_error}), send again")
    snippet = item.get("transcript") or item.get("ocr_text")
    if snippet:
        lines.append(" ".join(snippet.split())[:SNIPPET_CHARS])
    if item["kind"] in SECOND_TASK_KINDS | {"job"}:
        line = _task_line(item, lang)
        if line:
            lines.append(line)
    if item["kind"] == "job" and fields.get("job_images_skipped"):
        lines.append(SKIPPED_LINE["ru" if ru else "en"].format(names=", ".join(fields["job_images_skipped"])))
    reminder = reminder_line(item.get("fields"), calendar.tz, lang)
    if reminder:
        lines.append(reminder)
    if fields.get("possible_duplicate_of"):
        lines.append(("Возможный дубль " if ru else "Possible duplicate of ")
                     + _earlier(fields, "possible_duplicate_of", calendar, ru, False))
    return "\n".join(lines)


def captured_for(deps: Deps, item_id: int, now: datetime | None = None) -> Captured:
    item = deps.store.get_item(item_id)
    reply = build_reply(item, deps.calendar, now or deps.now_fn())
    return Captured(item_id, item["kind"], item.get("title") or "", reply, keyboard_for(item))


# -- routing ------------------------------------------------------------


async def queue_and_drain(deps: Deps, item_id: int, now: datetime) -> None:
    """Queue a `new` item for Second and drain the outbox. The caller must hold `deps.drain_lock`."""
    item = deps.store.get_item(item_id)
    if item is None or item["status"] != "new":  # deleted or already queued/filed meanwhile
        return
    deps.store.update_item(item_id, status="routing", route_error=None)

    async def notify(filed_id: int, second_id: int, project: str) -> None:
        if filed_id != item_id and deps.notify is not None:  # the reply itself announces this item
            await deps.notify(filed_id, second_id, project)

    await drain_outbox(deps.store, deps.second, now, notify)


async def route_task(deps: Deps, item_id: int, now: datetime) -> None:
    async with deps.drain_lock:
        await queue_and_drain(deps, item_id, now)


async def _drain(deps: Deps) -> int:
    async def notify(item_id: int, second_id: int, project: str) -> None:
        if deps.notify is not None:
            await deps.notify(item_id, second_id, project)

    return await drain_outbox(deps.store, deps.second, deps.now_fn(), notify)


async def sync_now(deps: Deps, lang: str = "en") -> str:
    """Refresh the context, then drain the outbox; the /sync reply text (ru or en)."""
    ru = lang == "ru"
    refresh = "контекст без изменений" if ru else "context unchanged"
    async with deps.drain_lock:
        if deps.refresh_context is not None:
            try:
                await deps.refresh_context()
                age = deps.context_cache.age_seconds(deps.now_fn())
                if ru:
                    refresh = "контекст обновлён" + ("" if age is None else f" {ago(age, lang)}")
                else:
                    refresh = "context refreshed" + ("" if age is None else f" {ago(age)}")
            except Exception as exc:  # noqa: BLE001 - the drain below does not depend on it
                log.warning("context refresh failed: %s", type(exc).__name__)
                refresh = "не удалось обновить контекст" if ru else "context refresh failed"
        try:
            filed = await _drain(deps)
        except Exception as exc:  # noqa: BLE001
            log.error("sync drain failed: %s", type(exc).__name__)
            head = "Синк не удался" if ru else "Sync failed"
            return f"{head} ({type(exc).__name__}), {refresh}"
    return f"{'Отправлено' if ru else 'Filed'} {filed}, {refresh}"


async def flush_now(deps: Deps, lang: str = "en") -> str:
    """Drain the outbox now; the /flush reply text (ru or en)."""
    ru = lang == "ru"
    async with deps.drain_lock:
        try:
            filed = await _drain(deps)
        except Exception as exc:  # noqa: BLE001
            log.error("flush failed: %s", type(exc).__name__)
            return f"{'Отправка не удалась' if ru else 'Flush failed'} ({type(exc).__name__})"
        queued = len(deps.store.outbox())
    if ru:
        return f"Отправлено {filed}" + (f", ещё в очереди: {queued}" if queued else "")
    return f"Filed {filed}" + (f", {queued} still queued" if queued else "")


# -- pipeline -----------------------------------------------------------


async def _step(deps: Deps, item_id: int, name: str, awaitable: Awaitable[Any]) -> Any:
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001 - the saved item must survive any step failure
        log.warning("capture step %s failed for item %s: %s", name, item_id, type(exc).__name__)
        try:
            deps.store.add_event(item_id, "error", {"step": name, "type": type(exc).__name__})
        except Exception:  # noqa: BLE001
            pass
        return None


def _add_media(store: Store, item_id: int, path: Path, mime: str) -> None:
    item = store.get_item(item_id)
    paths = list(item.get("media_paths") or []) + [str(path)]
    extra = {} if item.get("media_path") else {"media_path": str(path), "media_mime": mime}
    store.update_item(item_id, media_paths=paths, **extra)


def _duration(message: Any) -> int | None:
    for attr in ("voice", "audio"):
        value = getattr(getattr(message, attr, None), "duration", None)
        if value is not None:
            return int(value)
    return None


def _record_media_error(deps: Deps, item_id: int, reason: str, error_type: str) -> None:
    fields = dict(deps.store.get_item(item_id).get("fields") or {})
    fields["media_error"] = reason
    deps.store.update_item(item_id, fields=fields)
    deps.store.add_event(item_id, "error", {"step": "media", "type": error_type})


async def _download(deps: Deps, message: Any, item_id: int, stem: str) -> None:
    try:
        res = await deps.save_media_fn(
            deps.bot, message, deps.settings.state_dir / "media", deps.settings.max_document_mb, stem=stem)
    except Exception as exc:  # noqa: BLE001 - the item stays saved without its media
        log.warning("capture step media failed for item %s: %s", item_id, type(exc).__name__)
        _record_media_error(deps, item_id, "download failed", type(exc).__name__)
        return
    if res:
        _add_media(deps.store, item_id, *res)
        return
    reason = media.skip_reason(message, deps.settings.max_document_mb)
    if reason:
        _record_media_error(deps, item_id, reason, reason.replace(" ", "_"))


async def _schedule(deps: Deps, item_id: int, result: Any, now: datetime) -> None:
    schedule_reminders(deps.store, deps.store.get_item(item_id), result, deps.calendar, now)


async def _enrich_and_route(message: Any, deps: Deps, item_id: int, now: datetime,
                            glue: tuple[str, str] | None = None) -> None:
    item = deps.store.get_item(item_id)
    mime = item.get("media_mime") or ""
    if mime.startswith("audio/") and item.get("media_path"):
        text = await _step(deps, item_id, "transcribe",
                           deps.transcribe_fn(Path(item["media_path"]), _duration(message), deps.settings))
        if text:
            deps.store.update_item(item_id, transcript=text)
            item["transcript"] = text
    if glue is not None:  # the tagger classifies on her comment; the forward is delimited context
        item["text"] = glue[0]
        item["forwarded_text"] = "\n\n".join(p for p in (glue[1], item.get("transcript")) if p)
    image = Path(item["media_path"]) if mime.startswith("image/") and item.get("media_path") else None
    result = await _step(deps, item_id, "tag", deps.tag_fn(
        item, deps.context_cache.load(), deps.settings, image_path=image, now=now, client=deps.ai_client))
    if result is None:
        return
    await apply_tag(deps, item_id, result, now)


def _resolve_duplicate(deps: Deps, item_id: int, now: datetime) -> str | None:
    """Merge a just-tagged vacancy into an earlier copy of it; a failed check keeps the item as new."""
    try:
        return resolve(deps.store, item_id, now)
    except Exception as exc:  # noqa: BLE001 - the saved item must survive any step failure
        log.warning("capture step dedupe failed for item %s: %s", item_id, type(exc).__name__)
        try:
            deps.store.add_event(item_id, "error", {"step": "dedupe", "type": type(exc).__name__})
        except Exception:  # noqa: BLE001
            pass
        return None


async def apply_tag(deps: Deps, item_id: int, result: Any, now: datetime) -> None:
    """The post-tag half of the pipeline: store a tag result, schedule reminders, route a task.

    Used by phase 1 captures after the tag call and by the conversation engine with the
    capture its understanding call already produced (no second tag call).
    """
    kind = result.kind if result.kind in KINDS else "unsorted"
    # date/time as YYYY-MM-DD / HH:MM before scheduling; a changed value keeps its raw string.
    fields = normalise_when(result.fields, now, deps.calendar.tz)
    media_error = (deps.store.get_item(item_id).get("fields") or {}).get("media_error")
    if media_error:
        fields["media_error"] = media_error
    if result.reminder_at:
        fields["reminder_at"] = result.reminder_at
    deps.store.update_item(item_id, kind=kind, title=result.title, fields=fields, project=result.project,
                           confidence=result.confidence, model=result.model)
    if kind == "job" and _resolve_duplicate(deps, item_id, now) == "same":  # no await since the update above
        deps.store.add_event(item_id, "tagged", {"kind": kind, "model": result.model})
        return
    await _step(deps, item_id, "reminders", _schedule(deps, item_id, replace(result, kind=kind, fields=fields), now))
    if kind in SECOND_TASK_KINDS:
        await _step(deps, item_id, "route", route_task(deps, item_id, now))
    if kind == "job":  # R17: a "same" duplicate returned above and is never submitted
        await _step(deps, item_id, "job", _submit_job(deps, item_id, now))
    deps.store.add_event(item_id, "tagged", {"kind": kind, "model": result.model})


async def _submit_job(deps: Deps, item_id: int, now: datetime) -> None:
    try:
        await submit_job(deps, item_id, now)
    except SecondUnavailable:  # the item stays `routing`; the poll loop retries
        pass


async def _lead(message: Any, deps: Deps, item_id: int, in_group: bool, now: datetime) -> None:
    await _download(deps, message, item_id, str(item_id))
    if in_group and deps.group_wait_s > 0:
        await asyncio.sleep(deps.group_wait_s)  # let the rest of the album arrive before tagging
    await _enrich_and_route(message, deps, item_id, now)


async def _follow(message: Any, deps: Deps, group: _Group, data: dict[str, Any]) -> Captured:
    await _download(deps, message, group.item_id, f"{group.item_id}-{data['tg_message_id']}")
    if data["caption"] and not deps.store.get_item(group.item_id).get("caption"):
        deps.store.update_item(group.item_id, caption=data["caption"])
    return replace(await asyncio.shield(group.done), duplicate=True)


async def process(message: Any, deps: Deps) -> Captured:
    now = deps.now_fn()
    data = extract(message)
    gid = data["media_group_id"]
    for key in [k for k, g in deps.groups.items() if g.done.done() and time.monotonic() - g.started > GROUP_KEY_TTL_S]:
        del deps.groups[key]
    if gid and gid in deps.groups:
        return await _follow(message, deps, deps.groups[gid], data)
    item_id = deps.store.create_item(
        status="new", kind="unsorted", text=data["text"], caption=data["caption"], urls=data["urls"] or None,
        forward_from=data["forward_from"], tg_message_id=data["tg_message_id"], tg_date=data["tg_date"],
        media_group_id=gid, chat_id=getattr(getattr(message, "chat", None), "id", None))
    group = None
    if gid:  # registered before any await so siblings arriving meanwhile attach to this item
        group = deps.groups[gid] = _Group(item_id, time.monotonic(), asyncio.get_running_loop().create_future())
    try:
        await _step(deps, item_id, "capture", _lead(message, deps, item_id, group is not None, now))
        captured = captured_for(deps, item_id, now)
    finally:
        if group is not None and not group.done.done():
            try:
                group.done.set_result(captured_for(deps, item_id, now))
            except Exception:  # noqa: BLE001
                group.done.set_result(Captured(item_id, "unsorted", "", f"Saved #{item_id} · unsorted", undo_keyboard(item_id)))
    return captured


async def _lead_glued(comment: str, forwards: list[Any], deps: Deps, item_id: int, fwd_text: str, now: datetime) -> None:
    await _download(deps, forwards[0], item_id, str(item_id))
    for message in forwards[1:]:
        await _download(deps, message, item_id, f"{item_id}-{getattr(message, 'message_id', None)}")
    await _enrich_and_route(forwards[0], deps, item_id, now, glue=(comment, fwd_text))


def _note_on_original(deps: Deps, item_id: int, comment: str) -> None:
    """A glued vacancy merged into an earlier one: her comment is not lost, it becomes a note there."""
    try:
        original_id = (deps.store.get_item(item_id).get("fields") or {}).get("duplicate_of")
        original = deps.store.get_item(original_id) if original_id else None
        if original is not None and comment.strip():
            notes = f"{original['notes']}\n{comment}" if original.get("notes") else comment
            deps.store.update_item(original["id"], notes=notes)
    except Exception as exc:  # noqa: BLE001 - the saved item must survive any step failure
        log.warning("capture step note failed for item %s: %s", item_id, type(exc).__name__)


async def process_glued(comment: str, forwards: list[Any], deps: Deps) -> Captured:
    """One item from her typed comment plus the forward(s) that came with it.

    `text` is her comment, a blank line, then the forwarded text; forward_from and urls come from
    the forwards (and the public t.me link when derivable). The tagger gets her comment as the
    instruction and the forwarded text as context only.
    """
    now = deps.now_fn()
    datas = [extract(m) for m in forwards]
    fwd_text = "\n\n".join(t for d in datas for t in [d["text"] or d["caption"]] if t and t.strip())
    links = [d["forward_from"]["link"] for d in datas if d["forward_from"] and d["forward_from"].get("link")]
    urls = list(dict.fromkeys(_urls(comment, None) + [u for d in datas for u in d["urls"]] + links))
    first = datas[0]
    item_id = deps.store.create_item(
        status="new", kind="unsorted", text=f"{comment}\n\n{fwd_text}" if fwd_text else comment, urls=urls or None,
        forward_from=first["forward_from"], tg_message_id=first["tg_message_id"], tg_date=first["tg_date"],
        chat_id=getattr(getattr(forwards[0], "chat", None), "id", None))
    await _step(deps, item_id, "capture", _lead_glued(comment, forwards, deps, item_id, fwd_text, now))
    _note_on_original(deps, item_id, comment)
    return captured_for(deps, item_id, now)
