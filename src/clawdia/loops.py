"""Background loops that run beside Telegram polling in the same process.

One `tick` per minute: reminders, outbox drain, heartbeat, context refresh and
the stuck-task warning. Each step catches its own exceptions and logs only the
exception type and item ids, never message text or tokens, so one failure
cannot stop the others or the loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from clawdia import __version__
from clawdia.calls import check_waits
from clawdia.capture import Deps
from clawdia.jobs import poll_jobs
from clawdia.mapwatch import check_map
from clawdia.morning import check as check_digest
from clawdia.review import check as check_review
from clawdia.notices import send_resolution_notices
from clawdia.ops_alerts import check_ops
from clawdia.reminders import send_due
from clawdia.render import tasks_word
from clawdia.schedule import FLUSH_TIME
from clawdia.second_client import apply_pending_patches, drain_outbox
from clawdia.storage import utc_iso

log = logging.getLogger(__name__)

DRAIN_EVERY = timedelta(minutes=5)
CONTEXT_EVERY = timedelta(hours=6)
WEEKDAYS_RU = ('пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс')


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _due(deps: Deps, key: str, every: timedelta, now: datetime) -> bool:
    last = _parse(deps.store.meta_get(key))
    return last is None or now - last >= every


async def refresh_context(deps: Deps) -> None:
    """Fetch /v1/context with the cached etag and save a new body. Raises on API errors.

    A 304 keeps the snapshot and still counts as a refresh. Callers (tick, /sync,
    startup) decide how to treat a failure; the old snapshot is never touched then.
    """
    cached = deps.context_cache.load()
    fetched = await deps.second.get_context(etag=cached.etag)
    if fetched is not None:
        body, etag = fetched
        deps.context_cache.save(body, etag)
    deps.store.meta_set("last_context_refresh_at", utc_iso(deps.now_fn()))


async def _no_notify(item_id: int, second_id: int, project: str) -> None:
    return None


FAILURES = ("unavailable", "unauthorized")


def _track_failure(deps: Deps, now: datetime, filed: int) -> None:
    """After a drain that really ran: remember when Second started failing, or forget it."""
    queue = deps.store.outbox()
    error = queue[0].get("route_error") if queue else None
    if filed > 0 or error not in FAILURES:
        deps.store.meta_set("second_down_since", "")
        deps.store.meta_set("second_down_reason", "")
        return
    if _parse(deps.store.meta_get("second_down_since")) is None:
        deps.store.meta_set("second_down_since", utc_iso(now))
    deps.store.meta_set("second_down_reason", error)


async def _patches(deps: Deps) -> None:
    async with deps.drain_lock:
        await apply_pending_patches(deps.store, deps.second)


async def _drain(deps: Deps, now: datetime) -> None:
    async with deps.drain_lock:
        filed = await drain_outbox(deps.store, deps.second, now, deps.notify or _no_notify)
        _track_failure(deps, now, filed)


def _captured_today(deps: Deps, now: datetime) -> int:
    tz = deps.calendar.tz
    midnight = datetime.combine(now.astimezone(tz).date(), datetime.min.time(), tzinfo=tz)
    return len(deps.store.list_items(since=midnight, limit=1_000_000))


async def _heartbeat(deps: Deps, now: datetime) -> None:
    # Advance first: a down API must not be retried every minute.
    deps.store.meta_set("last_heartbeat_at", utc_iso(now))
    await deps.second.heartbeat({
        "captured_today": _captured_today(deps, now),
        "outbox_depth": len(deps.store.outbox()),
        "last_update_at": deps.store.meta_get("last_update_at"),
        "version": __version__,
    })


async def _stuck_warning(deps: Deps, now: datetime) -> None:
    """Warn once per work day when Second has been failing for `stuck_minutes` on drains that really ran."""
    settings, tz = deps.settings, deps.calendar.tz
    if now < deps.calendar.next_flush_at(now):  # quiet day or before 07:00: nothing is being sent
        return
    local = now.astimezone(tz)
    today = local.date().isoformat()
    if deps.store.meta_get("last_stuck_warning_date") == today:
        return
    flush_today = datetime.combine(local.date(), FLUSH_TIME, tzinfo=tz)
    last_drain = _parse(deps.store.meta_get("last_drain_at"))
    if last_drain is None or last_drain < flush_today:  # the failure state is not from a drain today
        return
    down_since = _parse(deps.store.meta_get("second_down_since"))
    if down_since is None or now - down_since < timedelta(minutes=settings.stuck_minutes):
        return
    queue = deps.store.outbox()
    if not queue:
        return
    from clawdia.converse import chat_language  # lazy: converse imports this package's capture module

    lang = "ru" if chat_language(deps, settings.owner_id) == "ru" else "en"
    down_local = down_since.astimezone(tz)
    if down_local.date() == local.date():
        since = down_local.strftime("%H:%M")
    else:
        day = WEEKDAYS_RU[down_local.weekday()] if lang == "ru" else down_local.strftime("%a")
        since = f"{day} {down_local.strftime('%H:%M')}"
    waiting = tasks_word(len(queue), lang)
    unauthorized = deps.store.meta_get("second_down_reason") == "unauthorized"
    if lang == "ru":
        reason = "Second API отклоняет токен Клавдии" if unauthorized else "Second API недоступен"
        text = f"В очереди {waiting}, {reason} с {since}"
    else:
        reason = "Second API rejects Clawdia's token" if unauthorized else "Second API unreachable"
        text = f"{waiting} waiting, {reason} since {since}"
    await deps.bot.send_message(settings.owner_id, text=text)  # primary owner only
    deps.store.meta_set("last_stuck_warning_date", today)  # only after a successful send


async def _step(name: str, coro) -> None:
    try:
        await coro
    except Exception as exc:  # noqa: BLE001 - one failing step must not stop the others
        log.warning("loop step %s failed: %s", name, type(exc).__name__)


async def tick(deps: Deps, now: datetime) -> None:
    """One loop iteration; never raises."""
    try:
        deps.store.meta_set("last_poll_at", utc_iso(now))
    except Exception as exc:  # noqa: BLE001
        log.warning("loop step poll stamp failed: %s", type(exc).__name__)
    await _step("reminders", send_due(deps.store, deps.bot, deps.settings.owner_id, now))  # primary owner only
    if _due(deps, "last_drain_at", DRAIN_EVERY, now):
        deps.store.meta_set("last_drain_at", utc_iso(now))
        await _step("drain", _drain(deps, now))
        await _step("patches", _patches(deps))  # corrections are explicit requests: no work-day gate
    await _step("jobs", poll_jobs(deps, now))
    if _due(deps, "last_heartbeat_at", timedelta(minutes=deps.settings.heartbeat_minutes), now):
        await _step("heartbeat", _heartbeat(deps, now))
    if _due(deps, "last_ops_check_at", timedelta(minutes=deps.settings.heartbeat_minutes), now):
        deps.store.meta_set("last_ops_check_at", utc_iso(now))  # advance first: a down API is not retried every minute
        await _step("ops", check_ops(deps, now))
    if _due(deps, "last_context_refresh_at", CONTEXT_EVERY, now):
        await _step("context", refresh_context(deps))
    await _step("calls", check_waits(deps, now))
    await _step("mapwatch", check_map(deps, now))
    await _step("morning", check_digest(deps, now))
    await _step("review", check_review(deps, now))
    await _step("stuck", _stuck_warning(deps, now))
    await _step("notices", send_resolution_notices(deps))


async def run_loops(deps: Deps, interval_s: float = 60) -> None:
    while True:
        try:
            await tick(deps, deps.now_fn())
        except Exception as exc:  # noqa: BLE001 - last resort, the loop must keep going
            log.error("tick crashed: %s", type(exc).__name__)
        await asyncio.sleep(interval_s)
