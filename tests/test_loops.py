import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawdia import __version__
from clawdia.loops import refresh_context, run_loops, tick
from clawdia.second_client import SecondAuthFailed, SecondUnavailable
from clawdia.tagger import TagResult

MON_9 = datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)  # Monday, Lisbon is UTC+0 in November
SAT_9 = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)


def mk(tmp_path, now=MON_9, post=None):
    from tests.test_capture import deps as mkdeps

    d = mkdeps(tmp_path, TagResult("idea", "x", None, {}, 0.9, None, "m"), now=now)
    d.bot = SimpleNamespace(send_message=AsyncMock())
    d.second.heartbeat = AsyncMock()
    d.second.get_context = AsyncMock(return_value=None)
    d.store.meta_set("digest_enabled", "0")  # these tests are about the other steps; the digest has its own tests
    if post is not None:
        d.second.post_task = post
    return d


def routing(d, created="2026-11-02T07:00:00+00:00", error="unavailable"):
    return d.store.create_item(kind="task", status="routing", title="x", created_at=created, route_error=error)


def sent_texts(d):
    return [c.kwargs["text"] for c in d.bot.send_message.await_args_list]


def mins(m, base=MON_9):
    return base + timedelta(minutes=m)


@pytest.mark.asyncio
async def test_stuck_warning_once_per_workday(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d)
    await tick(d, MON_9)  # first failed drain: the clock starts here
    assert d.store.meta_get("second_down_since") == "2026-11-02T09:00:00.000000+00:00"
    assert d.bot.send_message.await_count == 0
    await tick(d, mins(55))
    assert d.bot.send_message.await_count == 0
    await tick(d, mins(60))
    await tick(d, mins(65))
    assert d.bot.send_message.await_count == 1
    assert d.bot.send_message.await_args.kwargs["text"] == "В очереди 1 задача, Second API недоступен с 09:00"
    assert d.store.meta_get("last_stuck_warning_date") == "2026-11-02"


@pytest.mark.asyncio
async def test_stuck_warning_repeats_next_day(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d)
    await tick(d, MON_9)
    await tick(d, mins(60))
    await tick(d, MON_9 + timedelta(days=1))
    assert d.bot.send_message.await_count == 2
    assert sent_texts(d)[1].endswith("с пн 09:00")  # the outage began Monday and still lasts


@pytest.mark.asyncio
async def test_stuck_warning_unauthorized_wording_and_count(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondAuthFailed()))
    routing(d, error=None)
    routing(d, created="2026-11-02T08:00:00+00:00", error=None)
    await tick(d, MON_9)
    assert d.store.meta_get("second_down_reason") == "unauthorized"
    await tick(d, mins(60))
    assert sent_texts(d) == ["В очереди 2 задачи, Second API отклоняет токен Клавдии с 09:00"]


@pytest.mark.asyncio
async def test_stuck_since_is_the_failure_start_in_local_time(tmp_path):
    base = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)  # Lisbon is UTC+1 in July
    d = mk(tmp_path, now=base, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d, created="2026-07-06T05:00:00+00:00", error=None)  # captured long before the failure
    await tick(d, base)
    await tick(d, mins(60, base))
    assert sent_texts(d) == ["В очереди 1 задача, Second API недоступен с 13:00"]


@pytest.mark.asyncio
async def test_no_stuck_warning_under_threshold_or_on_a_weekend(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d)
    await tick(d, MON_9)
    await tick(d, mins(59))
    assert d.bot.send_message.await_count == 0

    d2 = mk(tmp_path / "sat", now=SAT_9)
    routing(d2, created="2026-11-06T07:00:00+00:00")
    d2.store.meta_set("second_down_since", "2026-11-06T18:00:00+00:00")
    d2.store.meta_set("second_down_reason", "unavailable")
    await tick(d2, SAT_9)  # not a work day
    assert d2.bot.send_message.await_count == 0


@pytest.mark.asyncio
async def test_stale_failure_at_midnight_does_not_warn(tmp_path):
    """Friday failure, warned Friday, API healthy since: Monday 00:01 must stay silent."""
    fri = datetime(2026, 11, 6, 18, 0, tzinfo=timezone.utc)
    d = mk(tmp_path, now=fri, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d, created="2026-11-06T18:00:00+00:00")
    await tick(d, fri)
    await tick(d, mins(60, fri))
    assert d.bot.send_message.await_count == 1
    d.second.post_task = AsyncMock(return_value={"id": 1, "project": "Acme"})  # healthy again
    d.store.meta_set("last_stuck_warning_date", "2026-11-06")
    # Spec 6.9: the drain runs at any hour, so Monday 00:01 already files it and clears the state.
    await tick(d, datetime(2026, 11, 9, 0, 1, tzinfo=timezone.utc))
    assert d.second.post_task.await_count == 1 and d.bot.send_message.await_count == 1
    assert not d.store.meta_get("second_down_since")


@pytest.mark.asyncio
async def test_overnight_capture_one_failed_morning_drain_is_not_an_alarm(tmp_path):
    d = mk(tmp_path, now=datetime(2026, 11, 2, 7, 0, tzinfo=timezone.utc), post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d, created="2026-11-02T06:00:00+00:00", error=None)  # captured an hour before the 07:00 gate
    base = datetime(2026, 11, 2, 7, 0, tzinfo=timezone.utc)
    await tick(d, base)
    await tick(d, mins(5, base))
    assert d.bot.send_message.await_count == 0
    await tick(d, mins(65, base))  # 08:05, still failing
    assert sent_texts(d) == ["В очереди 1 задача, Second API недоступен с 07:00"]


@pytest.mark.asyncio
async def test_successful_drain_clears_the_failure_state(tmp_path):
    d = mk(tmp_path, post=AsyncMock(return_value={"id": 5, "project": "Acme"}))
    routing(d)
    d.store.meta_set("second_down_since", "2026-11-02T07:00:00+00:00")
    d.store.meta_set("second_down_reason", "unavailable")
    await tick(d, MON_9)
    assert not d.store.meta_get("second_down_since") and not d.store.meta_get("second_down_reason")
    assert d.bot.send_message.await_count == 0


@pytest.mark.asyncio
async def test_stuck_warning_still_waits_for_work_day(tmp_path):
    d = mk(tmp_path, now=SAT_9, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d, created="2026-11-06T18:00:00+00:00")
    await tick(d, SAT_9)  # Saturday: the drain really tries and the failure clock starts
    assert d.second.post_task.await_count == 1 and d.store.meta_get("second_down_since")
    await tick(d, mins(120, SAT_9))
    assert d.bot.send_message.await_count == 0  # but the stuck warning is a work-day message


@pytest.mark.asyncio
async def test_no_warning_when_the_state_predates_todays_first_drain(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d)
    d.store.meta_set("second_down_since", "2026-10-30T18:00:00+00:00")
    d.store.meta_set("second_down_reason", "unavailable")
    d.store.meta_set("last_drain_at", "2026-11-02T06:57:00+00:00")  # held by the gate; not due again at 07:00
    await tick(d, datetime(2026, 11, 2, 7, 0, tzinfo=timezone.utc))
    assert d.second.post_task.await_count == 0 and d.bot.send_message.await_count == 0


@pytest.mark.asyncio
async def test_failed_warning_send_is_retried_and_date_not_set(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d)
    d.store.meta_set("second_down_since", "2026-11-02T07:00:00+00:00")
    d.store.meta_set("second_down_reason", "unavailable")
    d.bot.send_message = AsyncMock(side_effect=[RuntimeError("tg down"), None])
    await tick(d, MON_9)
    assert d.store.meta_get("last_stuck_warning_date") is None
    await tick(d, mins(1))
    assert d.store.meta_get("last_stuck_warning_date") == "2026-11-02"


@pytest.mark.asyncio
async def test_heartbeat_every_15_minutes_not_each_tick(tmp_path):
    d = mk(tmp_path)
    for m in (0, 1, 2, 14, 15, 16):
        await tick(d, MON_9 + timedelta(minutes=m))
    assert d.second.heartbeat.await_count == 2


@pytest.mark.asyncio
async def test_heartbeat_stats(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondUnavailable()))
    d.store.create_item(kind="idea", title="a", created_at="2026-11-01T23:00:00+00:00")  # yesterday
    d.store.create_item(kind="idea", title="b", created_at="2026-11-02T08:00:00+00:00")
    routing(d, created="2026-11-02T08:30:00+00:00")
    d.store.meta_set("last_update_at", "2026-11-02T08:55:00+00:00")
    await tick(d, MON_9)
    d.second.heartbeat.assert_awaited_once_with(
        {"captured_today": 2, "outbox_depth": 1, "last_update_at": "2026-11-02T08:55:00+00:00", "version": __version__})


@pytest.mark.asyncio
async def test_heartbeat_error_is_swallowed_and_not_hammered(tmp_path):
    d = mk(tmp_path)
    d.second.heartbeat = AsyncMock(side_effect=SecondUnavailable())
    await tick(d, MON_9)
    await tick(d, MON_9 + timedelta(minutes=1))
    assert d.second.heartbeat.await_count == 1
    assert d.store.meta_get("last_heartbeat_at") is not None


@pytest.mark.asyncio
async def test_drain_every_5_minutes_under_lock_with_notify(tmp_path):
    d = mk(tmp_path, post=AsyncMock(return_value={"id": 77, "project": "Acme"}))
    d.notify = AsyncMock()
    i = routing(d, error=None)
    async with d.drain_lock:
        t = asyncio.create_task(tick(d, MON_9))
        await asyncio.sleep(0.05)
        assert d.second.post_task.await_count == 0  # waiting for the lock
        assert not t.done()
    await t
    assert d.second.post_task.await_count == 1 and d.store.get_item(i)["status"] == "filed"
    d.notify.assert_awaited_once_with(i, 77, "Acme")
    routing(d, error=None)
    await tick(d, MON_9 + timedelta(minutes=4))
    assert d.second.post_task.await_count == 1
    await tick(d, MON_9 + timedelta(minutes=5))
    assert d.second.post_task.await_count == 2


@pytest.mark.asyncio
async def test_drain_without_notify_does_not_crash(tmp_path):
    d = mk(tmp_path, post=AsyncMock(return_value={"id": 77, "project": "Acme"}))
    d.notify = None
    i = routing(d, error=None)
    await tick(d, MON_9)
    assert d.store.get_item(i)["status"] == "filed"


@pytest.mark.asyncio
async def test_context_refresh_saves_on_200_and_sends_etag(tmp_path):
    d = mk(tmp_path)
    d.context_cache.save({"projects": [{"name": "Acme"}], "generated_at": "2026-11-02T08:59:00+00:00"}, "e1")
    d.second.get_context = AsyncMock(return_value=({"projects": [{"name": "New"}], "generated_at": "2026-11-02T09:00:00+00:00"}, "e2"))
    await tick(d, MON_9)
    d.second.get_context.assert_awaited_once_with(etag="e1")
    snap = d.context_cache.load()
    assert snap.etag == "e2" and snap.projects == [{"name": "New"}]
    assert d.store.meta_get("last_context_refresh_at") is not None
    await tick(d, MON_9 + timedelta(hours=5, minutes=59))
    assert d.second.get_context.await_count == 1
    await tick(d, MON_9 + timedelta(hours=6))
    assert d.second.get_context.await_count == 2


@pytest.mark.asyncio
async def test_context_304_marks_refreshed_without_saving(tmp_path):
    d = mk(tmp_path)
    d.context_cache.save({"projects": [{"name": "Acme"}]}, "e1")
    await tick(d, MON_9)
    assert d.context_cache.load().etag == "e1" and d.store.meta_get("last_context_refresh_at") is not None


@pytest.mark.asyncio
async def test_context_error_keeps_snapshot_and_retries(tmp_path):
    d = mk(tmp_path)
    d.context_cache.save({"projects": [{"name": "Acme"}]}, "e1")
    d.second.get_context = AsyncMock(side_effect=SecondUnavailable())
    await tick(d, MON_9)
    assert d.context_cache.load().etag == "e1" and d.store.meta_get("last_context_refresh_at") is None
    with pytest.raises(SecondUnavailable):
        await refresh_context(d)


@pytest.mark.asyncio
async def test_one_failing_step_does_not_stop_the_others(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=RuntimeError("boom")))
    d.second.heartbeat = AsyncMock(side_effect=RuntimeError("boom"))
    d.second.get_context = AsyncMock(side_effect=RuntimeError("boom"))
    routing(d)
    d.store.meta_set("second_down_since", "2026-11-02T07:00:00+00:00")
    d.store.meta_set("second_down_reason", "unavailable")
    await tick(d, MON_9)
    assert sent_texts(d) == ["В очереди 1 задача, Second API недоступен с 07:00"]
    assert d.second.heartbeat.await_count == 1 and d.second.get_context.await_count == 1
    assert d.store.meta_get("last_poll_at") is not None


@pytest.mark.asyncio
async def test_reminders_sent_every_tick_and_poll_stamp(tmp_path):
    d = mk(tmp_path)
    i = d.store.create_item(kind="event", title="Dentist", created_at="2026-11-01T10:00:00+00:00")
    d.store.add_reminder(i, MON_9 - timedelta(minutes=1))
    await tick(d, MON_9)
    assert d.bot.send_message.await_count == 1 and "Dentist" in sent_texts(d)[0]
    assert d.store.meta_get("last_poll_at") == "2026-11-02T09:00:00.000000+00:00"
    await tick(d, MON_9 + timedelta(minutes=1))
    assert d.store.meta_get("last_poll_at") == "2026-11-02T09:01:00.000000+00:00"
    assert d.bot.send_message.await_count == 1  # already sent


@pytest.mark.asyncio
async def test_run_loops_survives_a_crashing_tick_and_cancels(tmp_path, monkeypatch):
    d = mk(tmp_path)
    calls = []

    async def flaky(deps, now):
        calls.append(now)
        if len(calls) == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr("clawdia.loops.tick", flaky)
    task = asyncio.create_task(run_loops(d, interval_s=0.01))
    await asyncio.sleep(0.1)
    assert len(calls) >= 2 and not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_serve_wires_context_refresh_and_runs_loops_beside_polling(tmp_path, monkeypatch):
    from clawdia import bot as botmod
    from clawdia.config import Settings

    seen = {}

    class FakeBot:
        def __init__(self, token):
            self.session = SimpleNamespace(close=AsyncMock())

        async def send_message(self, *a, **k):
            return None

    class FakeDispatcher:
        def include_router(self, router):
            seen["router"] = router

        async def start_polling(self, bot):
            await asyncio.sleep(0.05)  # let the loop task run its first tick
            seen["deps_loop_ran"] = True

    class FakeSecond:
        def __init__(self, url, token):
            self.heartbeat = AsyncMock()
            self.get_context = AsyncMock(return_value=({"projects": [], "generated_at": "2026-11-02T09:00:00+00:00"}, "e1"))
            self.post_task = AsyncMock()
            self.aclose = AsyncMock()

    ai = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(botmod, "Bot", FakeBot)
    monkeypatch.setattr(botmod, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(botmod, "SecondClient", FakeSecond)
    monkeypatch.setattr(botmod, "_ai_client", lambda settings: ai)
    built = []
    real_build = botmod.build_router
    monkeypatch.setattr(botmod, "build_router", lambda deps: built.append(deps) or real_build(deps))
    settings = Settings(bot_token="t", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k", state_dir=tmp_path)
    await botmod.serve(settings)
    deps = built[0]
    assert deps.refresh_context is not None and seen["deps_loop_ran"]
    assert deps.second.heartbeat.await_count == 1  # the loop ticked while polling was running
    assert (tmp_path / "context.json").exists()
    deps.second.aclose.assert_awaited_once()
    assert not [t for t in asyncio.all_tasks() if "run_loops" in repr(t)]  # cancelled and awaited


@pytest.mark.asyncio
async def test_tick_applies_patches_on_weekend(tmp_path):
    d = mk(tmp_path, now=SAT_9)
    i = d.store.create_item(kind="task", status="filed", title="x", filed_ref="2601", project="Acme",
                            fields={"second_patch": {"deadline": "2026-11-09"}})
    d.second.patch_task = AsyncMock(return_value={"id": 2601, "project": "Acme", "deadline": "2026-11-09"})
    await tick(d, SAT_9)
    d.second.patch_task.assert_awaited_once_with(2601, deadline="2026-11-09", project=None)
    assert d.store.get_item(i)["fields"] == {"deadline": "2026-11-09"}


@pytest.mark.asyncio
async def test_tick_patch_failure_does_not_stop_other_steps(tmp_path):
    d = mk(tmp_path)
    d.store.create_item(kind="task", status="filed", title="x", filed_ref="2601", fields={"second_patch": {"deadline": "2026-11-09"}})
    d.second.patch_task = AsyncMock(side_effect=RuntimeError("boom"))
    await tick(d, MON_9)
    assert d.second.heartbeat.await_count == 1


@pytest.mark.asyncio
async def test_stuck_warning_names_the_weekday_when_the_outage_began_earlier(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondUnavailable()))
    routing(d)
    d.store.meta_set("second_down_since", "2026-10-31T09:00:00+00:00")  # Saturday
    d.store.meta_set("second_down_reason", "unavailable")
    d.store.meta_set("last_drain_at", "2026-11-02T07:30:00+00:00")
    await tick(d, MON_9)
    assert sent_texts(d) == ["В очереди 1 задача, Second API недоступен с сб 09:00"]


@pytest.mark.asyncio
async def test_stuck_warning_english_chat_has_a_correct_singular(tmp_path):
    d = mk(tmp_path, post=AsyncMock(side_effect=SecondUnavailable()))
    d.store.add_turn(d.settings.owner_id, "user", "what is new with the bot please", at=MON_9 - timedelta(hours=2))
    routing(d)
    d.store.meta_set("second_down_since", "2026-11-02T07:00:00+00:00")
    d.store.meta_set("second_down_reason", "unavailable")
    d.store.meta_set("last_drain_at", "2026-11-02T07:30:00+00:00")
    await tick(d, MON_9)
    assert sent_texts(d) == ["1 task waiting, Second API unreachable since 07:00"]


@pytest.mark.asyncio
async def test_tick_runs_calls_step(tmp_path, monkeypatch):
    from clawdia import loops

    d = mk(tmp_path)
    seen = AsyncMock()
    monkeypatch.setattr(loops, "check_waits", seen)
    await tick(d, MON_9)
    seen.assert_awaited_once_with(d, MON_9)


@pytest.mark.asyncio
async def test_tick_runs_morning_step(tmp_path, monkeypatch):
    from clawdia import loops

    d = mk(tmp_path)
    seen = AsyncMock()
    monkeypatch.setattr(loops, "check_digest", seen)
    await tick(d, MON_9)
    seen.assert_awaited_once_with(d, MON_9)


@pytest.mark.asyncio
async def test_tick_calls_review(tmp_path, monkeypatch):
    from clawdia import loops

    d = mk(tmp_path)
    seen = AsyncMock()
    monkeypatch.setattr(loops, "check_review", seen)
    await tick(d, MON_9)
    seen.assert_awaited_once_with(d, MON_9)
