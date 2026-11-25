import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from clawdia.loops import tick
from clawdia.ops_alerts import check_ops
from clawdia.second_client import SecondAuthFailed, SecondUnavailable
from tests.test_loops import MON_9, SAT_9, mk, sent_texts

INGEST = {"code": "ingest_stale", "message": "Ingest is stale (last succeeded 130 min ago)",
          "fix": "restart the ingest job and read its log"}
MON_3 = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)
MON_7 = datetime(2026, 11, 2, 7, 0, tzinfo=timezone.utc)


def ops(d, *problems):
    d.second.get_ops_health = AsyncMock(return_value={"ok": not problems, "problems": list(problems),
                                                      "checked_at": "2026-11-02T09:00:00Z"})


def state(d):
    return json.loads(d.store.meta_get("ops_state"))


def at(m, base=MON_9):
    return base + timedelta(minutes=m)


def en(d):
    d.store.add_turn(d.settings.owner_id, "user", "what is new with the bot please", at=MON_9 - timedelta(hours=1))


@pytest.mark.asyncio
async def test_new_problem_sends_once(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, MON_9)
    assert sent_texts(d) == [f"Second: импорт сообщений давно не проходил. Починка: `{INGEST['fix']}`"]
    assert d.bot.send_message.await_args.args[0] == d.settings.owner_id
    assert list(state(d)["known"]) == ["ingest_stale"]
    assert d.store.recent_turns(d.settings.owner_id, 5)[-1]["role"] == "bot"  # the agent sees it as data


@pytest.mark.asyncio
async def test_english_chat_gets_english_text_with_the_fix_verbatim(tmp_path):
    d = mk(tmp_path)
    en(d)
    ops(d, INGEST)
    await check_ops(d, MON_9)
    assert sent_texts(d) == [f"Second: {INGEST['message']}. Fix: `{INGEST['fix']}`"]


@pytest.mark.asyncio
async def test_same_problem_next_check_sends_nothing(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, MON_9)
    await check_ops(d, MON_9 + timedelta(minutes=15))
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_cleared_problem_sends_fixed_line(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, MON_9)
    ops(d)
    await check_ops(d, at(15))
    assert d.bot.send_message.await_count == 1  # one clean check is not enough (flap damping)
    await check_ops(d, at(30))
    assert sent_texts(d)[1] == "Починилось: импорт сообщений давно не проходил"
    assert state(d)["known"] == {}
    en(d)
    ops(d, INGEST)
    await check_ops(d, at(45))
    ops(d)
    await check_ops(d, at(60))
    await check_ops(d, at(75))
    assert sent_texts(d)[-1] == f"Fixed: {INGEST['message']}"


@pytest.mark.asyncio
async def test_unknown_code_falls_back_to_the_english_message_and_unit_name(tmp_path):
    d = mk(tmp_path)
    ops(d, {"code": "weird_new", "message": "Something odd", "fix": "do x"},
        {"code": "unit_down:second-api", "message": "Service second-api is failed", "fix": "restart it"})
    await check_ops(d, MON_9)
    assert sent_texts(d) == ["Second: Something odd. Починка: `do x`\n\nSecond: сервис second-api остановлен. Починка: `restart it`"]


@pytest.mark.asyncio
async def test_night_problem_waits_for_0700_work_day(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, MON_3)
    assert d.bot.send_message.await_count == 0
    assert len(state(d)["pending"]) == 1
    ops(d, INGEST, {"code": "jobs_failed", "message": "2 vacancy intake(s) failed", "fix": "retry the failed intakes"})
    await check_ops(d, MON_7)  # the first check of the work day
    assert d.bot.send_message.await_count == 1  # merged into one message
    text = sent_texts(d)[0]
    assert "импорт сообщений" in text and "разбор вакансий" in text
    assert state(d)["pending"] == []


@pytest.mark.asyncio
async def test_weekend_problem_waits_for_monday(tmp_path):
    d = mk(tmp_path, now=SAT_9)
    ops(d, INGEST)
    await check_ops(d, SAT_9)
    await check_ops(d, SAT_9 + timedelta(days=1))
    assert d.bot.send_message.await_count == 0
    await check_ops(d, datetime(2026, 11, 9, 7, 0, tzinfo=timezone.utc))
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_problem_raised_and_cleared_over_the_weekend_sends_nothing(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, SAT_9)
    ops(d)
    await check_ops(d, SAT_9 + timedelta(hours=1))
    await check_ops(d, datetime(2026, 11, 9, 7, 0, tzinfo=timezone.utc))
    assert d.bot.send_message.await_count == 0
    assert state(d)["pending"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [SecondUnavailable("down"), SecondAuthFailed("x"), RuntimeError("boom")])
async def test_unreachable_api_sends_nothing(tmp_path, exc):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, MON_9)
    d.second.get_ops_health = AsyncMock(side_effect=exc)
    await check_ops(d, MON_9 + timedelta(minutes=15))
    assert d.bot.send_message.await_count == 1  # only the first one
    assert list(state(d)["known"]) == ["ingest_stale"]  # not read as "fixed"


@pytest.mark.asyncio
async def test_malformed_payload_changes_nothing(tmp_path):
    d = mk(tmp_path)
    d.second.get_ops_health = AsyncMock(return_value=["nope"])
    await check_ops(d, MON_9)
    assert d.bot.send_message.await_count == 0 and d.store.meta_get("ops_state") is None


@pytest.mark.asyncio
async def test_failed_send_is_retried_next_check(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    d.bot.send_message = AsyncMock(side_effect=[RuntimeError("tg"), None])
    await check_ops(d, MON_9)
    assert len(state(d)["pending"]) == 1
    await check_ops(d, MON_9 + timedelta(minutes=15))
    assert d.bot.send_message.await_count == 2 and state(d)["pending"] == []


@pytest.mark.asyncio
async def test_tick_runs_ops_check_on_the_heartbeat_cadence(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await tick(d, MON_9)
    await tick(d, MON_9 + timedelta(minutes=5))
    assert d.second.get_ops_health.await_count == 1
    await tick(d, MON_9 + timedelta(minutes=15))
    assert d.second.get_ops_health.await_count == 2


def test_russian_plural_forms():
    from clawdia.render import tasks_word
    assert [tasks_word(n, "ru") for n in (1, 2, 5, 11, 21, 24)] == [
        "1 задача", "2 задачи", "5 задач", "11 задач", "21 задача", "24 задачи"]
    assert [tasks_word(n, "en") for n in (1, 2)] == ["1 task", "2 tasks"]


@pytest.mark.asyncio
async def test_sync_and_flush_replies_follow_the_language(tmp_path):
    from clawdia.capture import flush_now, sync_now
    d = mk(tmp_path)
    assert await flush_now(d) == "Filed 0" and await flush_now(d, "ru") == "Отправлено 0"
    assert await sync_now(d) == "Filed 0, context unchanged"
    assert await sync_now(d, "ru") == "Отправлено 0, контекст без изменений"


FRI_2330 = datetime(2026, 11, 6, 23, 30, tzinfo=timezone.utc)
MON_2300 = datetime(2026, 11, 2, 23, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_problem_at_2300_waits_for_next_work_day_0700(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, MON_2300)
    assert d.bot.send_message.await_count == 0 and len(state(d)["pending"]) == 1
    await check_ops(d, datetime(2026, 11, 3, 6, 59, tzinfo=timezone.utc))
    assert d.bot.send_message.await_count == 0
    await check_ops(d, datetime(2026, 11, 3, 7, 0, tzinfo=timezone.utc))
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_problem_at_2330_friday_waits_for_monday_0700(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, FRI_2330)
    await check_ops(d, FRI_2330 + timedelta(days=1))
    assert d.bot.send_message.await_count == 0
    await check_ops(d, datetime(2026, 11, 9, 7, 0, tzinfo=timezone.utc))
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_2200_is_night_and_2159_is_not(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, datetime(2026, 11, 2, 21, 59, tzinfo=timezone.utc))
    assert d.bot.send_message.await_count == 1
    d2 = mk(tmp_path / "b")
    ops(d2, INGEST)
    await check_ops(d2, datetime(2026, 11, 2, 22, 0, tzinfo=timezone.utc))
    assert d2.bot.send_message.await_count == 0


def fail(d, exc):
    d.second.get_ops_health = AsyncMock(side_effect=exc)


@pytest.mark.asyncio
async def test_twenty_minutes_of_failed_gets_raise_one_message(tmp_path):
    d = mk(tmp_path)
    fail(d, SecondUnavailable("down"))
    for m in (0, 5, 10, 19):
        await check_ops(d, at(m))
    assert d.bot.send_message.await_count == 0  # under 20 minutes: nothing
    await check_ops(d, at(20))
    await check_ops(d, at(35))
    assert sent_texts(d) == ["Second: Second API не отвечает. Починка: `check that the Second API is running and reachable at CLAWDIA_SECOND_API_URL`"]


@pytest.mark.asyncio
async def test_recovery_after_unreachable_sends_one_fixed_line(tmp_path):
    d = mk(tmp_path)
    fail(d, SecondUnavailable("down"))
    await check_ops(d, at(0))
    await check_ops(d, at(20))
    ops(d)
    await check_ops(d, at(30))
    await check_ops(d, at(45))
    assert sent_texts(d)[1:] == ["Починилось: Second API не отвечает"]
    assert state(d)["known"] == {} and state(d)["first_failure"] is None


@pytest.mark.asyncio
async def test_failure_streak_resets_on_a_good_check(tmp_path):
    d = mk(tmp_path)
    fail(d, SecondUnavailable("down"))
    await check_ops(d, at(0))
    ops(d)
    await check_ops(d, at(15))
    fail(d, SecondUnavailable("down"))
    await check_ops(d, at(20))
    await check_ops(d, at(35))  # only 15 minutes into the new streak
    assert d.bot.send_message.await_count == 0


@pytest.mark.asyncio
async def test_auth_failure_has_its_own_wording(tmp_path):
    d = mk(tmp_path)
    fail(d, SecondAuthFailed("401"))
    await check_ops(d, at(0))
    await check_ops(d, at(20))
    assert sent_texts(d) == ["Second: Second не принимает ключ Клавдии. Починка: `check the Second API logs and CLAWDIA_SECOND_API_TOKEN`"]
    assert list(state(d)["known"]) == ["second_api_rejects_key"]


@pytest.mark.asyncio
async def test_no_synthetic_problem_while_the_stuck_warning_state_is_set(tmp_path):
    d = mk(tmp_path)
    d.store.meta_set("second_down_since", "2026-11-02T08:00:00+00:00")
    fail(d, SecondUnavailable("down"))
    await check_ops(d, at(0))
    await check_ops(d, at(30))
    assert d.bot.send_message.await_count == 0 and state(d)["known"] == {}


@pytest.mark.asyncio
async def test_failed_gets_leave_real_problems_untouched(tmp_path):
    d = mk(tmp_path)
    ops(d, INGEST)
    await check_ops(d, at(0))
    fail(d, SecondUnavailable("down"))
    await check_ops(d, at(15))
    await check_ops(d, at(30))
    assert "ingest_stale" in state(d)["known"] and state(d)["clean"] == {}


@pytest.mark.asyncio
async def test_flapping_code_sends_nothing_new(tmp_path):
    d = mk(tmp_path)
    for i, present in enumerate([True, False, True, False, True, False]):
        ops(d, *([INGEST] if present else []))
        await check_ops(d, at(15 * i))
    assert d.bot.send_message.await_count == 1  # the first alert only; never two clean checks in a row
    assert "ingest_stale" in state(d)["known"]


@pytest.mark.asyncio
async def test_state_is_one_blob_and_old_keys_are_migrated(tmp_path):
    d = mk(tmp_path)
    d.store.meta_set("ops_known", json.dumps({"ingest_stale": "2026-11-02T08:00:00+00:00"}))
    d.store.meta_set("ops_texts", json.dumps({"ingest_stale": {"message": INGEST["message"], "fix": INGEST["fix"]}}))
    ops(d, INGEST)
    await check_ops(d, MON_9)
    assert d.bot.send_message.await_count == 0  # already known from the old keys
    assert set(state(d)) == {"known", "texts", "pending", "clean", "first_failure"}
    assert state(d)["known"]["ingest_stale"] == "2026-11-02T08:00:00+00:00"


@pytest.mark.asyncio
async def test_merged_message_is_capped_with_a_more_line(tmp_path):
    from clawdia.ops_alerts import render
    events = [{"kind": "new", "code": f"c{i}", "message": "m" * 400, "fix": "f" * 400} for i in range(10)]
    text = render(events, "en")
    assert len(text) <= 3500 and text.endswith("more") and text.split("\n\n")[-1].startswith("+")
    assert render(events, "ru").split("\n\n")[-1].startswith("и ещё ")
    small = [{"kind": "new", "code": "a", "message": "m", "fix": "f"}]
    assert "more" not in render(small, "en")
