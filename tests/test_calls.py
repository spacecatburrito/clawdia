import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawdia import calls
from clawdia.calls import CHECK_EVERY, check_waits, match
from clawdia.converse_actions import Outcome
from clawdia.second_client import SecondUnavailable
from clawdia.tagger import TagResult

NOW = datetime(2026, 11, 20, 19, 0, tzinfo=timezone.utc)


def row(i, name="Wexford intro and Sam Lee 2026-11-12  [drive:abc]"):
    return {"id": i, "uploaded_at": "x", "untrusted": {"filename": name}}


def wait(**kw):
    return {"id": 1, "baseline_id": 437, "title_hint": None, **kw}


def mk(tmp_path, items=None, pull=None, now=NOW):
    from tests.test_capture import deps as mkdeps

    d = mkdeps(tmp_path, TagResult("idea", "x", None, {}, 0.9, None, "m"), now=now)
    d.bot = SimpleNamespace(send_message=AsyncMock())
    d.second.pull_transcripts = pull or AsyncMock(return_value={"status": "ok"})
    reader = SimpleNamespace(call=AsyncMock(return_value={"items": [row(438)] if items is None else items}),
                             close=AsyncMock())
    d.reader = reader
    d.second_reader_fn = lambda: reader
    return d


def open_wait(d, hint=None, ttl=timedelta(hours=3), msg_id=77, text="how did the call go?"):
    turn = d.store.add_turn(1, "user", text, processed=False, at=NOW - timedelta(minutes=5))
    return d.store.wait_open(1, turn, msg_id, hint, 437, NOW - timedelta(minutes=5), ttl, text)


def fake_agent(monkeypatch, outcome=Outcome("Wexford call: summary", None), seen=None):
    async def run(message, deps, **kw):
        if seen is not None:
            seen.append((message, kw))
        return outcome
    monkeypatch.setattr(calls, "run_agent", run)


def test_match_ignores_ids_at_or_below_baseline():
    assert match(wait(), [row(437), row(436)]) is None  # a re-filled old transcript is not new


def test_match_with_hint_is_case_insensitive_substring():
    items = [row(438, "Other call"), row(439, "WEXFORD intro")]
    assert match(wait(title_hint="wexford"), items)["id"] == 439
    assert match(wait(title_hint="zzz"), items) is None


def test_match_without_hint_takes_lowest_new_id():
    assert match(wait(), [row(440), row(438), row(439)])["id"] == 438


@pytest.mark.asyncio
async def test_check_skips_when_no_waits(tmp_path):
    d = mk(tmp_path)
    await check_waits(d, NOW)
    d.second.pull_transcripts.assert_not_awaited()
    d.reader.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_runs_at_most_every_10_minutes(tmp_path, monkeypatch):
    d = mk(tmp_path, items=[])
    open_wait(d)
    await check_waits(d, NOW)
    await check_waits(d, NOW + CHECK_EVERY - timedelta(seconds=1))
    assert d.second.pull_transcripts.await_count == 1
    await check_waits(d, NOW + CHECK_EVERY)
    assert d.second.pull_transcripts.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("pull", [AsyncMock(return_value={"status": "busy"}),
                                  AsyncMock(return_value={"status": "cooldown"}),
                                  AsyncMock(side_effect=SecondUnavailable())])
async def test_busy_cooldown_or_error_pull_still_reads(tmp_path, pull):
    d = mk(tmp_path, items=[], pull=pull)
    open_wait(d)
    await check_waits(d, NOW)
    d.reader.call.assert_awaited_once_with("list_transcripts", {"days": 1, "limit": 50})


@pytest.mark.asyncio
async def test_read_failure_leaves_waits_open(tmp_path):
    d = mk(tmp_path)
    d.reader.call = AsyncMock(side_effect=SecondUnavailable())
    w = open_wait(d, ttl=timedelta(minutes=1))  # just expired (inside the grace of one check): stays open
    await check_waits(d, NOW)
    assert d.store.wait_get(w)["status"] == "open"
    d.bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reader_missing", [False, True])
async def test_long_overdue_wait_expires_even_when_nothing_was_read(tmp_path, reader_missing):
    d = mk(tmp_path)
    d.reader.call = AsyncMock(side_effect=SecondUnavailable())
    if reader_missing:
        d.second_reader_fn = None
    w = open_wait(d, ttl=timedelta(minutes=1))  # expires NOW-4min; grace ends NOW+6min
    await check_waits(d, NOW + timedelta(minutes=7))
    assert d.store.wait_get(w)["status"] == "expired"
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_match_two_days_after_an_outage_is_not_answered(tmp_path, monkeypatch):
    d = mk(tmp_path)  # the list holds a new transcript, but the wait expired long ago
    w = open_wait(d)
    seen = []
    fake_agent(monkeypatch, seen=seen)
    await check_waits(d, NOW + timedelta(days=2))
    assert seen == [] and d.store.wait_get(w)["status"] == "expired"
    assert d.bot.send_message.await_args.kwargs["text"].startswith("No transcript of that call arrived")


@pytest.mark.asyncio
async def test_one_failed_expiry_send_does_not_stop_the_other_waits(tmp_path):
    d = mk(tmp_path, items=[])
    a, b = open_wait(d, ttl=timedelta(minutes=1)), open_wait(d, ttl=timedelta(minutes=1))
    d.bot.send_message = AsyncMock(side_effect=[RuntimeError("blocked"), None])
    await check_waits(d, NOW)
    assert [d.store.wait_get(w)["status"] for w in (a, b)] == ["expired", "expired"]
    assert d.bot.send_message.await_count == 2


@pytest.mark.asyncio
async def test_cap_with_no_outcome_and_no_cap_sends_one_failure_line(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d)
    fake_agent(monkeypatch, outcome=None)
    monkeypatch.setattr(calls, "cap_reached", lambda deps, now: False)
    await check_waits(d, NOW)
    await check_waits(d, NOW + timedelta(hours=1))
    assert d.bot.send_message.await_count == 1
    assert d.bot.send_message.await_args.kwargs["text"] == (
        "Transcript #438 (Wexford intro and Sam Lee 2026-11-12) is in Second, but the summary did not work out; ask me again.")
    assert d.store.wait_get(w)["status"] == "answered"


@pytest.mark.asyncio
async def test_expired_wait_sends_one_line_once(tmp_path):
    d = mk(tmp_path, items=[])
    w = open_wait(d, ttl=timedelta(minutes=1))
    await check_waits(d, NOW)
    await check_waits(d, NOW + timedelta(hours=1))
    assert d.store.wait_get(w)["status"] == "expired"
    assert d.bot.send_message.await_count == 1
    assert d.bot.send_message.await_args.kwargs["text"] == "No transcript of that call arrived in 3 hours; it was probably not recorded."


@pytest.mark.asyncio
async def test_match_wins_over_expiry(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d, ttl=timedelta(minutes=1))
    fake_agent(monkeypatch)
    await check_waits(d, NOW)
    assert d.store.wait_get(w)["status"] == "answered"
    assert [c.kwargs["text"] for c in d.bot.send_message.await_args_list] == ["Wexford call: summary"]


@pytest.mark.asyncio
async def test_one_answer_per_wait_under_two_concurrent_checks(tmp_path, monkeypatch):
    d = mk(tmp_path)
    open_wait(d)
    seen = []
    fake_agent(monkeypatch, seen=seen)
    await asyncio.gather(check_waits(d, NOW), check_waits(d, NOW))
    assert len(seen) == 1 and d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_answer_replies_to_stored_message_with_stored_question(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d, text="итоги звонка?")
    seen = []
    fake_agent(monkeypatch, seen=seen)
    await check_waits(d, NOW)
    message, kw = seen[0]
    assert message.text == "итоги звонка?" and message.chat.id == 1 and message.reply_to_message is None
    assert kw["followup_for"] == w and "#438" in kw["followup_note"]
    call = d.bot.send_message.await_args
    assert call.kwargs["reply_to_message_id"] == 77 and call.kwargs["text"] == "Wexford call: summary"
    assert d.store.recent_turns(1, 8)[-1]["role"] == "bot"


@pytest.mark.asyncio
async def test_reply_refused_falls_back_to_plain_message(tmp_path, monkeypatch):
    d = mk(tmp_path)
    open_wait(d)
    fake_agent(monkeypatch)
    d.bot.send_message = AsyncMock(side_effect=[RuntimeError("gone"), None])
    await check_waits(d, NOW)
    assert "reply_to_message_id" not in d.bot.send_message.await_args.kwargs


@pytest.mark.asyncio
async def test_cap_hit_sends_reference_once_and_closes(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d, text="how did the call go?")
    fake_agent(monkeypatch, outcome=None)
    monkeypatch.setattr(calls, "cap_reached", lambda deps, now: True)
    await check_waits(d, NOW)
    await check_waits(d, NOW + timedelta(hours=1))
    assert d.bot.send_message.await_count == 1
    text = d.bot.send_message.await_args.kwargs["text"]
    assert text == "Transcript #438 (Wexford intro and Sam Lee 2026-11-12) is in Second; the daily spend cap stops the summary today."
    assert d.store.wait_get(w)["status"] == "answered"
    assert d.store.recent_turns(1, 8)[-1]["source"] == "agent_data"


@pytest.mark.asyncio
async def test_answer_exception_reopens_wait(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d)

    async def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(calls, "run_agent", boom)
    await check_waits(d, NOW)
    assert d.store.wait_get(w)["status"] == "open"
    d.bot.send_message.assert_not_awaited()


def test_startup_resets_stale_answering(tmp_path):
    d = mk(tmp_path)
    w = open_wait(d)
    assert d.store.wait_claim(w, 438, NOW - timedelta(minutes=20))
    assert d.store.waits_reset_answering(NOW) == 1
    assert d.store.wait_get(w)["status"] == "open"


def test_startup_reset_takes_a_fresh_claim(tmp_path):
    d = mk(tmp_path)
    w = open_wait(d)
    assert d.store.wait_claim(w, 438, NOW - timedelta(minutes=1))
    assert d.store.waits_reset_answering(NOW) == 0  # the 15-minute variant leaves it
    assert d.store.waits_reset_answering(NOW, older_than=None) == 1
    assert d.store.wait_get(w)["status"] == "open"


@pytest.mark.asyncio
async def test_recovery_skips_the_standin_turn_after_a_failure(tmp_path, monkeypatch):
    from clawdia.converse import recover_unprocessed

    d = mk(tmp_path)
    open_wait(d)
    d.store.mark_turn(1, processed=True)  # her original question was answered

    async def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(calls, "run_agent", boom)
    await check_waits(d, NOW)
    assert not [t for t in d.store.unprocessed_turns() if t["source"] == "followup"]
    d.store.add_turn(1, "user", "q", processed=False, at=NOW, source="followup")  # even an unprocessed one
    assert await recover_unprocessed(d) == 0


@pytest.mark.asyncio
async def test_bookkeeping_failure_after_send_keeps_wait_answered(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d)
    fake_agent(monkeypatch)
    real = d.store.add_turn

    def flaky(chat_id, role, *a, **k):
        if role == "bot":
            raise RuntimeError("db")
        return real(chat_id, role, *a, **k)
    monkeypatch.setattr(d.store, "add_turn", flaky)
    await check_waits(d, NOW)
    assert d.store.wait_get(w)["status"] == "answered"
    await check_waits(d, NOW + timedelta(hours=1))
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_pruned_question_turn_still_answers(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d, text="итоги звонка?")
    with d.store._lock, d.store._conn:
        d.store._conn.execute("DELETE FROM turns WHERE source = 'typed'")
    seen = []
    fake_agent(monkeypatch, seen=seen)
    await check_waits(d, NOW)
    assert seen[0][0].text == "итоги звонка?" and d.store.wait_get(w)["status"] == "answered"


@pytest.mark.asyncio
async def test_three_failures_close_the_wait_with_one_failure_line(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d)

    async def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(calls, "run_agent", boom)
    for n in range(5):
        await check_waits(d, NOW + CHECK_EVERY * n)
    assert d.store.wait_get(w)["status"] == "answered" and d.store.wait_get(w)["attempts"] == 3
    assert d.bot.send_message.await_count == 1
    assert "the summary did not work out" in d.bot.send_message.await_args.kwargs["text"]
    assert d.store.recent_turns(1, 8)[-1]["source"] == "agent_data"


@pytest.mark.asyncio
async def test_failure_past_expiry_closes_at_once(tmp_path, monkeypatch):
    d = mk(tmp_path)
    w = open_wait(d, ttl=timedelta(minutes=1))

    async def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(calls, "run_agent", boom)
    await check_waits(d, NOW)
    assert d.store.wait_get(w)["status"] == "answered" and d.bot.send_message.await_count == 1


def test_followup_note_forbids_repeating_her_changes():
    assert "only answer about the call" in calls.NOTE
