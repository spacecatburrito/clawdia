from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawdia import review
from clawdia.review import BUDGET, due, group, render, send_review
from clawdia.tagger import TagResult

EM = chr(0x2014)
FRI = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc)  # Lisbon is UTC in November


def mk(tmp_path, now=FRI):
    from tests.test_capture import deps as mkdeps

    d = mkdeps(tmp_path, TagResult("idea", "x", None, {}, 0.9, None, "m"), now=now)
    d.bot = SimpleNamespace(send_message=AsyncMock())
    return d


def at(day, h, m=0):
    return datetime(2026, 11, day, h, m, tzinfo=timezone.utc)


def test_due_friday_window_and_once(tmp_path):
    d = mk(tmp_path)
    assert not due(d, at(19, 18))  # Thursday
    assert not due(d, at(20, 16, 59)) and due(d, at(20, 17)) and due(d, at(20, 18, 30))
    assert not due(d, at(20, 21, 1))
    d.store.meta_set(review.week_key(at(20, 17)), "x")
    assert not due(d, at(20, 18)) and due(d, at(27, 18))


def test_due_off_setting(tmp_path):
    d = mk(tmp_path)
    d.store.meta_set("review_enabled", "0")
    assert not due(d, at(20, 18))
    d.store.meta_set("review_enabled", "")  # left by an Undo: on
    assert due(d, at(20, 18))


async def test_empty_week_sends_nothing_and_counts(tmp_path):
    d = mk(tmp_path)
    assert await send_review(d, at(20, 17)) is False
    d.bot.send_message.assert_not_awaited()
    assert not due(d, at(20, 18))


def miss(i, topic, request="r"):
    return {"id": i, "topic": topic, "topic_key": topic.casefold(), "request": request}


def test_group_order_and_caps():
    rows = [miss(i, f"t{i}") for i in range(10, 3, -1)]  # 7 topics, one miss each, newest first
    rows += [miss(20, "t4"), miss(21, "t4")]
    rows.sort(key=lambda r: -r["id"])
    out = group(rows)
    assert len(out) == 5 and out[0]["topic"] == "t4" and out[0]["count"] == 3
    assert [g["newest_id"] for g in out[1:]] == [10, 9, 8, 7]  # ties: newest first
    many = group([miss(i, "same", f"q{i}") for i in range(9, 0, -1)])
    assert many[0]["examples"] == ["q9", "q8", "q7"]


def test_render_escapes_and_budget():
    groups = [{"topic": "<b>x</b>", "count": 2, "newest_id": 5, "examples": [f"a & b {EM} c"]}]
    text, kb = render(groups, "en")
    assert "&lt;b&gt;x&lt;/b&gt;" in text and "a &amp; b  -  c".replace("  ", " ") in text and EM not in text
    assert text.startswith("Weekly review: what I could not do") and "x" in text
    assert kb == [[("To backlog: <b>x</b>", "missbl:5"), ("Skip", "missno:5")]]
    ru, kbr = render(groups, "ru")
    assert "2 раз" in ru and kbr[0][1][0] == "Не надо"
    big = [{"topic": f"topic {i}", "count": 1, "newest_id": i, "examples": ["z" * 300] * 3} for i in range(1, 6)]
    for lang in ("ru", "en"):
        assert len(render(big, lang)[0]) < BUDGET
    assert len(render([{"topic": "q" * 60, "count": 1, "newest_id": 1, "examples": ["x"]}], "en")[1][0][0]) <= 12 + 20 + 1


async def test_send_review_stores_agent_data_turn(tmp_path):
    d = mk(tmp_path)
    d.store.miss_add(1, None, "Spotify", "play music", "no_tool", at(19, 10))
    assert await send_review(d, at(20, 17)) is True
    kwargs = d.bot.send_message.await_args.kwargs
    assert kwargs["parse_mode"] == "HTML" and "Spotify" in kwargs["text"]
    assert d.bot.send_message.await_args.args[0] == d.settings.owner_id
    turns = d.store.recent_turns(d.settings.owner_id, 5)
    assert not due(d, at(20, 18))
    assert any(t.get("source") == "agent_data" and "Spotify" in t["text"] for t in turns)


async def test_button_closes_topic_and_repeat_is_harmless(tmp_path):
    from clawdia.bot import on_miss_button

    d = mk(tmp_path)
    mid = d.store.miss_add(1, None, "Spotify", "play", "no_tool", at(19, 10))
    d.store.miss_add(1, None, "other", "x", "no_tool", at(19, 11))

    def cb(data):
        kb = SimpleNamespace(inline_keyboard=[
            [SimpleNamespace(text="a", callback_data=f"missbl:{mid}"), SimpleNamespace(text="b", callback_data=f"missno:{mid}")],
            [SimpleNamespace(text="c", callback_data="missbl:99"), SimpleNamespace(text="d", callback_data="missno:99")]])
        return SimpleNamespace(data=data, answer=AsyncMock(), from_user=SimpleNamespace(id=1),
                               message=SimpleNamespace(reply_markup=kb, edit_reply_markup=AsyncMock()))

    q = cb(f"missbl:{mid}")
    await on_miss_button(q, d)
    assert d.store.miss_get(mid)["status"] == "backlog"
    markup_arg = q.message.edit_reply_markup.await_args.kwargs["reply_markup"]
    assert [b.callback_data for row in markup_arg.inline_keyboard for b in row] == ["missbl:99", "missno:99"]
    q2 = cb(f"missno:{mid}")
    await on_miss_button(q2, d)
    assert d.store.miss_get(mid)["status"] == "backlog"
    q2.message.edit_reply_markup.assert_not_awaited()
    await on_miss_button(cb("missbl:junk"), d)  # junk changes nothing


def test_buttons_are_routed():
    import inspect
    from clawdia import bot
    src = inspect.getsource(bot.build_router)
    assert '"missbl:"' in src and '"missno:"' in src


def test_blank_examples_give_no_blockquote():
    g = [{"topic": "x", "count": 2, "newest_id": 2, "examples": ["", "  "]}]
    assert "<blockquote" not in render(g, "ru")[0]


async def test_failed_send_leaves_the_week_open_for_a_retry(tmp_path):
    d = mk(tmp_path)
    d.store.miss_add(1, None, "Spotify", "play music", "no_tool", at(19, 10))
    d.bot.send_message = AsyncMock(side_effect=RuntimeError("telegram down"))
    assert await send_review(d, at(20, 17)) is False
    assert due(d, at(20, 18))
    d.bot.send_message = AsyncMock()
    assert await send_review(d, at(20, 18)) is True
    assert not due(d, at(20, 19))


async def test_check_logs_a_failed_send_and_empty_week_is_marked(tmp_path):
    d = mk(tmp_path)
    await review.check(d, at(20, 17))  # no misses: marked, nothing sent
    assert not due(d, at(20, 18))
    d2 = mk(tmp_path / "b")
    d2.store.miss_add(1, None, "Spotify", "play", "no_tool", at(19, 10))
    d2.bot.send_message = AsyncMock(side_effect=RuntimeError("down"))
    await review.check(d2, at(20, 17))  # logged, not raised
    assert due(d2, at(20, 18))


async def test_button_does_not_close_misses_logged_after_the_review(tmp_path):
    from clawdia.bot import on_miss_button
    d = mk(tmp_path)
    mid = d.store.miss_add(1, None, "Spotify", "play", "no_tool", at(19, 10))
    later = d.store.miss_add(1, None, "Spotify", "again", "no_tool", at(21, 10))
    kb = SimpleNamespace(inline_keyboard=[[SimpleNamespace(text="a", callback_data=f"missbl:{mid}")]])
    q = SimpleNamespace(data=f"missbl:{mid}", answer=AsyncMock(), from_user=SimpleNamespace(id=1),
                        message=SimpleNamespace(reply_markup=kb, edit_reply_markup=AsyncMock(), date=at(20, 17)))
    await on_miss_button(q, d)
    assert d.store.miss_get(mid)["status"] == "backlog" and d.store.miss_get(later)["status"] == "open"
