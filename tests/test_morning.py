import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawdia import morning
from clawdia.morning import due, render, send_digest
from clawdia.tagger import TagResult

MON = datetime(2026, 11, 23, 9, 31, tzinfo=timezone.utc)  # Monday, Lisbon = UTC in November
EM = chr(0x2014)


def item(id, headline, details="d", link=None, importance="normal", **untrusted):
    return {"id": id, "headline": headline, "details": details, "link": link,
            "importance": importance, "untrusted": untrusted}


def block(*items, error=False, total=None, **extra):
    return {"items": list(items), "total": len(items) if total is None else total, "error": error, **extra}


def blocks(calls=(), silent=(), promises=(), vacancies=(), **errors):
    out = {"calls": block(*calls), "silent": block(*silent), "promises": block(*promises),
           "vacancies": block(*vacancies, skipped=0), "freshness": {"latest_message_age_minutes": 5}}
    for name, flag in errors.items():
        out[name]["error"] = flag
    return out


MAP = """Built automatically at 08:10, energy normal.

## Decisions for you
- Calendar was unreadable <b>x</b>
- Energy set to normal

## Top-4 (4 open)

1. **Ship the digest** - today
2. **Two**
"""

CALL = item("call:1", "Call at 10:30", "Starts 10:30 Lisbon, 30 min", "https://cal/1", "must_see", title="Wexford sync", location=None)
CALL2 = item("call:2", "Call at 15:00", "Starts 15:00 Lisbon", None, "normal", title="Second call", location=None)
LATE = item("task:9", "Overdue 3 days", "Deadline 2026-11-20", None, "must_see", title="Send invoice", projects=[])
SILENT = item("silent:4", "Silent 4 days", "no reply", None, "normal", chat_title="Pinecrest chat", projects=["Pinecrest"])
VAC = item("vacancy:5", "Vacancy triaged: apply", "Verdict apply, rating 8", "https://notion.so/v", "normal",
           company="Acme", role="Head of Product")


def test_order_red_first_then_topics_and_closing():
    text, kb = render(blocks(calls=[CALL, CALL2], silent=[SILENT], promises=[LATE], vacancies=[VAC]), MAP, "en", MON)
    assert kb is None
    reds = [ln for ln in text.splitlines() if ln.startswith("🔴")]
    assert len(reds) == 3
    assert "Ship the digest" in reds[0] and "1 overdue: Send invoice" in reds[1] and "Wexford sync" in reds[2]
    pos = [text.index(w) for w in ("Calls", "Pinecrest chat", "promises", "Vacancies")]
    assert pos == sorted(pos) and text.index("🔴") < pos[0]
    assert text.rstrip().endswith("Ask me about any item.")
    assert text.count("<blockquote expandable>") == 4 and text.count("</blockquote>") == 4
    assert "Wexford sync" in text and "Second call" in text  # first two calls in the headline


def test_decisions_lines_when_room():
    text, _ = render(blocks(), MAP, "en", MON)
    reds = [ln for ln in text.splitlines() if ln.startswith("🔴")]
    assert len(reds) == 3 and "Ship the digest" in reds[0]
    assert "Calendar was unreadable &lt;b&gt;x&lt;/b&gt;" in reds[1] and "Energy set to normal" in reds[2]


def test_five_overdue_and_soon_call_all_show_aggregated():
    late = [item(f"t{i}", "Overdue 2 days", "d", None, "must_see", title=f"Task {i}", projects=[]) for i in range(5)]
    text, _ = render(blocks(promises=late, calls=[CALL]), MAP, "en", MON)
    reds = [ln for ln in text.splitlines() if ln.startswith("🔴")]
    assert len(reds) == 3
    assert "Ship the digest" in reds[0] and reds[1].startswith("🔴 5 overdue: Task 0, Task 1") and "Wexford sync" in reds[2]


def test_total_limit_with_many_topics():
    t = "a" * 199
    def many(n, **u):
        return [item(f"x{k}", "Call at 10:00", "x", None, "normal", **u) for k in range(n)]
    for n in range(1, 40, 3):
        b = {"calls": block(*many(n, title=t)), "silent": block(*many(30, chat_title="b" * 199)),
             "promises": block(*many(30, title="c" * 199)), "vacancies": block(*many(30, company="d" * 199, role="e"))}
        text, _ = render(b, "## Top-4\n\n1. x\n", "en", MON)
        assert len(text) <= 3500 and text.count("<blockquote expandable>") == text.count("</blockquote>")


def test_missing_map_takes_first_red_slot_and_button():
    text, kb = render(blocks(promises=[LATE], calls=[CALL]), None, "en", MON)
    reds = [ln for ln in text.splitlines() if ln.startswith("🔴")]
    assert reds[0] == "🔴 No day map yet" and "1 overdue: Send invoice" in reds[1] and "Wexford sync" in reds[2]
    assert kb == [[("Build on laptop", "focusreq:open")]]
    ru, kb_ru = render(blocks(), None, "ru", MON)
    assert "🔴 Карты дня нет" in ru and kb_ru == [[("Собрать на ноутбуке", "focusreq:open")]]
    assert "Спроси меня про любой пункт." in ru


def test_empty_topics_omitted_and_nothing_when_all_empty():
    text, _ = render(blocks(vacancies=[VAC]), MAP, "en", MON)
    assert "Calls" not in text and "Silent" not in text and "promises" not in text.lower()
    assert render(blocks(), "## Top-4\n\n", "en", MON) == ("", None)  # no red line, no content
    assert render({}, "## Top-4\n\n", "en", MON) == ("", None)


def test_error_block_one_short_line_only_without_content():
    text, _ = render(blocks(vacancies=[VAC], calls=[], promises=[], silent=[]) | {"calls": block(error=True)}, MAP, "en", MON)
    assert "Calls today: not available right now" in text
    both = blocks(calls=[CALL])
    both["calls"]["error"] = True
    text2, _ = render(both, MAP, "en", MON)
    assert "not available" not in text2
    only_err = {"calls": block(error=True), "silent": block(error=True), "promises": block(), "vacancies": block()}
    assert render(only_err, "## Top-4\n\n", "en", MON) == ("", None)  # partial errors alone never trigger a send
    all_err = {k: block(error=True) for k in ("calls", "silent", "promises", "vacancies")}
    assert render(all_err, "## Top-4\n\n", "en", MON) == ("Digest unavailable: Second did not answer.", None)
    assert render(all_err, "## Top-4\n\n", "ru", MON)[0] == "Сводка недоступна: Second не ответил."
    assert render(all_err, MAP, "en", MON)[0].startswith("🔴")  # a red line still wins


def test_escaping_including_blockquote_breakout():
    evil = item("task:1", "Overdue 1 days", "x", None, "must_see", title="</blockquote><b>boom</b> & <a href=x>")
    text, _ = render(blocks(promises=[evil], calls=[item("c", "Call at 10:00", title="</blockquote><i>")]), MAP, "en", MON)
    assert "<b>boom" not in text and "<i>" not in text and "<a href=x>" not in text
    assert text.count("</blockquote>") == text.count("<blockquote expandable>")
    assert "&lt;/blockquote&gt;" in text and "&amp;" in text
    # a hostile map line is escaped too
    text2, _ = render(blocks(), "## Top-4\n\n1. <script>x</script>\n", "en", MON)
    assert "<script>" not in text2 and "&lt;script&gt;" in text2


def test_links_only_http_and_escaped():
    bad = item("v", "Vacancy triaged: apply", "d", 'javascript:alert(1)"', "normal", company="A", role="B")
    ok = item("v2", "Vacancy triaged: apply", "d", 'https://x.io/a?b="c"&d', "normal", company="A", role="B")
    text, _ = render(blocks(vacancies=[bad, ok]), MAP, "en", MON)
    assert "javascript" not in text and 'href="https://x.io/a?b=&quot;c&quot;&amp;d"' in text


def test_budget_with_huge_inputs_keeps_all_red_lines():
    huge = "Z" * 5000
    many = [item(f"task:{i}", "Overdue 9 days", "d" * 300, None, "must_see", title=huge, projects=[]) for i in range(40)]
    calls = [item(f"c{i}", "Call at 10:00", "d", None, "must_see", title=huge, location=None) for i in range(40)]
    silent = [item(f"s{i}", "Silent 5 days", "d", None, "normal", chat_title=huge, projects=[huge]) for i in range(40)]
    vac = [item(f"v{i}", "Vacancy triaged: apply", "d", "https://n.so/" + "q" * 400, "normal", company=huge, role=huge)
           for i in range(40)]
    big_map = "## Decisions for you\n" + "".join(f"- {huge} {i}\n" for i in range(10)) + "\n## Top-4\n\n1. " + huge + "\n"
    text, _ = render(blocks(calls=calls, silent=silent, promises=many, vacancies=vac), big_map, "en", MON)
    assert len(text) <= 3500
    import re
    visible = re.sub(r"<blockquote expandable>.*?</blockquote>", "", text, flags=re.S)
    assert len(visible) <= 900
    assert sum(1 for ln in text.splitlines() if ln.startswith("🔴")) == 3
    assert "+" in text and "more" in text  # details trimmed with a count
    assert text.rstrip().endswith("Ask me about any item.")
    assert text.count("<blockquote expandable>") == text.count("</blockquote>")


def test_budget_with_escape_heavy_input():
    amp = "&" * 3000
    prom = [item(f"t{i}", "Overdue 1 days", "x", None, "must_see", title=amp, projects=[]) for i in range(30)]
    text, _ = render(blocks(promises=prom), MAP, "en", MON)
    assert len(text) <= 3500 and text.count("<blockquote expandable>") == text.count("</blockquote>")


def test_no_em_dash_anywhere():
    for lang in ("en", "ru"):
        text, _ = render(blocks(calls=[CALL], silent=[SILENT], promises=[LATE], vacancies=[VAC]), None, lang, MON)
        assert EM not in text


# -- schedule -------------------------------------------------------------------------------------


def mk(tmp_path, now=MON, reader=None, map_text=None):
    from tests.test_capture import deps as mkdeps

    d = mkdeps(tmp_path, TagResult("idea", "x", None, {}, 0.9, None, "m"), now=now)
    d.bot = SimpleNamespace(send_message=AsyncMock())
    d.second.refresh_sources = AsyncMock(return_value={"refreshed": []})
    d.settings.focus_dir = tmp_path / "focus"
    d.settings.focus_dir.mkdir()
    if map_text is not None:
        (d.settings.focus_dir / f"{now.date().isoformat()}.md").write_text(map_text, encoding="utf-8")
    data = blocks(calls=[CALL], promises=[LATE])
    d.reader = reader or SimpleNamespace(call=AsyncMock(return_value=data), close=AsyncMock())
    d.second_reader_fn = lambda: d.reader
    return d


def at(h, m=0, day=23):
    return datetime(2026, 11, day, h, m, tzinfo=timezone.utc)


def test_due_window_weekday_and_once(tmp_path):
    d = mk(tmp_path)
    assert not due(d, at(9, 29)) and due(d, at(9, 30)) and due(d, at(11, 59)) and not due(d, at(12, 0))
    assert not due(d, at(9, 30, day=21)) and not due(d, at(9, 30, day=22))  # Saturday, Sunday
    d.store.meta_set("digest_sent:2026-11-23", "x")
    assert not due(d, at(10, 0))
    assert due(d, at(10, 0, day=24))


def test_due_settings(tmp_path):
    d = mk(tmp_path)
    d.store.meta_set("digest_enabled", "0")
    assert not due(d, at(10, 0))
    d.store.meta_set("digest_enabled", "1")
    d.store.meta_set("digest_time", "10:00")
    assert not due(d, at(9, 45)) and due(d, at(10, 0))
    d.store.meta_set("digest_days", "tue,wed")
    assert not due(d, at(10, 0)) and due(d, at(10, 0, day=24))
    d.store.meta_set("digest_time", "garbage")  # a bad value falls back to the default
    assert due(d, at(9, 30, day=24))


async def test_send_digest_sends_html_to_owner_and_stores_agent_data(tmp_path):
    d = mk(tmp_path)
    await send_digest(d, MON)
    d.second.refresh_sources.assert_awaited_once()
    d.reader.call.assert_awaited_once_with("morning_digest", {"since_hours": 24})
    d.reader.close.assert_awaited()
    kwargs = d.bot.send_message.await_args.kwargs
    assert d.bot.send_message.await_args.args == (1,) and kwargs["parse_mode"] == "HTML"
    assert "🔴" in kwargs["text"] and "No day map yet" in kwargs["text"] or "Карты дня нет" in kwargs["text"]
    assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "focusreq:open"
    assert d.store.recent_turns(1, 5)[-1]["source"] == "agent_data"
    assert d.store.meta_get("digest_sent:2026-11-23")


async def test_send_digest_with_map_has_no_button(tmp_path):
    d = mk(tmp_path, map_text=MAP)
    await send_digest(d, MON)
    kwargs = d.bot.send_message.await_args.kwargs
    assert kwargs.get("reply_markup") is None and "Ship the digest" in kwargs["text"]


async def test_refresh_failure_is_ignored(tmp_path):
    d = mk(tmp_path, map_text=MAP)
    d.second.refresh_sources = AsyncMock(side_effect=RuntimeError("down"))
    await send_digest(d, MON)
    assert d.bot.send_message.await_count == 1


async def test_read_failure_still_sends_the_map_lines(tmp_path):
    d = mk(tmp_path, map_text=MAP, reader=SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("x")), close=AsyncMock()))
    await send_digest(d, MON)
    text = d.bot.send_message.await_args.kwargs["text"]
    assert "Ship the digest" in text and ("not available" in text or "недоступн" in text)


async def test_nothing_sent_when_empty(tmp_path):
    d = mk(tmp_path, map_text="## Top-4\n\n", reader=SimpleNamespace(call=AsyncMock(return_value=blocks()), close=AsyncMock()))
    await send_digest(d, MON)
    d.bot.send_message.assert_not_awaited()
    assert d.store.meta_get("digest_sent:2026-11-23")  # and not retried every minute


async def test_check_once_across_double_tick_and_restart(tmp_path):
    from clawdia.storage import Store
    d = mk(tmp_path, map_text=MAP)
    await morning.check(d, MON)
    await morning.check(d, MON + timedelta(minutes=1))
    assert d.bot.send_message.await_count == 1
    d.store.close()
    d.store = Store(tmp_path / "c.sqlite")
    await morning.check(d, MON + timedelta(minutes=2))
    assert d.bot.send_message.await_count == 1


async def test_check_catch_up_before_noon_and_skip_after(tmp_path):
    d = mk(tmp_path, map_text=MAP)
    await morning.check(d, at(11, 30))
    assert d.bot.send_message.await_count == 1
    d2 = mk(tmp_path / "b", map_text=MAP) if (tmp_path / "b").mkdir() is None else None
    await morning.check(d2, at(12, 5))
    d2.bot.send_message.assert_not_awaited()


async def test_check_weekend_and_disabled_skip(tmp_path):
    d = mk(tmp_path, map_text=MAP)
    await morning.check(d, at(10, 0, day=21))
    d.store.meta_set("digest_enabled", "0")
    await morning.check(d, at(10, 0))
    d.bot.send_message.assert_not_awaited()


async def test_send_failure_marks_sent_first_so_it_never_loops(tmp_path):
    d = mk(tmp_path, map_text=MAP)
    d.bot.send_message = AsyncMock(side_effect=RuntimeError("tg down"))
    with pytest.raises(RuntimeError):
        await morning.check(d, MON)
    assert d.store.meta_get("digest_sent:2026-11-23")
    assert not due(d, MON + timedelta(minutes=1))


def test_loop_has_morning_step():
    import inspect
    from clawdia import loops
    assert '"morning"' in inspect.getsource(loops.tick)


async def test_since_hours_monday_after_friday_is_72_and_stamp_only_after_send(tmp_path):
    d = mk(tmp_path, map_text=MAP)
    d.store.meta_set("digest_last_sent_at", datetime(2026, 11, 20, 9, 31, tzinfo=timezone.utc).isoformat())
    await send_digest(d, MON)
    d.reader.call.assert_awaited_once_with("morning_digest", {"since_hours": 72})
    assert d.store.meta_get("digest_last_sent_at") == MON.isoformat()
    # no stamp: 24; a failed send leaves the stamp alone
    d2 = mk(tmp_path / "x") if (tmp_path / "x").mkdir() is None else None
    d2.bot.send_message = AsyncMock(side_effect=RuntimeError("tg"))
    with pytest.raises(RuntimeError):
        await send_digest(d2, MON)
    d2.reader.call.assert_awaited_once_with("morning_digest", {"since_hours": 24})
    assert not d2.store.meta_get("digest_last_sent_at")


def test_since_hours_clamps(tmp_path):
    d = mk(tmp_path)
    d.store.meta_set("digest_last_sent_at", (MON - timedelta(hours=2)).isoformat())
    assert morning.since_hours(d.store, MON) == 24
    d.store.meta_set("digest_last_sent_at", (MON - timedelta(days=20)).isoformat())
    assert morning.since_hours(d.store, MON) == 72


async def test_read_failure_without_map_lines_sends_unavailable(tmp_path):
    d = mk(tmp_path, map_text="## Top-4\n\n", reader=SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("x")), close=AsyncMock()))
    await send_digest(d, MON)
    assert d.bot.send_message.await_args.kwargs["text"] in ("Digest unavailable: Second did not answer.", "Сводка недоступна: Second не ответил.")


async def test_stamp_not_advanced_when_vacancies_block_failed(tmp_path):
    failed = blocks(calls=[CALL], promises=[LATE], vacancies=[])
    failed["vacancies"]["error"] = True
    d = mk(tmp_path, reader=SimpleNamespace(call=AsyncMock(return_value=failed), close=AsyncMock()))
    await send_digest(d, MON)
    d.bot.send_message.assert_awaited_once()  # the digest still goes out
    assert not d.store.meta_get("digest_last_sent_at")


async def test_stamp_not_advanced_when_vacancies_block_missing(tmp_path):
    data = blocks(calls=[CALL])
    del data["vacancies"]
    d = mk(tmp_path, reader=SimpleNamespace(call=AsyncMock(return_value=data), close=AsyncMock()))
    await send_digest(d, MON)
    assert not d.store.meta_get("digest_last_sent_at")


def test_since_hours_rounds_up(tmp_path):
    d = mk(tmp_path)
    d.store.meta_set("digest_last_sent_at", (MON - timedelta(hours=48, minutes=20)).isoformat())
    assert morning.since_hours(d.store, MON) == 49
