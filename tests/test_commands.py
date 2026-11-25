import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawdia.bot import (build_router, handle_done, handle_drop, on_done, on_drop, on_flush, on_help,
                         on_inbox, on_list, on_show, on_start, on_sync, render_inbox, render_list, render_show)
from clawdia.storage import Store
from clawdia.tagger import TagResult
from tests.test_capture import deps

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def store(tmp_path):
    s = Store(tmp_path / "c.sqlite"); s.ensure_schema(); return s


def test_render_list_newest_first_and_paged(tmp_path):
    s = store(tmp_path)
    ids = [s.create_item(kind="film", status="new", title=f"Film {i}") for i in range(25)]
    out = render_list(s.list_items(kind="film", status="new", limit=1000), "film", 1)
    assert out.splitlines()[0] == "film · 25 open" and f"#{ids[-1]} · Film 24" in out.splitlines()[1] and "page 1/2" in out
    page2 = render_list(s.list_items(kind="film", status="new", limit=1000), "film", 2)
    assert f"#{ids[0]} · Film 0" in page2 and "Film 24" not in page2 and "page 2/2" in page2
    assert "page 2/2" in render_list(s.list_items(kind="film", limit=1000), "film", 99)  # clamped
    s.close()


def test_render_list_empty_and_single_page_has_no_footer(tmp_path):
    s = store(tmp_path)
    assert render_list([], "buy", 1) == "buy · 0 open"
    s.create_item(kind="buy", status="new", title="Oil")
    assert "page" not in render_list(s.list_items(kind="buy"), "buy", 1)
    s.close()


def test_render_list_truncates_to_telegram_limit():
    items = [{"id": i, "title": "x" * 300, "status": "new"} for i in range(20)]
    out = render_list(items, "idea", 1)
    assert len(out) <= 4000 and out.endswith("...")


def test_handle_done_unknown_id(tmp_path):
    s = store(tmp_path)
    assert handle_done(s, "99999") == "No item #99999" and handle_done(s, "abc") == "Usage: /done <id>"
    assert handle_done(s, "") == "Usage: /done <id>" and handle_drop(s, "x") == "Usage: /drop <id>"
    i = s.create_item(kind="buy", status="new", title="Olive oil")
    assert handle_done(s, str(i)) == f"Done #{i} · Olive oil" and s.get_item(i)["status"] == "done"
    s.close()


def test_handle_drop_and_repeat_reports_current_status(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="buy", status="new", title="Oil")
    s.add_reminder(i, NOW)
    assert handle_drop(s, str(i)) == f"Dropped #{i} · Oil"
    assert s.get_item(i)["status"] == "dismissed" and s.due_reminders(NOW) == []
    assert handle_drop(s, str(i)) == f"Already dropped #{i} · Oil"
    assert handle_done(s, str(i)) == f"Already dropped #{i} · Oil" and s.get_item(i)["status"] == "dismissed"
    j = s.create_item(kind="buy", status="new", title="Salt")
    handle_done(s, str(j))
    assert handle_done(s, str(j)) == f"Already done #{j} · Salt"
    s.close()


def test_handle_done_on_filed_item_warns_second_stays_open(tmp_path):
    s = store(tmp_path)
    i = s.create_item(kind="task", status="filed", title="Call Bruno", filed_ref="2601")
    assert handle_done(s, str(i)) == f"Done #{i} · Call Bruno\nSecond #2601 stays open, close it there"
    k = s.create_item(kind="task", status="filed", title="x", filed_ref="7")
    assert handle_drop(s, str(k)).endswith("\nSecond #7 stays open, close it there")
    s.close()


def test_notion_filed_item_gets_no_second_note(tmp_path):
    from clawdia.render import second_note
    s = store(tmp_path)
    url = "https://www.notion.so/Northwind-Operations-Lead-1a2b3c"
    i = s.create_item(kind="job", status="filed", title="Northwind OL", filed_ref=url)
    assert handle_done(s, str(i)) == f"Done #{i} · Northwind OL"
    j = s.create_item(kind="job", status="filed", title="x", filed_ref=url)
    assert handle_drop(s, str(j)) == f"Dropped #{j} · x" and handle_drop(s, str(j)) == f"Already dropped #{j} · x"
    assert second_note({"filed_ref": "12a"}) == "" and second_note({"filed_ref": None}) == ""
    assert second_note({"filed_ref": "٣"}) == ""  # non-ASCII digits are not a Second id
    s.close()


def test_render_inbox_counts_and_context(tmp_path):
    s = store(tmp_path)
    s.create_item(kind="job", status="new", title="j"); s.create_item(kind="task", status="routing", title="t", route_error="unavailable")
    out = render_inbox(s, None, datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc))
    assert "waiting for session: 1 jobs, 0 events" in out and "routing: 1" in out and "context: none" in out
    s.close()


def test_render_inbox_oldest_routing_context_age_and_filed_today(tmp_path):
    s = store(tmp_path)
    old = s.create_item(kind="task", status="routing", title="t", created_at=datetime(2026, 10, 30, 10, 48, tzinfo=timezone.utc))
    s.create_item(kind="film", status="new", title="f"); s.create_item(kind="film", status="new", title="g")
    s.create_item(kind="task", status="filed", title="a", filed_at=datetime(2026, 10, 30, 8, 0, tzinfo=timezone.utc))
    s.create_item(kind="task", status="filed", title="b", filed_at=datetime(2026, 10, 29, 8, 0, tzinfo=timezone.utc))
    s.create_item(kind="film", status="done", title="d")
    out = render_inbox(s, 7200, NOW)
    assert "routing: 1 (oldest 72 min)" in out and "context: 2h ago" in out and "filed today: 1" in out
    assert "film: 2" in out and "task: 3" in out and old
    s.close()


def _item(**kw):
    base = dict(id=12, kind="task", status="filed", title="Call Bruno", project="Acme", created_at="2026-10-30T12:05:00.000000+00:00",
                forward_from=None, fields=None, transcript=None, ocr_text=None, urls=None, filed_ref=None, route_error=None,
                media_paths=None, text=None, caption=None)
    base.update(kw)
    return base


def test_render_show_full_block():
    out = render_show(_item(forward_from={"name": "Ana", "chat_title": "Founders Circle", "link": "https://t.me/c/1"},
                            fields={"due": "2026-11-01", "place": "Lisbon"}, transcript="t" * 600, ocr_text="hello",
                            urls=["https://a.io", "https://b.io"], filed_ref="2601", route_error="unavailable",
                            media_paths=["/m/1.jpg", "/m/2.jpg"]), tz="Europe/Lisbon")
    lines = out.splitlines()
    assert lines[0] == "#12 · task · filed" and "Call Bruno" in lines[1]
    for expected in ("Project: Acme", "Created: 2026-10-30 12:05", "From: Ana · Founders Circle · https://t.me/c/1", "due: 2026-11-01", "place: Lisbon",
                     "OCR: hello", "https://a.io", "https://b.io", "Second: #2601", "Route error: unavailable", "Media: 2"):
        assert expected in out, expected
    assert "t" * 500 in out and "t" * 501 not in out
    assert "\u2014" not in out


def test_render_show_local_time_and_minimal_item():
    # October 30 is already winter time in Lisbon (UTC+0); July is summer time (UTC+1)
    out = render_show(_item(created_at="2026-07-01T12:05:00+00:00", project=None), tz="Europe/Lisbon")
    assert "Created: 2026-07-01 13:05" in out and "Project" not in out and "From" not in out and "Media" not in out


def test_render_show_caps_length():
    out = render_show(_item(fields={f"k{i}": "v" * 200 for i in range(40)}))
    assert len(out) <= 4000 and out.endswith("...")


# -- handlers -----------------------------------------------------------


def cmd(text):
    return SimpleNamespace(text=text, answer=AsyncMock(), from_user=SimpleNamespace(id=1))


def reply(m):
    return m.answer.await_args.args[0]


def make(tmp_path, **kw):
    return deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"), now=NOW, **kw)


async def test_start_and_help_greet_without_commands(tmp_path):
    d = make(tmp_path)
    for fn in (on_start, on_help):
        m = cmd("/help"); await fn(m, d)
        assert reply(m).startswith("\u041f\u0440\u0438\u0432\u0435\u0442, \u044f \u041a\u043b\u0430\u0432\u0434\u0438\u044f")
        assert "/" not in reply(m).split("\u041d\u0430\u043f\u0440\u0438\u043c\u0435\u0440")[0] and not any(
            c in reply(m) for c in ("/list", "/done", "/drop", "/show", "/inbox", "/sync", "/flush"))
        assert chr(0x2014) not in reply(m) and "Undo" in reply(m)
    assert d.store.meta_get("last_update_at") is not None


async def test_greeting_follows_the_language_of_the_latest_message(tmp_path):
    d = make(tmp_path)
    chat_id = cmd("/help").from_user.id
    d.store.add_turn(chat_id, "user", "\u043f\u0440\u0438\u0432\u0435\u0442")
    d.store.add_turn(chat_id, "user", "show my films please")
    d.store.add_turn(chat_id, "user", "/start")
    m = cmd("/start"); await on_start(m, d)
    assert reply(m).startswith("Hi, I'm Clawdia")


async def test_list_command_shelf_aliases_all_and_page(tmp_path):
    d = make(tmp_path)
    for i in range(25):
        d.store.create_item(kind="film", status="new", title=f"Film {i}")
    done = d.store.create_item(kind="film", status="done", title="Seen it")
    for arg, kind_alias in (("film", 1), ("films", 1), ("фильмы", 1)):
        m = cmd(f"/list {arg}"); await on_list(m, d)
        assert reply(m).startswith("film · 25 open") and "Seen it" not in reply(m)
    m = cmd("/list film 2"); await on_list(m, d)
    assert "page 2/2" in reply(m)
    m = cmd("/list films all"); await on_list(m, d)
    assert f"#{done} · Seen it" in reply(m) and reply(m).startswith("film · 26")
    m = cmd("/list film all 2"); await on_list(m, d)
    assert "page 2/2" in reply(m)
    m = cmd("/list купить"); await on_list(m, d)
    assert reply(m) == "buy · 0 open"


async def test_list_command_unknown_shelf_and_no_args(tmp_path):
    d = make(tmp_path)
    for text in ("/list", "/list nonsense", "/list@clawdia_bot  nonsense"):
        m = cmd(text); await on_list(m, d)
        assert reply(m) == "Unknown shelf. Use one of: task job event recipe film place buy idea unsorted"


async def test_done_drop_commands_reply_and_hold_the_drain_lock(tmp_path):
    d = make(tmp_path)
    i = d.store.create_item(kind="buy", status="new", title="Oil")
    j = d.store.create_item(kind="buy", status="new", title="Salt")
    m = cmd(f"/done {i}"); await on_done(m, d)
    assert reply(m) == f"Done #{i} · Oil"
    m = cmd(f"/drop {j}"); await on_drop(m, d)
    assert reply(m) == f"Dropped #{j} · Salt"
    m = cmd("/done"); await on_done(m, d); assert reply(m) == "Usage: /done <id>"
    m = cmd("/drop 999"); await on_drop(m, d); assert reply(m) == "No item #999"
    k = d.store.create_item(kind="buy", status="new", title="Tea")
    await d.drain_lock.acquire()
    t = asyncio.create_task(on_done(cmd(f"/done {k}"), d))
    await asyncio.sleep(0.02)
    assert not t.done() and d.store.get_item(k)["status"] == "new"
    d.drain_lock.release(); await t
    assert d.store.get_item(k)["status"] == "done"


async def test_show_command(tmp_path):
    d = make(tmp_path)
    i = d.store.create_item(kind="idea", status="new", title="Big idea", text="body")
    m = cmd(f"/show {i}"); await on_show(m, d)
    assert reply(m).startswith(f"#{i} · idea · new") and "Big idea" in reply(m)
    m = cmd("/show 999"); await on_show(m, d); assert reply(m) == "No item #999"
    m = cmd("/show x"); await on_show(m, d); assert reply(m) == "Usage: /show <id>"
    assert m.answer.await_args.kwargs.get("parse_mode") is None


async def test_inbox_command_uses_context_age(tmp_path):
    d = make(tmp_path)
    d.context_cache.save({"generated_at": "2026-10-30T10:00:00Z", "projects": []}, "e1")
    m = cmd("/inbox"); await on_inbox(m, d)
    assert "context: 2h ago" in reply(m)


async def test_flush_and_sync_both_drain_on_a_quiet_day(tmp_path):
    d = deps(tmp_path, TagResult("task", "x", None, {}, 0.9, None, "m"), now=datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc))
    i = d.store.create_item(kind="task", status="routing", title="x", text="x")
    m = cmd("/flush"); await on_flush(m, d)
    assert d.store.get_item(i)["status"] == "filed" and reply(m).startswith("Отправлено 1")
    j = d.store.create_item(kind="task", status="routing", title="y", text="y")
    m = cmd("/sync"); await on_sync(m, d)  # spec 6.9: no quiet-day hold on a Saturday
    assert d.store.get_item(j)["status"] == "filed" and reply(m).startswith("Отправлено 1")


async def test_flush_runs_under_drain_lock(tmp_path):
    d = make(tmp_path)
    i = d.store.create_item(kind="task", status="routing", title="x", text="x")
    await d.drain_lock.acquire()
    t = asyncio.create_task(on_flush(cmd("/flush"), d))
    await asyncio.sleep(0.02)
    assert not t.done() and d.second.post_task.await_count == 0
    d.drain_lock.release(); await t
    assert d.store.get_item(i)["status"] == "filed"


async def test_sync_refreshes_context_then_drains(tmp_path):
    d = make(tmp_path)
    d.store.create_item(kind="task", status="routing", title="x", text="x")
    calls = []

    async def refresh():
        calls.append("refresh")
        d.context_cache.save({"generated_at": "2026-10-30T11:59:50Z"}, "e")

    d.refresh_context = refresh
    m = cmd("/sync"); await on_sync(m, d)
    assert calls == ["refresh"] and reply(m) == "Отправлено 1, контекст обновлён 10 с назад"


async def test_sync_reports_refresh_failure_and_still_drains(tmp_path):
    d = make(tmp_path)
    d.store.create_item(kind="task", status="routing", title="x", text="x")
    d.refresh_context = AsyncMock(side_effect=RuntimeError("boom"))
    m = cmd("/sync"); await on_sync(m, d)
    assert reply(m) == "Отправлено 1, не удалось обновить контекст"
    d.refresh_context = None
    m = cmd("/sync"); await on_sync(m, d)
    assert reply(m).startswith("Отправлено 0")


async def test_sync_reports_drain_failure(tmp_path):
    d = make(tmp_path)
    d.second.post_task = AsyncMock(side_effect=RuntimeError("bug"))
    d.store.create_item(kind="task", status="routing", title="x", text="x")
    m = cmd("/flush"); await on_flush(m, d)
    assert "не удалась" in reply(m) and not d.drain_lock.locked()


def test_router_registers_commands_before_catch_all(tmp_path):
    d = make(tmp_path)
    r = build_router(d)
    names = [h.callback.__name__ for h in r.message.handlers]
    assert names[-2:] == ["_on_text", "_on_unsupported"] and len(names) >= 14


# -- fix round 1 --------------------------------------------------------

BIG = "9" * 20


async def test_huge_ids_are_not_found_instead_of_crashing(tmp_path):
    d = make(tmp_path)
    for fn, name in ((on_done, "done"), (on_drop, "drop"), (on_show, "show")):
        m = cmd(f"/{name} {BIG}"); await fn(m, d)
        assert reply(m) == f"No item #{BIG}"
    assert handle_done(d.store, BIG) == f"No item #{BIG}" and handle_drop(d.store, str(2**63)) == f"No item #{2**63}"
    i = d.store.create_item(kind="buy", status="new", title="Oil")
    m = cmd(f"/done {2**63 - 1}"); await on_done(m, d)
    assert reply(m) == f"No item #{2**63 - 1}" and d.store.get_item(i)["status"] == "new"


def ent(text):
    n = len(text.split()[0]) if text.startswith("/") else 0
    return [SimpleNamespace(type="bot_command", offset=0, length=n)] if n else []


async def feed(d, text, entities, uid=1, chat=None, **extra):
    from aiogram import Bot, Dispatcher
    from aiogram.types import Chat, Message, Update, User
    bot = Bot("1:AAA")
    sent = []

    async def fake_request(b, method, **kw):
        if type(method).__name__ == "GetMe":  # /cmd@botname makes aiogram look up the bot username
            return User(id=99, is_bot=True, first_name="Clawdia", username="clawdia_bot")
        sent.append(method.text)

    bot.session.make_request = fake_request
    dp = Dispatcher(); dp.include_router(build_router(d))
    msg = Message(message_id=7, date=NOW, chat=chat or Chat(id=uid, type="private"), from_user=User(id=uid, is_bot=False, first_name="x"),
                  text=text, entities=[{"type": e.type, "offset": e.offset, "length": e.length} for e in entities] or None,
                  **extra)
    try:
        await dp.feed_update(bot, Update(update_id=1, message=msg))
    finally:
        await bot.session.close()
    return sent


async def test_unknown_command_is_not_captured(tmp_path):
    d = make(tmp_path)
    for text, shown in (("/lsit films", "/lsit"), ("/don 12", "/don"), ("/lsit@clawdia_bot films", "/lsit")):
        sent = await feed(d, text, ent(text))
        assert sent == [f"Unknown command {shown}. /help lists them"]
    assert d.store.list_items(limit=10) == [] and d.tag_fn.await_count == 0


async def test_known_commands_and_plain_text_are_unaffected(tmp_path):
    d = make(tmp_path)
    assert (await feed(d, "/list films", ent("/list films"))) == ["film · 0 open"]
    assert (await feed(d, "/help", ent("/help")))[0].startswith("Привет, я Клавдия")
    for text in ("r/foo is a subreddit", "buy milk /soon", "plain note"):
        sent = await feed(d, text, ent(text))
        assert sent and sent[0].startswith("Saved #")
    assert len(d.store.list_items(limit=10)) == 3 and d.tag_fn.await_count == 3


async def test_command_entity_not_at_offset_zero_is_a_capture(tmp_path):
    d = make(tmp_path)
    ents = [SimpleNamespace(type="bot_command", offset=5, length=5)]
    sent = await feed(d, "note /lsit", ents)
    assert sent[0].startswith("Saved #")


async def test_unknown_command_from_non_owner_is_dropped(tmp_path):
    d = make(tmp_path)
    assert await feed(d, "/lsit", ent("/lsit"), uid=2) == []


# -- /memory ------------------------------------------------------------


def test_render_list_of_memory_facts():
    from clawdia.memory import render_list as render_memory
    facts = [{"id": 1, "section": "about", "text": "Lives in Lisbon"},
             {"id": 4, "section": "about", "text": "She likes green tea"},
             {"id": 3, "section": "context", "text": "Moving in March"}]
    assert render_memory(facts, "ru") == "О тебе:\nM1 Lives in Lisbon\nM4 She likes green tea\n\nКонтекст:\nM3 Moving in March"
    assert render_memory(facts[2:], "en") == "Context:\nM3 Moving in March"
    assert render_memory(facts[:1], "en") == "About you:\nM1 Lives in Lisbon"
    assert render_memory([], "ru") == "Память пуста. Скажи «запомни, что ...», и я запишу"
    assert render_memory([], "en").startswith("Memory is empty.")
    assert "#" not in render_memory(facts, "ru") and chr(0x2014) not in render_memory([], "en")


async def test_memory_command_empty_and_listed(tmp_path):
    from clawdia.bot import on_memory
    d = make(tmp_path)
    m = cmd("/memory"); await on_memory(m, d)
    assert reply(m) == "Память пуста. Скажи «запомни, что ...», и я запишу"
    d.store.memory_add("about", "Lives in Lisbon", source="chat", turn_id=None, now=NOW)
    gone = d.store.memory_add("context", "Gone", source="chat", turn_id=None, now=NOW)
    d.store.memory_delete(gone, NOW)
    d.store.memory_add("context", "Moving in March", source="chat", turn_id=None, now=NOW)
    m = cmd("/memory"); await on_memory(m, d)
    assert reply(m) == "О тебе:\nM1 Lives in Lisbon\n\nКонтекст:\nM3 Moving in March"
    d.settings.brain = "agent"
    d.ai_client = None  # no model call: the command never touches the client
    d.store.add_turn(1, "user", "what do you remember about me")
    m = cmd("/memory"); await on_memory(m, d)
    assert reply(m).startswith("About you:")


async def test_memory_command_splits_a_long_list_under_the_telegram_limit(tmp_path):
    from clawdia.bot import on_memory
    d = make(tmp_path)
    for i in range(10):
        d.store.memory_add("about", f"{i} " + "a" * 197, source="chat", turn_id=None, now=NOW)
    for i in range(15):
        d.store.memory_add("context", f"{i} " + "b" * 197, source="chat", turn_id=None, now=NOW)
    m = cmd("/memory"); await on_memory(m, d)
    parts = [c.args[0] for c in m.answer.await_args_list]
    assert len(parts) >= 2 and all(len(p) <= 4000 for p in parts)
    assert sum(p.count("\nM") + p.startswith("M") for p in parts) >= 25


async def test_memory_command_is_routed_and_in_help_text(tmp_path):
    d = make(tmp_path)
    d.store.memory_add("about", "Lives in Lisbon", source="chat", turn_id=None, now=NOW)
    sent = await feed(d, "/memory", [SimpleNamespace(type="bot_command", offset=0, length=7)])
    assert sent == ["О тебе:\nM1 Lives in Lisbon"]
    from clawdia.render import HELP_TEXT
    assert "/memory" in HELP_TEXT
