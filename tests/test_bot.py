import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from clawdia.bot import OwnerOnly, build_router, on_undo
from clawdia.tagger import TagResult
from tests.test_capture import deps, msg


@pytest.mark.asyncio
async def test_owner_filter_drops_other_users():
    f = OwnerOnly(frozenset({1}))
    private = SimpleNamespace(id=1, type="private")
    assert await f(SimpleNamespace(from_user=SimpleNamespace(id=1), chat=private)) is True
    assert await f(SimpleNamespace(from_user=SimpleNamespace(id=2), chat=SimpleNamespace(id=2, type="private"))) is False
    assert await f(SimpleNamespace(from_user=None, chat=private)) is False


@pytest.mark.asyncio
async def test_owner_filter_requires_the_owners_private_chat():
    f = OwnerOnly(frozenset({1}))
    owner = SimpleNamespace(id=1)
    assert await f(SimpleNamespace(from_user=owner, chat=SimpleNamespace(id=-100, type="supergroup"))) is False
    assert await f(SimpleNamespace(from_user=owner, chat=SimpleNamespace(id=-5, type="group"))) is False
    assert await f(SimpleNamespace(from_user=owner, chat=SimpleNamespace(id=7, type="private"))) is False
    assert await f(SimpleNamespace(from_user=owner, chat=None)) is False


@pytest.mark.asyncio
async def test_owner_in_a_group_gets_nothing_and_nothing_is_saved(tmp_path):
    from aiogram.types import Chat
    from tests.test_commands import NOW, feed
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"), now=NOW)
    for chat in (Chat(id=-100123, type="supergroup", title="G"), Chat(id=-55, type="group", title="g")):
        assert await feed(d, "hello from a group", [], chat=chat) == []
        assert await feed(d, "/inbox", [_cmd("/inbox")], chat=chat) == []
    assert d.store.list_items(limit=10) == []
    assert (await feed(d, "hello", []))[0].startswith("Saved #")  # the private chat still works


def _cmd(text):
    return SimpleNamespace(type="bot_command", offset=0, length=len(text))


@pytest.mark.asyncio
async def test_owner_callback_from_a_group_message_is_dropped(tmp_path):
    from aiogram import Bot, Dispatcher
    from aiogram.types import CallbackQuery, Chat, Message, Update, User
    from tests.test_commands import NOW
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"), now=NOW)
    i = d.store.create_item(kind="idea", status="new", title="x")
    bot = Bot("1:AAA"); bot.session.make_request = AsyncMock(return_value=True)
    dp = Dispatcher(); dp.include_router(build_router(d))
    try:
        for n, chat in enumerate((Chat(id=-100123, type="supergroup", title="G"), Chat(id=1, type="private"))):
            m = Message(message_id=1, date=NOW, chat=chat, from_user=User(id=99, is_bot=True, first_name="b"), text="x")
            q = CallbackQuery(id=str(n), from_user=User(id=1, is_bot=False, first_name="x"), chat_instance="c",
                              data=f"undo:{i}", message=m)
            await dp.feed_update(bot, Update(update_id=n + 1, callback_query=q))
            assert (d.store.get_item(i)["status"] == "dismissed") is (chat.type == "private")
    finally:
        await bot.session.close()


def cb(data):
    return SimpleNamespace(data=data, message=SimpleNamespace(edit_text=AsyncMock()), answer=AsyncMock(), from_user=SimpleNamespace(id=1))


def test_build_router_registers_message_and_callback_handlers(tmp_path):
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    r = build_router(d)
    assert r.message.handlers and r.callback_query.handlers


@pytest.mark.asyncio
async def test_undo_waits_for_a_running_drain_instead_of_racing_it(tmp_path):
    import asyncio
    from clawdia.capture import process
    d = deps(tmp_path, TagResult("task", "new", None, {}, 0.9, None, "m"))
    a = d.store.create_item(kind="task", status="routing", title="a", text="a")
    b = d.store.create_item(kind="task", status="routing", title="b", text="b")
    gate, started = asyncio.Event(), asyncio.Event()

    async def post(item):
        if item["id"] == a:
            started.set()
            await gate.wait()
        return {"id": 9000 + item["id"], "project": None}

    d.second.post_task = AsyncMock(side_effect=post)
    t = asyncio.create_task(process(msg("new"), d))
    await started.wait()
    q = cb(f"undo:{b}")
    dt = asyncio.create_task(on_undo(q, d))
    await asyncio.sleep(0.02)
    assert not dt.done()  # blocked on the drain lock, status untouched
    assert d.store.get_item(b)["status"] == "routing"
    gate.set()
    await t
    await dt
    assert d.store.get_item(b)["status"] == "dismissed"
    # the drain filed b first, so the undo sees it in Second and says so
    assert q.message.edit_text.await_args.args[0] == f"Undone #{b}.\nSecond #{9000 + b} stays open, close it there"


@pytest.mark.asyncio
async def test_callbacks_update_last_update_at_and_survive_meta_failure(tmp_path):
    from clawdia.capture import process
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    c = await process(msg("x"), d)
    assert d.store.meta_get("last_update_at") is None
    await on_undo(cb("undo:999"), d)
    first = d.store.meta_get("last_update_at")
    assert first is not None
    d.store.meta_set = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("locked"))
    q = cb(f"undo:{c.item_id}")
    await on_undo(q, d)
    assert q.message.edit_text.await_args.args[0] == f"Undone #{c.item_id}."


@pytest.mark.asyncio
async def test_message_handler_replies_even_when_meta_write_fails(tmp_path):
    from clawdia.bot import on_message
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    d.store.meta_set = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("locked"))
    m = msg("hello"); m.answer = AsyncMock()
    await on_message(m, d)
    m.answer.assert_awaited()
    assert m.answer.await_args.args[0].startswith("Saved #")


@pytest.mark.asyncio
async def test_two_owner_accounts_each_in_their_own_private_chat():
    f = OwnerOnly(frozenset({1, 2}))

    def ev(uid, chat_id, chat_type="private"):
        return SimpleNamespace(from_user=SimpleNamespace(id=uid), chat=SimpleNamespace(id=chat_id, type=chat_type))

    assert await f(ev(1, 1)) is True  # primary
    assert await f(ev(2, 2)) is True  # secondary in its own private chat
    assert await f(ev(2, -100, "supergroup")) is False  # secondary in a group
    assert await f(ev(2, 1)) is False  # secondary writing in the primary's chat
    assert await f(ev(1, 2)) is False  # primary id in the secondary's chat
    assert await f(ev(3, 3)) is False  # unknown id
    assert await f(ev(3, 1)) is False


@pytest.mark.asyncio
async def test_owner_callback_secondary_in_own_chat_only():
    f = OwnerOnly(frozenset({1, 2}))

    def cb(uid, chat_id, chat_type="private"):
        from aiogram.types import CallbackQuery
        m = SimpleNamespace(chat=SimpleNamespace(id=chat_id, type=chat_type))
        return CallbackQuery.model_construct(id="1", from_user=SimpleNamespace(id=uid), chat_instance="c", data="x", message=m)

    assert await f(cb(2, 2)) is True
    assert await f(cb(2, -5, "group")) is False
    assert await f(cb(2, 1)) is False
    assert await f(cb(3, 3)) is False


@pytest.mark.asyncio
async def test_secondary_capture_notifies_and_reminds_the_primary_only(tmp_path):
    from datetime import datetime, timezone
    from clawdia.bot import make_notify
    from clawdia.capture import process
    from clawdia.config import Settings
    from clawdia.reminders import send_due
    now = datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc)
    d = deps(tmp_path, TagResult("task", "new one", None, {}, 0.9, None, "m"), now=now)
    d.settings = Settings(bot_token="t", owner_id=1, extra_owner_ids="2", second_api_token="t", anthropic_api_key="k",
                          openai_api_key="k", state_dir=tmp_path, brain="classic")
    assert d.settings.allowed_owner_ids == frozenset({1, 2})
    bot = SimpleNamespace(send_message=AsyncMock())
    d.notify = make_notify(bot, d.store, d.settings.owner_id)
    old = d.store.create_item(kind="task", status="routing", title="old", text="old")
    m = msg("new one")
    m.from_user = SimpleNamespace(id=2); m.chat = SimpleNamespace(id=2)  # captured from the secondary account
    await process(m, d)
    assert d.store.get_item(old)["status"] == "filed"
    assert [c.args[0] for c in bot.send_message.await_args_list] == [1]  # drain notification went to the primary
    assert "old" in bot.send_message.await_args.args[1]

    bot2 = SimpleNamespace(send_message=AsyncMock())
    i = d.store.create_item(kind="place", status="new", title="Market")
    d.store.add_reminder(i, now)
    assert await send_due(d.store, bot2, d.settings.owner_id, now) == 1
    assert [c.args[0] for c in bot2.send_message.await_args_list] == [1]


# -- phase 2 wiring (P6) ------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

P6_NOW = datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc)  # Thursday 10:00 Lisbon


def p6(tmp_path, tag=None, **kw):
    return deps(tmp_path, tag or TagResult("idea", "Test idea", None, {}, 0.8, None, "m"), now=P6_NOW, **kw)


async def feed_full(d, text=None, entities=(), uid=1, **extra):
    """Feed one private message through the real router; returns (text, keyboard callback rows) per send."""
    from aiogram import Bot, Dispatcher
    from aiogram.types import Chat, Message, Update, User
    bot = Bot("1:AAA")
    sent = []

    async def fake_request(b, method, **kw):
        if type(method).__name__ == "GetMe":
            return User(id=99, is_bot=True, first_name="Clawdia", username="clawdia_bot")
        markup = getattr(method, "reply_markup", None)
        rows = [[btn.callback_data for btn in row] for row in markup.inline_keyboard] if markup else None
        sent.append((getattr(method, "text", None), rows))

    bot.session.make_request = fake_request
    dp = Dispatcher(); dp.include_router(build_router(d))
    m = Message(message_id=7, date=P6_NOW, chat=Chat(id=uid, type="private"), from_user=User(id=uid, is_bot=False, first_name="x"),
                text=text, entities=[{"type": e.type, "offset": e.offset, "length": e.length} for e in entities] or None, **extra)
    try:
        await dp.feed_update(bot, Update(update_id=1, message=m))
    finally:
        await bot.session.close()
    return sent


def _hidden_forward():
    from aiogram.types import MessageOriginHiddenUser
    return {"forward_origin": MessageOriginHiddenUser(type="hidden_user", date=P6_NOW, sender_user_name="Ana")}


async def test_plain_text_goes_through_the_conversation_engine_with_one_undo_button(tmp_path):
    d = p6(tmp_path)
    [(text, rows)] = await feed_full(d, "test idea")
    [item] = d.store.list_items(limit=10)
    assert text.startswith(f"Saved #{item['id']} · idea") and rows == [[f"undo:{item['id']}"]]
    assert d.understand_fn.await_count == 1 and item["chat_id"] == 1
    user, bot = d.store.recent_turns(1, 20)
    assert (user["role"], user["processed"], user["item_id"]) == ("user", 1, item["id"]) and bot["role"] == "bot"


async def test_forward_saying_delete_is_captured_and_no_intent_call_is_made(tmp_path):
    # Review Focus 3: a forwarded "удали #3" is a capture; item 3 is untouched.
    d = p6(tmp_path)
    targets = [d.store.create_item(kind="idea", status="new", title=f"n{i}", text=f"n{i}", chat_id=1) for i in range(3)]
    assert targets[-1] == 3
    [(text, rows)] = await feed_full(d, "удали #3", **_hidden_forward())
    new = [i for i in d.store.list_items(limit=10) if i["id"] not in targets]
    assert len(new) == 1 and new[0]["text"] == "удали #3" and new[0]["forward_from"]["name"] == "Ana"
    assert d.store.get_item(3)["status"] == "new"
    assert d.understand_fn.await_count == 0 and d.tag_fn.await_count == 1 and d.reply_fn.await_count == 1
    assert rows == [[f"undo:{new[0]['id']}"]] and f"#{new[0]['id']}" in text and new[0]["chat_id"] == 1


async def test_forwarded_command_text_is_captured_without_an_intent_call(tmp_path):
    d = p6(tmp_path)
    sent = await feed_full(d, "/list films", [SimpleNamespace(type="bot_command", offset=0, length=5)], **_hidden_forward())
    assert sent[0][0].startswith("Saved #") and len(d.store.list_items(limit=10)) == 1
    assert d.understand_fn.await_count == 0


async def test_slash_list_makes_no_model_call(tmp_path):
    d = p6(tmp_path)
    d.store.create_item(kind="film", status="new", title="Dune")
    sent = await feed_full(d, "/list film", [SimpleNamespace(type="bot_command", offset=0, length=5)])
    assert sent[0][0].startswith("film · 1 open") and sent[0][1] is None
    assert d.understand_fn.await_count == 0 and d.reply_fn.await_count == 0 and d.tag_fn.await_count == 0
    assert d.store.recent_turns(1, 20) == []


async def test_unsupported_messages_get_a_short_reply_and_save_nothing(tmp_path):
    from aiogram.types import Contact, Location
    d = p6(tmp_path)
    loc = {"location": Location(latitude=38.7, longitude=-9.1)}
    assert await feed_full(d, None, **loc) == [("Пока не умею сохранять такое.", None)]  # no turns yet: Russian
    d.store.add_turn(1, "user", "buy oat milk", at=P6_NOW)
    sent = await feed_full(d, None, contact=Contact(phone_number="1", first_name="A"))
    assert sent == [("I can't save this type yet.", None)]
    d.store.add_turn(1, "user", "[photo]", at=P6_NOW)  # a marker turn does not decide the language
    assert (await feed_full(d, None, **loc))[0][0] == "I can't save this type yet."
    assert d.store.list_items(limit=10) == []
    assert d.understand_fn.await_count == 0 and d.reply_fn.await_count == 0 and d.tag_fn.await_count == 0
    assert len(d.store.recent_turns(1, 20)) == 2


def _photo():
    from aiogram.types import PhotoSize
    return [PhotoSize(file_id="p", file_unique_id="u", width=1, height=1)]


async def test_media_with_a_command_caption_is_captured_not_run(tmp_path):
    d = p6(tmp_path)
    targets = [d.store.create_item(kind="idea", status="new", title=f"n{i}", text=f"n{i}", chat_id=1) for i in range(2)]
    assert targets[-1] == 2
    for caption in ("/drop 2", "/list film", "/lsit"):
        n = len(d.store.list_items(limit=100))
        sent = await feed_full(d, None, photo=_photo(), caption=caption,
                               caption_entities=[{"type": "bot_command", "offset": 0, "length": len(caption.split()[0])}])
        assert sent[0][0].startswith("Saved #"), caption
        assert len(d.store.list_items(limit=100)) == n + 1
    assert d.store.get_item(2)["status"] == "new" and d.understand_fn.await_count == 0


async def test_media_capture_reply_uses_the_reply_call_and_writes_turns(tmp_path):
    from clawdia.bot import on_message
    from tests.test_capture import msg
    d = p6(tmp_path, save_media=AsyncMock(return_value=(tmp_path / "a.jpg", "image/jpeg")))
    m = msg(photo=[SimpleNamespace(file_id="a")], caption="рецепт пасты"); m.answer = AsyncMock()
    await on_message(m, d)
    [item] = d.store.list_items(limit=10)
    assert item["chat_id"] == 1
    assert d.tag_fn.await_count == 1 and d.reply_fn.await_count == 1 and d.understand_fn.await_count == 0
    facts = d.reply_fn.await_args.args[0]
    assert facts.action == "saved" and facts.item_id == item["id"] and facts.language == "ru"
    text, kw = m.answer.await_args.args[0], m.answer.await_args.kwargs
    assert text.startswith(f"Сохранила #{item['id']}")  # Russian caption: Russian deterministic reply
    assert [[b.callback_data for b in row] for row in kw["reply_markup"].inline_keyboard] == [[f"undo:{item['id']}"]]
    user, bot = d.store.recent_turns(1, 20)
    assert (user["role"], user["processed"], user["item_id"], user["text"]) == ("user", 1, item["id"], "рецепт пасты")
    assert (bot["role"], bot["item_id"], bot["text"]) == ("bot", item["id"], text)


async def test_media_without_caption_gets_a_marker_turn(tmp_path):
    from clawdia.bot import on_message
    from tests.test_capture import msg
    d = p6(tmp_path)
    m = msg(voice=SimpleNamespace(duration=3, file_id="f")); m.answer = AsyncMock()
    await on_message(m, d)
    user = d.store.recent_turns(1, 20)[0]
    assert user["text"] == "[voice]" and user["processed"] == 1


async def test_album_follower_gets_no_reply(tmp_path):
    from clawdia.bot import on_message
    from tests.test_capture import msg
    d = p6(tmp_path, save_media=AsyncMock(side_effect=[(tmp_path / "a.jpg", "image/jpeg"), (tmp_path / "b.jpg", "image/jpeg")]))
    m1 = msg(photo=[SimpleNamespace(file_id="a")], mgid="g", mid=1); m1.answer = AsyncMock()
    m2 = msg(photo=[SimpleNamespace(file_id="b")], mgid="g", mid=2); m2.answer = AsyncMock()
    await on_message(m1, d); await on_message(m2, d)
    assert m1.answer.await_count == 1 and m2.answer.await_count == 0
    assert len(d.store.list_items(limit=10)) == 1 and len(d.store.recent_turns(1, 20)) == 2


async def test_media_reply_falls_back_when_the_reply_writer_breaks(tmp_path):
    from clawdia.bot import on_message
    from tests.test_capture import msg
    d = p6(tmp_path)
    d.reply_fn = AsyncMock(side_effect=RuntimeError("boom"))
    m = msg("forwarded words", forward=SimpleNamespace(type="hidden_user", sender_user_name="Ana", date=None)); m.answer = AsyncMock()
    await on_message(m, d)
    [item] = d.store.list_items(limit=10)
    assert m.answer.await_args.args[0].startswith(f"Saved #{item['id']} · idea · Test idea")


async def test_media_reply_asks_when_for_a_timeless_reminder_request(tmp_path):
    from clawdia.bot import on_message
    from tests.test_capture import msg
    d = p6(tmp_path)
    m = msg(photo=[SimpleNamespace(file_id="a")], caption="напомни про это"); m.answer = AsyncMock()
    await on_message(m, d)
    [item] = d.store.list_items(limit=10)
    assert item["fields"].get("ask_when") is True and item["fields"].get("asked") is True
    facts = d.reply_fn.await_args.args[0]
    assert facts.question == "Когда напомнить?"
    assert "When should I remind you?" not in facts.fallback and facts.fallback.endswith("\nКогда напомнить?")
    assert m.answer.await_args.args[0].count("напомнить") == 1
    rows = [[b.callback_data for b in row] for row in m.answer.await_args.kwargs["reply_markup"].inline_keyboard]
    assert rows == [[f"undo:{item['id']}"]]


# -- Undo ---------------------------------------------------------------------


async def test_undo_dismisses_disables_reminders_and_makes_no_model_call(tmp_path):
    from clawdia.bot import on_undo
    d = p6(tmp_path)
    i = d.store.create_item(kind="event", status="new", title="Party", text="party friday", chat_id=1)
    d.store.add_reminder(i, P6_NOW + timedelta(days=1))
    q = cb(f"undo:{i}")
    await on_undo(q, d)
    assert d.store.get_item(i)["status"] == "dismissed" and d.store.due_reminders(P6_NOW + timedelta(days=60)) == []
    assert q.message.edit_text.await_args.args[0] == f"Undone #{i}."
    assert q.message.edit_text.await_args.kwargs.get("reply_markup") is None
    events = [r["type"] for r in d.store._conn.execute("SELECT type FROM events WHERE item_id=?", (i,)).fetchall()]
    assert "undone" in events
    assert d.understand_fn.await_count == 0 and d.reply_fn.await_count == 0 and d.tag_fn.await_count == 0
    q.answer.assert_awaited()


async def test_undo_in_russian_names_the_open_second_task(tmp_path):
    # Acceptance 8: the item reached Second, so the reply says the Second task stays open.
    from clawdia.bot import on_undo
    d = p6(tmp_path)
    i = d.store.create_item(kind="task", status="filed", title="Счёт", text="выставить счёт", filed_ref="2601", chat_id=1)
    q = cb(f"undo:{i}")
    await on_undo(q, d)
    assert d.store.get_item(i)["status"] == "dismissed"
    assert q.message.edit_text.await_args.args[0] == f"Отменила #{i}.\nSecond #2601 stays open, close it there"


async def test_undo_of_a_queued_task_takes_it_out_of_the_outbox(tmp_path):
    from clawdia.bot import on_undo
    d = p6(tmp_path)
    i = d.store.create_item(kind="task", status="routing", title="x", text="x")
    await on_undo(cb(f"undo:{i}"), d)
    assert d.store.outbox() == []


async def test_undo_twice_says_already_undone(tmp_path):
    from clawdia.bot import on_undo
    d = p6(tmp_path)
    i = d.store.create_item(kind="idea", status="new", title="x", text="x")
    await on_undo(cb(f"undo:{i}"), d)
    q = cb(f"undo:{i}")
    await on_undo(q, d)
    q.answer.assert_awaited_with(f"Already undone #{i}")
    q.message.edit_text.assert_not_awaited()
    events = [r["type"] for r in d.store._conn.execute("SELECT type FROM events WHERE item_id=?", (i,)).fetchall()]
    assert events.count("undone") == 1


async def test_undo_rejects_bad_payloads(tmp_path):
    from clawdia.bot import on_undo
    d = p6(tmp_path)
    i = d.store.create_item(kind="idea", status="new", title="x", text="x")
    for data in ("undo:abc", "undo:999", "undo", f"undo:{i}:x", "undo:²", f"undo:{'9' * 30}", "undo:-1"):
        q = cb(data)
        await on_undo(q, d)
        q.message.edit_text.assert_not_awaited(); q.answer.assert_awaited()
    assert d.store.get_item(i)["status"] == "new"


async def test_undo_waits_for_a_running_drain(tmp_path):
    import asyncio
    from clawdia.bot import on_undo
    d = p6(tmp_path)
    i = d.store.create_item(kind="idea", status="new", title="x", text="x")
    await d.drain_lock.acquire()
    t = asyncio.create_task(on_undo(cb(f"undo:{i}"), d))
    await asyncio.sleep(0.02)
    assert not t.done() and d.store.get_item(i)["status"] == "new"
    d.drain_lock.release(); await t
    assert d.store.get_item(i)["status"] == "dismissed"


async def test_undo_through_the_router_is_owner_only(tmp_path):
    from aiogram import Bot, Dispatcher
    from aiogram.types import CallbackQuery, Chat, Message, Update, User
    d = p6(tmp_path)
    i = d.store.create_item(kind="idea", status="new", title="x", text="x")
    bot = Bot("1:AAA"); bot.session.make_request = AsyncMock(return_value=True)
    dp = Dispatcher(); dp.include_router(build_router(d))
    try:
        for n, uid in enumerate((2, 1)):
            m = Message(message_id=1, date=P6_NOW, chat=Chat(id=uid, type="private"), from_user=User(id=99, is_bot=True, first_name="b"), text="x")
            q = CallbackQuery(id=str(n), from_user=User(id=uid, is_bot=False, first_name="x"), chat_instance="c", data=f"undo:{i}", message=m)
            await dp.feed_update(bot, Update(update_id=n + 1, callback_query=q))
            assert (d.store.get_item(i)["status"] == "dismissed") is (uid == 1)
    finally:
        await bot.session.close()


# -- buttons from phase 1 --------------------------------------------------------


@pytest.mark.parametrize("data", ["retag:{i}:task", "delete:{i}", "remind:{i}:tomorrow", "roff:{i}"])
async def test_old_buttons_answer_older_version_and_change_nothing(tmp_path, data):
    from clawdia.bot import on_old_button
    d = p6(tmp_path)
    i = d.store.create_item(kind="idea", status="new", title="x", text="x", fields={"ask_when": True})
    q = cb(data.format(i=i)); q.message.edit_reply_markup = AsyncMock()
    await on_old_button(q, d)
    q.answer.assert_awaited_with("This button is from an older version")
    q.message.edit_reply_markup.assert_awaited_with(reply_markup=None)
    item = d.store.get_item(i)
    assert item["status"] == "new" and item["kind"] == "idea" and item["fields"] == {"ask_when": True}
    assert d.store.due_reminders(P6_NOW + timedelta(days=60)) == []


async def test_old_buttons_through_the_router(tmp_path):
    from aiogram import Bot, Dispatcher
    from aiogram.types import CallbackQuery, Chat, Message, Update, User
    d = p6(tmp_path)
    i = d.store.create_item(kind="idea", status="new", title="x", text="x")
    calls = []

    async def request(b, method, **kw):
        calls.append(type(method).__name__)
        return True

    bot = Bot("1:AAA"); bot.session.make_request = request
    dp = Dispatcher(); dp.include_router(build_router(d))
    try:
        m = Message(message_id=1, date=P6_NOW, chat=Chat(id=1, type="private"), from_user=User(id=99, is_bot=True, first_name="b"), text="x")
        q = CallbackQuery(id="1", from_user=User(id=1, is_bot=False, first_name="x"), chat_instance="c", data=f"delete:{i}", message=m)
        await dp.feed_update(bot, Update(update_id=1, callback_query=q))
    finally:
        await bot.session.close()
    assert d.store.get_item(i)["status"] == "new"
    assert "AnswerCallbackQuery" in calls and "EditMessageReplyMarkup" in calls


async def test_reminder_ping_keyboard_is_unchanged(tmp_path):
    from clawdia.reminders import send_due
    d = p6(tmp_path)
    i = d.store.create_item(kind="place", status="new", title="Market")
    rid = d.store.add_reminder(i, P6_NOW)
    bot = SimpleNamespace(send_message=AsyncMock())
    assert await send_due(d.store, bot, 1, P6_NOW) == 1
    rows = [[(b.text, b.callback_data) for b in row] for row in bot.send_message.await_args.kwargs["reply_markup"].inline_keyboard]
    assert rows == [[("Done", f"rdone:{i}"), ("Snooze 1d", f"rsnooze:{rid}:1d"), ("Snooze 1w", f"rsnooze:{rid}:7d")]]


async def test_help_explains_talking_and_cancel_has_nothing_to_cancel(tmp_path):
    d = p6(tmp_path)
    [(help_text, _)] = await feed_full(d, "/help", [SimpleNamespace(type="bot_command", offset=0, length=5)])
    for phrase in ("покажи фильмы", "это задача", "напомни завтра", "не напоминай", "удали это", "голосом", "Undo"):
        assert phrase in help_text, phrase
    assert chr(0x2014) not in help_text and "/list" not in help_text and "/cancel" not in help_text
    assert await feed_full(d, "/cancel", [SimpleNamespace(type="bot_command", offset=0, length=7)]) == [("Nothing to cancel", None)]


def test_router_order_forward_commands_unknown_media_text(tmp_path):
    d = p6(tmp_path)
    names = [h.callback.__name__ for h in build_router(d).message.handlers]
    assert names[:2] == ["_on_message", "_on_message"] and names[-2:] == ["_on_text", "_on_unsupported"]
    assert "_on_date_entry" not in names


# -- startup recovery (Review Focus 5) ---------------------------------------------


def _bot():
    return SimpleNamespace(send_message=AsyncMock())


async def test_recover_handles_an_unprocessed_turn_exactly_once(tmp_path):
    from clawdia.converse import recover_unprocessed
    d = p6(tmp_path); d.bot = _bot()
    turn = d.store.add_turn(1, "user", "купить молоко", processed=False, at=P6_NOW - timedelta(minutes=10))
    assert await recover_unprocessed(d) == 1
    [item] = d.store.list_items(limit=10)
    assert item["text"] == "купить молоко" and item["chat_id"] == 1
    turns = d.store.recent_turns(1, 20)
    assert [(t["role"], t["processed"]) for t in turns] == [("user", 1), ("bot", 1)] and turns[0]["id"] == turn
    assert turns[0]["item_id"] == item["id"]
    chat, kw = d.bot.send_message.await_args.args[0], d.bot.send_message.await_args.kwargs
    assert chat == 1 and kw["text"].startswith(f"Сохранила #{item['id']}")
    assert [[b.callback_data for b in row] for row in kw["reply_markup"].inline_keyboard] == [[f"undo:{item['id']}"]]
    # a second recovery is a no-op: no duplicate item, no second message
    assert await recover_unprocessed(d) == 0
    assert len(d.store.list_items(limit=10)) == 1 and d.bot.send_message.await_count == 1
    assert d.understand_fn.await_count == 1


async def test_recover_saves_turns_older_than_an_hour_as_notes_and_skips_processed_ones(tmp_path):
    from clawdia.converse import recover_unprocessed
    d = p6(tmp_path); d.bot = _bot()
    old = d.store.add_turn(1, "user", "old", processed=False, at=P6_NOW - timedelta(minutes=61))
    d.store.add_turn(1, "user", "done already", processed=True, at=P6_NOW - timedelta(minutes=5))
    assert await recover_unprocessed(d) == 0
    [item] = d.store.list_items(limit=10)  # the late turn is a quiet unsorted note, never answered
    assert (item["text"], item["kind"], item["chat_id"]) == ("old", "unsorted", 1)
    assert d.bot.send_message.await_count == 0 and d.understand_fn.await_count == 0
    assert [t for t in d.store.recent_turns(1, 20) if t["id"] == old][0]["processed"] == 1


async def test_recover_continues_after_one_turn_fails(tmp_path):
    from clawdia.converse import recover_unprocessed
    from tests.test_capture import capture_only
    d = p6(tmp_path); d.bot = _bot()
    d.store.add_turn(1, "user", "first", processed=False, at=P6_NOW - timedelta(minutes=3))
    d.store.add_turn(2, "user", "second", processed=False, at=P6_NOW - timedelta(minutes=2))
    async def flaky(text, **kw):
        if text == "first":
            raise RuntimeError("boom")
        return await capture_only(text, **kw)

    d.understand_fn = AsyncMock(side_effect=flaky)
    assert await recover_unprocessed(d) == 1
    [item] = d.store.list_items(limit=10)
    assert item["text"] == "second" and item["chat_id"] == 2
    assert d.bot.send_message.await_args.args[0] == 2


async def test_recover_continues_when_a_send_fails(tmp_path):
    from clawdia.converse import recover_unprocessed
    d = p6(tmp_path); d.bot = _bot()
    d.bot.send_message = AsyncMock(side_effect=[RuntimeError("blocked"), None])
    d.store.add_turn(1, "user", "first", processed=False, at=P6_NOW - timedelta(minutes=3))
    d.store.add_turn(1, "user", "second", processed=False, at=P6_NOW - timedelta(minutes=2))
    assert await recover_unprocessed(d) == 2
    assert len(d.store.list_items(limit=10)) == 2 and d.bot.send_message.await_count == 2


async def test_recover_does_not_duplicate_an_item_saved_before_the_crash(tmp_path):
    # The crash came after the item was saved but before the turn was marked processed.
    from clawdia.converse import recover_unprocessed
    d = p6(tmp_path); d.bot = _bot()
    turn = d.store.add_turn(1, "user", "купить молоко", processed=False, at=P6_NOW - timedelta(minutes=10))
    i = d.store.create_item(kind="buy", status="new", title="Milk", text="купить молоко", chat_id=1,
                            created_at=P6_NOW - timedelta(minutes=10))
    assert await recover_unprocessed(d) == 1
    assert [x["id"] for x in d.store.list_items(limit=10)] == [i]
    assert d.understand_fn.await_count == 0 and d.tag_fn.await_count == 0
    user = [t for t in d.store.recent_turns(1, 20) if t["id"] == turn][0]
    assert user["processed"] == 1 and user["item_id"] == i
    assert d.bot.send_message.await_args.kwargs["text"].startswith(f"Сохранила #{i} · покупка · Milk")


async def test_serve_recovers_after_the_context_refresh_and_before_polling(tmp_path, monkeypatch):
    import clawdia.bot as botmod
    from clawdia.config import Settings
    order = []

    class FakeBot:
        def __init__(self, token):
            self.session = SimpleNamespace(close=AsyncMock())

    class FakeDispatcher:
        def include_router(self, router):
            pass

        async def start_polling(self, bot):
            order.append("poll")

    async def refresh(deps):
        order.append("refresh")

    async def recover(deps):
        order.append("recover")
        return 0

    async def loops(deps):
        order.append("loops")

    monkeypatch.setattr(botmod, "Bot", FakeBot)
    monkeypatch.setattr(botmod, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(botmod, "refresh_context", refresh)
    monkeypatch.setattr(botmod, "recover_unprocessed", recover)
    monkeypatch.setattr(botmod, "run_loops", loops)
    monkeypatch.setattr(botmod, "_ai_client", lambda s: SimpleNamespace(close=AsyncMock()))
    settings = Settings(bot_token="1:AAA", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k", state_dir=tmp_path)
    await botmod.serve(settings)
    assert order.index("refresh") < order.index("recover") < order.index("poll")


async def test_save_that_after_a_photo_does_not_save_the_marker(tmp_path):
    from clawdia.bot import on_message
    from clawdia.converse import handle_text
    from clawdia.understand import Understanding
    from tests.test_capture import msg
    d = p6(tmp_path)
    m = msg(photo=[SimpleNamespace(file_id="a")]); m.answer = AsyncMock()
    await on_message(m, d)
    [item] = d.store.list_items(limit=10)
    d.understand_fn = AsyncMock(return_value=Understanding(intent="followup", language="ru", save_previous=True))
    out = await handle_text(msg("сохрани это"), d)
    assert out.text == f"Уже сохранила #{item['id']}." and len(d.store.list_items(limit=10)) == 1


async def test_captionless_media_replies_in_the_chat_language(tmp_path):
    from clawdia.bot import on_message
    from tests.test_capture import msg
    d = p6(tmp_path)
    m = msg(photo=[SimpleNamespace(file_id="a")]); m.answer = AsyncMock()
    await on_message(m, d)
    assert d.reply_fn.await_args.args[0].language == "ru"  # no typed turn yet: Russian
    d.store.add_turn(1, "user", "buy oat milk", at=P6_NOW)
    m2 = msg(voice=SimpleNamespace(duration=2, file_id="v"), mid=2); m2.answer = AsyncMock()
    await on_message(m2, d)  # the "[photo]" marker turn in between does not count
    assert d.reply_fn.await_args.args[0].language == "en"
    m3 = msg(photo=[SimpleNamespace(file_id="b")], caption="рецепт", mid=3); m3.answer = AsyncMock()
    await on_message(m3, d)  # its own words win
    assert d.reply_fn.await_args.args[0].language == "ru"


async def test_failed_text_turn_is_marked_processed_and_not_recovered(tmp_path):
    from clawdia.converse import recover_unprocessed
    d = p6(tmp_path); d.bot = _bot()
    d.understand_fn = AsyncMock(side_effect=RuntimeError("boom"))
    sent = await feed_full(d, "купить молоко")
    assert sent == [("Could not save, try again", None)]
    [turn] = d.store.recent_turns(1, 20)
    assert (turn["role"], turn["text"], turn["processed"]) == ("user", "купить молоко", 1)
    assert await recover_unprocessed(d) == 0 and d.bot.send_message.await_count == 0
    assert d.store.list_items(limit=10) == []


async def test_recovered_correction_is_judged_at_its_send_time(tmp_path):
    # Sent 14 minutes after the bot reply, recovered 2 minutes later (16 after it): still a correction.
    from clawdia.converse import recover_unprocessed
    from clawdia.understand import Understanding
    d = p6(tmp_path); d.bot = _bot()
    i = d.store.create_item(kind="idea", status="new", title="Invoice", text="invoice", chat_id=1)
    d.store.add_turn(1, "user", "invoice", item_id=i, at=P6_NOW - timedelta(minutes=16, seconds=5))
    d.store.add_turn(1, "bot", f"Saved #{i}", item_id=i, at=P6_NOW - timedelta(minutes=16))
    d.store.add_turn(1, "user", "это задача", processed=False, at=P6_NOW - timedelta(minutes=2))
    d.understand_fn = AsyncMock(return_value=Understanding(intent="correct", language="ru", new_kind="buy"))
    assert await recover_unprocessed(d) == 1
    assert d.store.get_item(i)["kind"] == "buy" and len(d.store.list_items(limit=10)) == 1
    assert d.understand_fn.await_args.kwargs["now"] == P6_NOW - timedelta(minutes=2)


# -- owner memory: Remember button, Undo of memory writes ------------------------------------------


def mem_cb(data, text="Хорошо.", rows=(), lang_text=None):
    """A callback on an agent reply: its text and inline keyboard are what the bot rebuilds from."""
    keyboard = SimpleNamespace(inline_keyboard=[
        [SimpleNamespace(text=label, callback_data=cd) for label, cd in row] for row in rows])
    message = SimpleNamespace(text=text, reply_markup=keyboard, edit_text=AsyncMock())
    return SimpleNamespace(data=data, message=message, answer=AsyncMock(), from_user=SimpleNamespace(id=1))


def mem_deps(tmp_path):
    from tests.test_commands import NOW
    return deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"), now=NOW), NOW


def edited(q):
    call = q.message.edit_text.await_args
    return call.args[0], call.kwargs.get("reply_markup")


def button_rows(markup):
    return None if markup is None else [[(b.text, b.callback_data) for b in row] for row in markup.inline_keyboard]


@pytest.mark.asyncio
async def test_memok_stores_and_edits(tmp_path):
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    p1 = d.store.pending_add("about", "Lives in Lisbon", turn_id=5, now=now)
    p2 = d.store.pending_add("context", "Moving in March", turn_id=5, now=now)
    rows = [[("Запомнить 1", f"memok:{p1}")], [("Запомнить 2", f"memok:{p2}")], [("Undo", "undoturn:5")]]
    q = mem_cb(f"memok:{p1}", "Хорошо, это я не записала.", rows)
    await on_memok(q, d)
    [fact] = d.store.memory_list()
    assert fact["text"] == "Lives in Lisbon" and fact["section"] == "about" and fact["source"] == "button"
    assert fact["turn_id"] == 5
    text, kb = edited(q)
    assert text == f"Хорошо, это я не записала.\nЗапомнила (M{fact['id']}): Lives in Lisbon"
    assert button_rows(kb) == [[("Запомнить 2", f"memok:{p2}")], [("Undo", "undoturn:5")]]
    pending = d.store.pending_get(p1)
    assert pending["status"] == "stored" and pending["memory_id"] == fact["id"]
    [w] = d.store.agent_writes(5)
    assert w["tool"] == "remember" and w["item_id"] is None
    assert w["before"] == {"memory_id": fact["id"], "op": "remember"} and w["after"] == {"memory_id": fact["id"]}
    q.answer.assert_awaited()


@pytest.mark.asyncio
async def test_memok_only_row_removes_the_keyboard_and_english_wording(tmp_path):
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    d.store.add_turn(1, "user", "please remember I like tea")
    pid = d.store.pending_add("about", "Likes tea", turn_id=2, now=now)
    q = mem_cb(f"memok:{pid}", "Noted.", [[("Remember 1", f"memok:{pid}")]])
    await on_memok(q, d)
    text, kb = edited(q)
    assert text == "Noted.\nRemembered (M1): Likes tea" and button_rows(kb) == [[("Undo", "undoturn:2")]]


@pytest.mark.asyncio
async def test_memok_old_message_still_stores(tmp_path):
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    pid = d.store.pending_add("about", "Fact", turn_id=5, now=now)
    q = mem_cb(f"memok:{pid}", "x", [[("Запомнить", f"memok:{pid}")]])
    q.message.edit_text = AsyncMock(side_effect=RuntimeError("message can't be edited"))
    await on_memok(q, d)
    assert len(d.store.memory_list()) == 1
    q.answer.assert_awaited()


@pytest.mark.asyncio
async def test_memok_twice_stores_once(tmp_path):
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    pid = d.store.pending_add("about", "Fact", turn_id=5, now=now)
    await on_memok(mem_cb(f"memok:{pid}", "x", [[("Запомнить", f"memok:{pid}")]]), d)
    q = mem_cb(f"memok:{pid}", "x", [[("Запомнить", f"memok:{pid}")]])
    await on_memok(q, d)
    assert len(d.store.memory_list()) == 1 and len(d.store.agent_writes(5)) == 1
    q.answer.assert_awaited_with("Уже обработано")
    q.message.edit_text.assert_not_awaited()
    q = mem_cb("memok:999")
    await on_memok(q, d)  # a proposal that never existed
    q.answer.assert_awaited_with("Уже обработано")
    q = mem_cb("memok:abc")
    await on_memok(q, d)
    assert len(d.store.memory_list()) == 1


@pytest.mark.asyncio
async def test_memok_expired(tmp_path):
    from datetime import timedelta
    from clawdia.bot import on_memok
    from clawdia.memory import PENDING_TTL
    d, now = mem_deps(tmp_path)
    pid = d.store.pending_add("about", "Old", turn_id=5, now=now - PENDING_TTL - timedelta(minutes=1))
    q = mem_cb(f"memok:{pid}", "Reply", [[("Запомнить", f"memok:{pid}")], [("Undo", "undoturn:5")]])
    await on_memok(q, d)
    assert d.store.memory_list() == [] and d.store.agent_writes(5) == []
    assert d.store.pending_get(pid)["status"] == "expired"
    text, kb = edited(q)
    assert text == "Reply\nПредложение устарело, скажи ещё раз" and button_rows(kb) == [[("Undo", "undoturn:5")]]
    q2 = mem_cb(f"memok:{pid}")
    await on_memok(q2, d)
    q2.answer.assert_awaited_with("Уже обработано")
    fresh = d.store.pending_add("about", "New", turn_id=6, now=now - PENDING_TTL + timedelta(minutes=1))
    await on_memok(mem_cb(f"memok:{fresh}", "r", [[("Запомнить", f"memok:{fresh}")]]), d)
    assert [f["text"] for f in d.store.memory_list()] == ["New"]


@pytest.mark.asyncio
async def test_memok_full_keeps_pending(tmp_path):
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    for i in range(7):
        d.store.memory_add("about", "x" * 285, source="chat", turn_id=None, now=now)  # 1995 of 2000
    pid = d.store.pending_add("about", "Will not fit", turn_id=5, now=now)
    q = mem_cb(f"memok:{pid}", "r", [[("Запомнить", f"memok:{pid}")]])
    await on_memok(q, d)
    assert len(d.store.memory_list()) == 7 and d.store.agent_writes(5) == []
    assert d.store.pending_get(pid)["status"] == "open"
    assert "2000" in q.answer.await_args.args[0]
    q.message.edit_text.assert_not_awaited()
    d.store.memory_delete(d.store.memory_list()[0]["id"], now)  # drop something, press again
    q = mem_cb(f"memok:{pid}", "r", [[("Запомнить", f"memok:{pid}")]])
    await on_memok(q, d)
    assert d.store.pending_get(pid)["status"] == "stored" and len(d.store.memory_list()) == 7


@pytest.mark.asyncio
async def test_undo_turn_reverts_remember_and_forget(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    kept = d.store.memory_add("about", "Old fact", source="chat", turn_id=None, now=now)
    d.store.memory_delete(kept, now)  # forgotten in turn 5
    added = d.store.memory_add("context", "New fact", source="chat", turn_id=5, now=now)
    d.store.add_agent_write(5, None, "remember", {"memory_id": added, "op": "remember"}, {"memory_id": added})
    d.store.add_agent_write(5, None, "forget", {"memory_id": kept, "op": "forget"}, {"memory_id": kept, "op": "forget"})
    q = cb("undoturn:5")
    await on_undo_turn(q, d)
    assert [f["id"] for f in d.store.memory_list()] == [kept]
    assert d.store.memory_get(added)["deleted_at"] is not None
    assert q.message.edit_text.await_args.args[0] == "Отменила."
    assert q.message.edit_text.await_args.kwargs.get("reply_markup") is None
    q = cb("undoturn:5")
    await on_undo_turn(q, d)
    q.answer.assert_awaited_with("Уже отменено")
    assert [f["id"] for f in d.store.memory_list()] == [kept]


@pytest.mark.asyncio
async def test_undo_forget_ignores_bounds(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    old = d.store.memory_add("about", "y" * 200, source="chat", turn_id=None, now=now)
    d.store.memory_delete(old, now)
    for _ in range(7):
        d.store.memory_add("about", "x" * 285, source="chat", turn_id=None, now=now)
    d.store.add_agent_write(5, None, "forget", {"memory_id": old, "op": "forget"}, {"memory_id": old, "op": "forget"})
    await on_undo_turn(cb("undoturn:5"), d)
    assert d.store.memory_total("about") == 1995 + 200  # over the 2000 bound, by design


@pytest.mark.asyncio
async def test_undo_remember_already_forgotten(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    mid = d.store.memory_add("about", "Fact", source="chat", turn_id=5, now=now)
    d.store.add_agent_write(5, None, "remember", {"memory_id": mid, "op": "remember"}, {"memory_id": mid})
    d.store.memory_delete(mid, now)  # forgotten in a later turn
    q = cb("undoturn:5")
    await on_undo_turn(q, d)
    assert d.store.memory_list() == [] and q.message.edit_text.await_args.args[0] == "Отменила."
    # Undo of a forget whose fact was re-added as a new fact: no duplicate live fact
    gone = d.store.memory_add("about", "Same words", source="chat", turn_id=None, now=now)
    d.store.memory_delete(gone, now)
    d.store.add_agent_write(6, None, "forget", {"memory_id": gone, "op": "forget"}, {"memory_id": gone, "op": "forget"})
    d.store.memory_add("about", "Same words", source="chat", turn_id=None, now=now)
    await on_undo_turn(cb("undoturn:6"), d)
    assert [f["text"] for f in d.store.memory_list()] == ["Same words"]
    # and a missing fact is not an error either
    d.store.add_agent_write(7, None, "forget", {"memory_id": 4040, "op": "forget"}, {"memory_id": 4040, "op": "forget"})
    q = cb("undoturn:7")
    await on_undo_turn(q, d)
    assert q.message.edit_text.await_args.args[0] == "Отменила."


@pytest.mark.asyncio
async def test_undo_turn_memory_and_item_writes_together(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    a = d.store.create_item(kind="idea", status="done", title="A")
    mid = d.store.memory_add("about", "Fact", source="chat", turn_id=5, now=now)
    d.store.add_agent_write(5, a, "mark_done", {"item": {"status": "new"}}, {"status": "done"})
    d.store.add_agent_write(5, None, "remember", {"memory_id": mid, "op": "remember"}, {"memory_id": mid})
    q = cb("undoturn:5")
    await on_undo_turn(q, d)
    assert d.store.get_item(a)["status"] == "new" and d.store.memory_list() == []
    assert q.message.edit_text.await_args.args[0] == f"Отменила #{a}."


@pytest.mark.asyncio
async def test_memok_on_proposals_only_reply_adds_an_undo_row_that_reverts(tmp_path):
    from clawdia.bot import on_memok, on_undo_turn
    d, now = mem_deps(tmp_path)
    p1 = d.store.pending_add("about", "Fact one", turn_id=5, now=now)
    p2 = d.store.pending_add("about", "Fact two", turn_id=5, now=now)
    q = mem_cb(f"memok:{p1}", "r", [[("Запомнить 1", f"memok:{p1}")], [("Запомнить 2", f"memok:{p2}")]])
    await on_memok(q, d)
    _, kb = edited(q)
    assert button_rows(kb) == [[("Запомнить 2", f"memok:{p2}")], [("Undo", "undoturn:5")]]
    [fact] = d.store.memory_list()
    q = cb("undoturn:5")
    await on_undo_turn(q, d)
    assert d.store.memory_list() == [] and d.store.memory_get(fact["id"])["deleted_at"] is not None


@pytest.mark.asyncio
async def test_memok_edit_of_a_near_limit_message_fits_and_keeps_proposal_lines(tmp_path):
    from clawdia.bot import on_memok
    from clawdia.memory import proposal_line
    d, now = mem_deps(tmp_path)
    t1, t2 = "A" * 280, "B" * 290
    p1 = d.store.pending_add("about", t1, turn_id=5, now=now)
    p2 = d.store.pending_add("context", t2, turn_id=5, now=now)
    body = ("reply text line\n" * 250)[:3300]
    original = body + "\n\n" + proposal_line(1, "about", t1, "ru") + "\n" + proposal_line(2, "context", t2, "ru")
    assert 4096 - len(original) < 400
    q = mem_cb(f"memok:{p1}", original, [[("Запомнить 1", f"memok:{p1}")], [("Запомнить 2", f"memok:{p2}")]])
    await on_memok(q, d)
    text, _ = edited(q)
    assert len(text) <= 4096
    assert proposal_line(2, "context", t2, "ru") in text and proposal_line(1, "about", t1, "ru") in text
    assert text.endswith(f"Запомнила (M1): {t1}") and text.startswith("reply text line")


@pytest.mark.asyncio
async def test_memok_expired_edit_fits_too(tmp_path):
    from datetime import timedelta
    from clawdia.bot import on_memok
    from clawdia.memory import PENDING_TTL
    d, now = mem_deps(tmp_path)
    pid = d.store.pending_add("about", "Old", turn_id=5, now=now - PENDING_TTL - timedelta(minutes=1))
    q = mem_cb(f"memok:{pid}", "x" * 4090, [[("Запомнить 1", f"memok:{pid}")]])
    await on_memok(q, d)
    assert len(edited(q)[0]) <= 4096 and edited(q)[0].endswith("Предложение устарело, скажи ещё раз")


@pytest.mark.asyncio
async def test_memok_refuses_a_fact_already_remembered(tmp_path):
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    m = d.store.memory_add("about", "Lives in Lisbon", source="chat", turn_id=None, now=now)
    pid = d.store.pending_add("about", "lives in lisbon", turn_id=5, now=now)
    q = mem_cb(f"memok:{pid}", "r", [[("Запомнить 1", f"memok:{pid}")]])
    await on_memok(q, d)
    assert len(d.store.memory_list()) == 1 and d.store.pending_get(pid)["status"] == "open"
    assert q.answer.await_args.args[0] == f"already remembered as M{m}"
    q.message.edit_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_memok_waits_for_a_running_drain(tmp_path):
    import asyncio
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    pid = d.store.pending_add("about", "Fact", turn_id=5, now=now)
    await d.drain_lock.acquire()
    t = asyncio.create_task(on_memok(mem_cb(f"memok:{pid}", "r", [[("Запомнить 1", f"memok:{pid}")]]), d))
    await asyncio.sleep(0.02)
    assert not t.done() and d.store.memory_list() == []
    d.drain_lock.release(); await t
    assert len(d.store.memory_list()) == 1


@pytest.mark.asyncio
async def test_undo_turn_keeps_an_open_memory_proposal(tmp_path):
    from clawdia.bot import on_undo_turn
    from clawdia.memory import proposal_line
    d, now = mem_deps(tmp_path)
    i = d.store.create_item(kind="task", title="A", text="A", status="new", chat_id=1)
    d.store.update_item(i, status="done")
    d.store.add_agent_write(5, i, "mark_done", {"item": {"status": "new"}}, {"status": "done"})
    pid = d.store.pending_add("about", "Lives in Lisbon", turn_id=5, now=now)
    line = proposal_line(1, "about", "Lives in Lisbon", "ru")
    q = mem_cb("undoturn:5", f"Готово.\n\n{line}", [[("Запомнить 1", f"memok:{pid}")], [("Undo", "undoturn:5")]])
    await on_undo_turn(q, d)
    text, kb = edited(q)
    assert d.store.get_item(i)["status"] == "new"
    assert text.endswith(line) and "Предлагаю запомнить" in text
    assert button_rows(kb) == [[("Запомнить 1", f"memok:{pid}")]]


@pytest.mark.asyncio
async def test_memok_replaces_the_item_undo_when_it_is_a_write_of_the_turn(tmp_path):
    # the real path: a single save_note write keeps the capture Undo (undo:<id>) next to the Remember button
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    i = d.store.create_item(kind="idea", title="A", text="A", status="new", chat_id=1)
    d.store.add_agent_write(5, i, "save_note", {"created": True}, {})
    pid = d.store.pending_add("about", "Lives in Lisbon", turn_id=5, now=now)
    q = mem_cb(f"memok:{pid}", "Сохранила.", [[("Запомнить 1", f"memok:{pid}")], [("Undo", f"undo:{i}")]])
    await on_memok(q, d)
    assert button_rows(edited(q)[1]) == [[("Undo", "undoturn:5")]]


@pytest.mark.asyncio
async def test_memok_keeps_a_foreign_item_undo_and_labels_the_added_one(tmp_path):
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    pid = d.store.pending_add("about", "Lives in Lisbon", turn_id=5, now=now)
    q = mem_cb(f"memok:{pid}", "Сохранила.", [[("Запомнить 1", f"memok:{pid}")], [("Undo", "undo:77")]])
    await on_memok(q, d)
    assert button_rows(edited(q)[1]) == [[("Undo", "undo:77")], [("Отменить запоминание", "undoturn:5")]]


@pytest.mark.asyncio
async def test_undo_turn_of_a_created_vacancy_tells_second_skip(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    d.second.patch_job = AsyncMock(return_value={})
    v = d.store.create_item(kind="job", title="Role", text="Role", status="new", chat_id=1, fields={"job_intake_id": 41})
    d.store.add_agent_write(5, v, "save_note", {"created": True}, {})
    q = mem_cb("undoturn:5", "Сохранила.", [[("Undo", "undoturn:5")]])
    await on_undo_turn(q, d)
    assert d.store.get_item(v)["status"] == "dismissed"
    d.second.patch_job.assert_awaited_once_with(41, "skip", "Undone in Clawdia")


@pytest.mark.asyncio
async def test_item_undo_keeps_an_open_memory_proposal(tmp_path):
    from clawdia.memory import proposal_line
    d, now = mem_deps(tmp_path)
    i = d.store.create_item(kind="idea", title="A", text="A", status="new", chat_id=1)
    pid = d.store.pending_add("about", "Lives in Lisbon", turn_id=5, now=now)
    line = proposal_line(1, "about", "Lives in Lisbon", "ru")
    q = mem_cb(f"undo:{i}", f"Сохранила.\n\n{line}", [[("Запомнить 1", f"memok:{pid}")], [("Undo", f"undo:{i}")]])
    await on_undo(q, d)
    text, kb = edited(q)
    assert d.store.get_item(i)["status"] == "dismissed"
    assert text.endswith(line) and button_rows(kb) == [[("Запомнить 1", f"memok:{pid}")]]


@pytest.mark.asyncio
async def test_undo_drops_a_proposal_that_is_no_longer_open_and_memok_after_undo_keeps_text(tmp_path):
    from clawdia.bot import on_memok, on_undo_turn
    from clawdia.memory import proposal_line
    d, now = mem_deps(tmp_path)
    i = d.store.create_item(kind="task", title="A", text="A", status="new", chat_id=1)
    d.store.update_item(i, status="done")
    d.store.add_agent_write(5, i, "mark_done", {"item": {"status": "new"}}, {"status": "done"})
    p1 = d.store.pending_add("about", "Lives in Lisbon", turn_id=5, now=now)
    p2 = d.store.pending_add("context", "Moving in March", turn_id=5, now=now)
    d.store.pending_close(p1, "stored", None, now)  # stored from another copy of the message meanwhile
    l1, l2 = proposal_line(1, "about", "Lives in Lisbon", "ru"), proposal_line(2, "context", "Moving in March", "ru")
    rows = [[("Запомнить 1", f"memok:{p1}")], [("Запомнить 2", f"memok:{p2}")], [("Undo", "undoturn:5")]]
    q = mem_cb("undoturn:5", f"Готово.\n\n{l1}\n{l2}", rows)
    await on_undo_turn(q, d)
    text, kb = edited(q)
    assert button_rows(kb) == [[("Запомнить 2", f"memok:{p2}")]] and l2 in text and l1 not in text
    # the memok press then edits the current (undone) text, not the original reply
    q2 = mem_cb(f"memok:{p2}", text, button_rows(kb))
    await on_memok(q2, d)
    text2, _ = edited(q2)
    assert text2.startswith("Отменила #") and "Запомнила (M" in text2 and "Готово." not in text2


@pytest.mark.asyncio
async def test_undo_turn_save_then_apply_sends_only_skip_and_stays_dismissed(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    d.second.patch_job = AsyncMock(return_value={"id": 41, "status": "queued"})
    after = {"job_intake_id": 41, "job_decision": "apply"}
    v = d.store.create_item(kind="job", title="Role", text="Role", status="routing", chat_id=1, fields=after)
    d.store.add_agent_write(5, v, "save_note", {"created": True}, {})
    d.store.add_agent_write(5, v, "decide_vacancy", {"intake_id": 41, "item": {"fields": {"job_intake_id": 41}}},
                            {"fields": after})
    await on_undo_turn(mem_cb("undoturn:5", "ok", [[("Undo", "undoturn:5")]]), d)
    d.second.patch_job.assert_awaited_once_with(41, "skip", "Undone in Clawdia")
    assert d.store.get_item(v)["status"] == "dismissed"


@pytest.mark.asyncio
async def test_undo_does_not_skip_a_vacancy_decided_later(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    d.second.patch_job = AsyncMock(return_value={})
    v = d.store.create_item(kind="job", title="Role", text="Role", status="routing", chat_id=1,
                            fields={"job_intake_id": 41, "job_decision": "apply"})
    d.store.add_agent_write(5, v, "save_note", {"created": True}, {})
    q = mem_cb("undoturn:5", "ok", [[("Undo", "undoturn:5")]])
    await on_undo_turn(q, d)
    d.second.patch_job.assert_not_awaited()
    assert d.store.get_item(v)["status"] == "routing" and "изменилась после" in edited(q)[0]
    # parity: the item Undo of a decided vacancy leaves it too
    q2 = mem_cb(f"undo:{v}", "ok", [[("Undo", f"undo:{v}")]])
    await on_undo(q2, d)
    d.second.patch_job.assert_not_awaited()
    assert d.store.get_item(v)["status"] == "routing"


@pytest.mark.asyncio
async def test_callwait_cancel_button(tmp_path):
    from datetime import timedelta
    from clawdia.bot import on_callwait_cancel
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    w = d.store.wait_open(1, 1, 1, None, 0, d.now_fn(), timedelta(hours=3), "q")
    q = cb(f"callwait:cancel:{w}")
    await on_callwait_cancel(q, d)
    assert d.store.wait_get(w)["status"] == "cancelled"
    q.answer.assert_awaited_with("Отменено")
    d.store.wait_set_status(w, "answering")  # a claimed wait is not cancelled by a late press
    late = cb(f"callwait:cancel:{w}")
    await on_callwait_cancel(late, d)
    assert d.store.wait_get(w)["status"] == "answering"
    late.answer.assert_awaited_with("Уже отвечено или закрыто")
    await on_callwait_cancel(cb("callwait:cancel:x"), d)  # junk data changes nothing


@pytest.mark.asyncio
async def test_undo_turn_cancels_open_wait(tmp_path):
    from datetime import timedelta
    from clawdia.bot import on_undo_turn
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    open_w = d.store.wait_open(1, 5, 1, None, 0, d.now_fn(), timedelta(hours=3), "q")
    claimed = d.store.wait_open(1, 5, 1, None, 0, d.now_fn(), timedelta(hours=3), "q")
    d.store.wait_set_status(claimed, "answering")
    for w in (open_w, claimed):
        d.store.add_agent_write(5, None, "wait_for_call", {"wait_id": w, "op": "wait"}, {"wait_id": w})
    q = cb("undoturn:5")
    await on_undo_turn(q, d)
    assert d.store.wait_get(open_w)["status"] == "cancelled"
    assert d.store.wait_get(claimed)["status"] == "answering"
    assert all(w["undone"] for w in d.store.agent_writes(5))
    assert q.message.edit_text.await_args.args[0] == "Отменила."


@pytest.mark.asyncio
async def test_focusreq_button_stores_one_request(tmp_path):
    from clawdia.bot import on_focusreq
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    q = cb("focusreq:open")
    await on_focusreq(q, d)
    assert d.store.meta_get("focus_request") and d.store.meta_get("focus_requested_at")
    first = d.store.meta_get("focus_request")
    q.answer.assert_awaited_with("Запрос отправлен; ноутбук соберёт карту, когда проснётся.")
    await on_focusreq(cb("focusreq:open"), d)  # a second press keeps the first request
    assert d.store.meta_get("focus_request") == first
    await on_focusreq(cb("focusreq:junk"), d)  # junk data changes nothing and does not raise


@pytest.mark.asyncio
async def test_focusreq_refused_when_todays_map_exists(tmp_path):
    from clawdia.bot import on_focusreq
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    d.settings.focus_dir = tmp_path / "focus"
    d.settings.focus_dir.mkdir()
    (d.settings.focus_dir / f"{d.now_fn().date().isoformat()}.md").write_text("# Daily Focus\n", encoding="utf-8")
    q = cb("focusreq:open")
    await on_focusreq(q, d)
    q.answer.assert_awaited_with("Карта дня на сегодня уже есть.")
    assert not d.store.meta_get("focus_request")


@pytest.mark.asyncio
async def test_focusreq_answer_in_english(tmp_path):
    from clawdia.bot import on_focusreq
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    d.store.add_turn(1, "user", "what is on today please", processed=True, at=d.now_fn())
    q = cb("focusreq:open")
    await on_focusreq(q, d)
    q.answer.assert_awaited_with("Request sent; the laptop builds the map when it wakes.")


def test_focusreq_is_routed():
    import inspect
    from clawdia import bot
    assert '"focusreq:"' in inspect.getsource(bot.build_router)


@pytest.mark.asyncio
async def test_undo_turn_restores_digest_settings(tmp_path):
    from clawdia.bot import on_undo_turn
    d = deps(tmp_path, TagResult("idea", "x", None, {}, 0.8, None, "m"))
    d.store.meta_set("digest_time", "10:00")
    d.store.add_agent_write(7, None, "set_digest", {"enabled": None, "time": "09:30", "days": None},
                            {"enabled": None, "time": "10:00", "days": None})
    await on_undo_turn(cb("undoturn:7"), d)
    assert d.store.meta_get("digest_time") == "09:30"
    assert all(w["undone"] for w in d.store.agent_writes(7))


@pytest.mark.asyncio
async def test_memok_stores_a_rules_fact(tmp_path):
    from clawdia.bot import on_memok
    d, now = mem_deps(tmp_path)
    pid = d.store.pending_add("rules", "Answer in Russian", turn_id=5, now=now)
    q = mem_cb(f"memok:{pid}", "Ок.", [[("Запомнить", f"memok:{pid}")]])
    await on_memok(q, d)
    [fact] = d.store.memory_list()
    assert fact["section"] == "rules" and fact["text"] == "Answer in Russian" and fact["source"] == "button"


@pytest.mark.asyncio
async def test_undo_turn_logs_a_miss(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    tid = d.store.add_turn(1, "user", "посмотри вакансию", processed=False, at=now)
    i = d.store.create_item(kind="task", title="T", text="T", status="done", chat_id=1)
    d.store.add_agent_write(tid, i, "mark_done", {"item": {"status": "new", "route_error": None}, "reminders": []},
                            {"status": "done"})
    q = mem_cb(f"undoturn:{tid}", "Закрыла.", [[("Отменить", f"undoturn:{tid}")]])
    await on_undo_turn(q, d)
    [m] = d.store.misses_since(now.replace(year=2020))
    assert m["reason"] == "undone" and m["topic"] == "отмена ответа" and m["request"] == "посмотри вакансию"
    assert m["turn_id"] == tid


@pytest.mark.asyncio
async def test_undo_turn_of_only_settings_logs_no_miss_and_blank_text_has_a_label(tmp_path):
    from clawdia.bot import on_undo_turn
    d, now = mem_deps(tmp_path)
    tid = d.store.add_turn(1, "user", "выключи обзор", processed=False, at=now)
    d.store.add_agent_write(tid, None, "set_review", {"review_enabled": "1"}, {"review_enabled": "0"})
    await on_undo_turn(mem_cb(f"undoturn:{tid}", "Ок.", [[("Отменить", f"undoturn:{tid}")]]), d)
    assert d.store.misses_since(now.replace(year=2020)) == []
    t2 = d.store.add_turn(1, "user", "  ", processed=False, at=now)
    i = d.store.create_item(kind="task", title="T", text="T", status="done", chat_id=1)
    d.store.add_agent_write(t2, i, "mark_done", {"item": {"status": "new", "route_error": None}, "reminders": []},
                            {"status": "done"})
    await on_undo_turn(mem_cb(f"undoturn:{t2}", "Ок.", [[("Отменить", f"undoturn:{t2}")]]), d)
    [m] = d.store.misses_since(now.replace(year=2020))
    assert m["request"] == "(без текста)"
