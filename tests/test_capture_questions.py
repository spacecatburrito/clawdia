"""Her question sent together with a capture (photo caption, comment glued to a forward) gets an answer."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from clawdia.capture import Captured
from clawdia.converse import _own_words, handle_capture
from tests.test_agent import CHAT, NOW, FakeClient, make, seed, text_resp, tool_resp


def photo_msg(caption, forward=None):
    return SimpleNamespace(text=None, caption=caption, entities=[], caption_entities=[], forward_origin=forward,
                           media_group_id=None, message_id=70, date=NOW, chat=SimpleNamespace(id=CHAT),
                           from_user=SimpleNamespace(id=CHAT), reply_to_message=None, photo=[object()], voice=None,
                           audio=None, document=None)


def captured(item_id):
    return Captured(item_id, "unsorted", "Photo", f"Сохранила как #{item_id}. Это задача?", [[("Undo", f"undo:{item_id}")]])


def test_own_words_never_take_a_forwards_text():
    assert _own_words(photo_msg("Что на картинке?"), glued=False) == "Что на картинке?"
    assert _own_words(photo_msg("Что на картинке?", forward=SimpleNamespace(type="user")), glued=False) == ""
    glued = SimpleNamespace(text="Эту вакансию проверяли уже?", caption=None, forward_origin=None)
    assert _own_words(glued, glued=True) == "Эту вакансию проверяли уже?"


async def test_photo_question_is_answered_with_the_item_in_view(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила как #N. Это задача?")
    i = seed(d, "Фото: каменная плита с розеткой из отверстий", kind="unsorted")
    d.ai_client = FakeClient(tool_resp(("get_item", {"id": i})), text_resp("На фото каменная плита с розеткой."))
    out = await handle_capture(photo_msg("Что изображено на картинке?"), d, captured(i))
    assert out.text == "На фото каменная плита с розеткой."  # only the answer, no "saved as"
    assert not out.keyboard and d.store.get_item(i) is not None
    first = d.ai_client.requests[0]["messages"][-1]["content"]
    assert f"together with item #{i}" in first and first.rstrip().endswith("Что изображено на картинке?")


async def test_no_question_or_a_forward_keeps_the_capture_reply(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Photo", kind="unsorted")
    await handle_capture(photo_msg("рецепт супа"), d, captured(i))
    await handle_capture(photo_msg("Что это?", forward=SimpleNamespace(type="user")), d, captured(i))
    assert d.ai_client.requests == []


async def test_agent_failure_keeps_the_capture_reply(tmp_path):
    d = make(tmp_path, RuntimeError("boom"))
    i = seed(d, "Photo", kind="unsorted")
    out = await handle_capture(photo_msg("Что это?"), d, captured(i))
    assert out.item_id == i and "\n\n" not in out.text


async def test_item_notes_reach_the_agent(tmp_path):
    import json

    from clawdia.agent_tools import call_tool
    from tests.test_agent import _turn

    d = make(tmp_path)
    i = seed(d, "Granite early team", kind="job", status="dismissed", notes="No product role posted")
    out = json.loads(await call_tool(_turn(d), "get_item", {"id": i}))
    assert out["notes"] == "No product role posted"


async def test_context_item_turn_is_dirty_so_remember_gives_a_proposal(tmp_path):
    # the item she sent is captured content: a memory write in that turn must wait for a button
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: визитка, Иван Петров, CEO", kind="unsorted")
    d.ai_client = FakeClient(tool_resp(("remember", {"section": "context", "text": "Иван Петров - CEO"})),
                             text_resp("Предложила запомнить."))
    out = await handle_capture(photo_msg("Что на фото? И запомни это"), d, captured(i))
    assert d.store.memory_list() == []  # nothing stored without the button
    row = d.store.pending_get(1)
    assert row is not None and row["status"] == "open" and row["text"] == "Иван Петров - CEO"
    assert "Иван Петров - CEO" in out.text  # the proposal line shows the exact text
    assert out.keyboard and any(cb == "memok:1" for rows in out.keyboard for _, cb in rows)
    assert out.source == "agent_data" and out.item_id == i


# -- an answered question does not clutter the inbox ----------------------------------------------

OPEN = ("new", "routing")  # what `clawdia inbox` lists by default


def tagged(d, item_id):
    """The phase 1 tag call succeeded for this item (a "tagged" event)."""
    d.store.add_event(item_id, "tagged", {"kind": "unsorted", "model": "test"})


def glued_msg(question):
    m = photo_msg(None, forward=SimpleNamespace(type="user"))
    m.text, m.photo = question, None
    return m


async def test_photo_question_answered_closes_the_unsorted_item(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: каменная плита", kind="unsorted", fields={"asked": True})
    tagged(d, i)
    d.ai_client = FakeClient(text_resp("На фото каменная плита."))
    out = await handle_capture(photo_msg("Что изображено на картинке?"), d, captured(i))
    assert out.text == "На фото каменная плита."
    item = d.store.get_item(i)
    assert item["status"] == "done" and item["fields"] == {"asked": True, "answered": True}
    assert "answered" in d.store.event_types(i)
    assert d.store.list_items(status=OPEN) == []  # `clawdia inbox` default view
    assert [x["id"] for x in d.store.list_items(status=None)] == [i]  # /list "all" still has it


async def test_inbox_cli_does_not_list_an_answered_item(tmp_path):
    from clawdia.cli import cmd_inbox
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: каменная плита", kind="unsorted")
    tagged(d, i)
    d.ai_client = FakeClient(text_resp("Плита."))
    await handle_capture(photo_msg("Что это?"), d, captured(i))
    args = SimpleNamespace(kind=None, status=None, since=None, limit=50, json=True)
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_inbox(args, SimpleNamespace(timezone="UTC"), d.store, NOW)
    assert buf.getvalue().strip() == "[]"
    args.status = "done"
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_inbox(args, SimpleNamespace(timezone="UTC"), d.store, NOW)
    assert f'"id": {i}' in buf.getvalue()


async def test_glued_question_about_an_unsorted_forward_closes_it(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Пересланное: что-то про отчёт", kind="unsorted")
    tagged(d, i)
    turn = d.store.add_turn(CHAT, "user", "Это про что?", processed=False, at=NOW)
    d.ai_client = FakeClient(text_resp("Про отчёты Nimbus."))
    out = await handle_capture(glued_msg("Это про что?"), d, captured(i), turn_id=turn)
    assert out.text == "Про отчёты Nimbus."
    assert d.store.get_item(i)["status"] == "done" and d.store.get_item(i)["fields"]["answered"] is True
    assert d.store.list_items(status=OPEN) == []


async def test_classified_capture_stays_open_after_the_answer(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Staff PM at Acme", kind="job")
    turn = d.store.add_turn(CHAT, "user", "Эту вакансию уже смотрели?", processed=False, at=NOW)
    d.ai_client = FakeClient(text_resp("Нет, новая."))
    out = await handle_capture(glued_msg("Эту вакансию уже смотрели?"), d, captured(i), turn_id=turn)
    assert out.text == "Нет, новая."
    item = d.store.get_item(i)
    assert item["status"] == "new" and not (item["fields"] or {}).get("answered")
    assert "answered" not in d.store.event_types(i)
    assert [x["id"] for x in d.store.list_items(status=OPEN)] == [i]


async def test_own_voice_question_leaves_no_open_item(tmp_path):
    from tests.test_voice_agent import voice_msg, with_audio
    from clawdia.converse import handle_voice
    d = make(tmp_path)
    with_audio(d, tmp_path, "что у меня сегодня по календарю?")
    d.ai_client = FakeClient(text_resp("Сегодня созвон в 15:00."))
    out = await handle_voice(voice_msg(), d)
    assert out is not None and "созвон" in out.text
    assert d.store.list_items(status=OPEN) == []


async def test_agent_failure_leaves_the_item_as_it_was(tmp_path):
    d = make(tmp_path, RuntimeError("boom"))
    i = seed(d, "Photo", kind="unsorted", fields={"asked": True})
    await handle_capture(photo_msg("Что это?"), d, captured(i))
    item = d.store.get_item(i)
    assert item["status"] == "new" and item["fields"] == {"asked": True}
    assert "answered" not in d.store.event_types(i)


async def test_answer_that_set_a_reminder_keeps_the_item_open(tmp_path):
    from datetime import timedelta
    from clawdia.reminders import send_due
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: афиша", kind="unsorted")
    tagged(d, i)
    at = (NOW + timedelta(days=2)).isoformat()
    d.ai_client = FakeClient(tool_resp(("set_reminder", {"id": i, "at": at})), text_resp("Напомню в субботу."))
    await handle_capture(photo_msg("Напомнишь мне про это в субботу?"), d, captured(i))
    assert d.store.get_item(i)["status"] == "new"
    assert "answered" not in d.store.event_types(i)
    bot = SimpleNamespace(send_message=AsyncMock())
    assert await send_due(d.store, bot, 1, NOW + timedelta(days=3)) == 1


async def test_item_with_a_phase_one_reminder_stays_open_after_the_answer(tmp_path):
    from datetime import timedelta
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: афиша", kind="unsorted")
    tagged(d, i)
    d.store.add_reminder(i, NOW + timedelta(days=2))
    d.ai_client = FakeClient(text_resp("Это афиша."))
    await handle_capture(photo_msg("Что это за афиша?"), d, captured(i))
    assert d.store.get_item(i)["status"] == "new"
    assert "answered" not in d.store.event_types(i)


async def test_sent_or_disabled_reminder_does_not_keep_the_item_open(tmp_path):
    from datetime import timedelta
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: афиша", kind="unsorted")
    tagged(d, i)
    rid = d.store.add_reminder(i, NOW - timedelta(days=2))
    d.store.mark_reminder_sent(rid, NOW)
    d.store.disable_reminders(i)
    d.ai_client = FakeClient(text_resp("Это афиша."))
    await handle_capture(photo_msg("Что это за афиша?"), d, captured(i))
    assert d.store.get_item(i)["status"] == "done"


async def test_ask_when_keeps_the_item_open(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: афиша", kind="unsorted", fields={"ask_when": True})
    tagged(d, i)
    d.ai_client = FakeClient(text_resp("Это афиша."))
    await handle_capture(photo_msg("Напомнишь про это? Что это?"), d, captured(i))
    assert d.store.get_item(i)["status"] == "new"


async def test_an_agent_write_on_the_item_keeps_it_open(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: афиша", kind="unsorted")
    tagged(d, i)
    d.ai_client = FakeClient(tool_resp(("update_item", {"id": i, "title": "Афиша"})), text_resp("Назвала афишей."))
    await handle_capture(photo_msg("Назови это как-нибудь?"), d, captured(i))
    assert d.store.get_item(i)["status"] == "new"


async def test_tag_failure_plus_question_keeps_the_item_open(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: афиша", kind="unsorted")  # no "tagged" event: tagging failed
    d.ai_client = FakeClient(text_resp("Это афиша."))
    await handle_capture(photo_msg("Что это?"), d, captured(i))
    assert d.store.get_item(i)["status"] == "new"
    assert "answered" not in d.store.event_types(i)


def test_asks_something_detects_requests_without_a_question_mark():
    from clawdia.converse import asks_something
    for yes in ("посмотри", "Ответь на это", "как тебе", "what do you think", "Check this", "ok?"):
        assert asks_something(yes), yes
    for no in ("вакансия", "#idea", "на завтра", "", "recheck later"):
        assert not asks_something(no), no


async def test_photo_request_without_question_mark_is_answered(tmp_path):
    d = make(tmp_path)
    d.reply_fn = AsyncMock(return_value="Сохранила.")
    i = seed(d, "Фото: плита", kind="unsorted")
    d.ai_client = FakeClient(text_resp("Это каменная плита."))
    out = await handle_capture(photo_msg("посмотри"), d, captured(i))
    assert out.text == "Это каменная плита."
