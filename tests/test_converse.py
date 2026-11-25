from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawdia.context import Snapshot
from clawdia.converse import Outcome, guard, handle_text
from clawdia.second_client import SecondUnavailable
from clawdia.storage import utc_iso
from clawdia.tagger import TagResult
from clawdia.understand import Understanding
from tests.test_capture import deps

NOW = datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc)  # Thursday 10:00 Lisbon: a work day after 07:00
FAR = datetime(2030, 1, 1, tzinfo=timezone.utc)
CHAT = 1


async def echo(facts, *, settings, client=None):
    return facts.fallback


def make(tmp_path, *understandings, now=NOW, tag=None, projects=()):
    d = deps(tmp_path, tag or TagResult("unsorted", "x", None, {}, 0.0, None, "m"), now=now)
    d.understand_fn = AsyncMock(side_effect=list(understandings))
    d.reply_fn = AsyncMock(side_effect=echo)
    d.second.patch_task = AsyncMock(return_value={"id": 2601, "project": None, "deadline": None})
    if projects:
        snap = Snapshot(projects=[{"name": p} for p in projects])
        d.context_cache = SimpleNamespace(load=lambda: snap, age_seconds=lambda _now: None)
    return d


def msg(text, mid=10, chat=CHAT):
    return SimpleNamespace(text=text, caption=None, entities=[], caption_entities=[], forward_origin=None,
                           media_group_id=None, message_id=mid, date=NOW, chat=SimpleNamespace(id=chat),
                           from_user=SimpleNamespace(id=chat))


def cap(kind="idea", title="Bot without commands", conf=0.9, fields=None, reminder_at=None):
    return TagResult(kind, title, None, fields or {}, conf, reminder_at, "m")


def U(intent, language="en", **kw):
    return Understanding(intent=intent, language=language, **kw)


def seed(d, kind="idea", title="Old note", status="new", chat=CHAT, bot_ago=5, **cols):
    """An item of this chat plus the user and bot turns that created it, the bot turn `bot_ago` minutes ago."""
    item_id = d.store.create_item(kind=kind, title=title, text=title, status=status, chat_id=chat, **cols)
    d.store.add_turn(chat, "user", title, item_id=item_id, at=NOW - timedelta(minutes=bot_ago, seconds=5))
    d.store.add_turn(chat, "bot", f"Saved #{item_id}", item_id=item_id, at=NOW - timedelta(minutes=bot_ago))
    return item_id


def items(d):
    return d.store.list_items(limit=1000)


def active_reminders(d, item_id):
    return [r for r in d.store.due_reminders(FAR) if r["item_id"] == item_id]


def facts(d, call=-1):
    return d.reply_fn.await_args_list[call].args[0]


# -- capture ------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_creates_one_item_with_chat_id_and_undo(tmp_path):
    d = make(tmp_path, U("capture", capture=cap("buy", "Oat milk", 0.95)))
    out = await handle_text(msg("купить овсяное молоко", chat=7), d)
    [item] = items(d)
    assert isinstance(out, Outcome) and out.item_id == item["id"]
    assert item["chat_id"] == 7 and item["kind"] == "buy" and item["text"] == "купить овсяное молоко"
    assert item["tg_message_id"] == 10
    assert out.keyboard == [[("Undo", f"undo:{item['id']}")]]
    assert out.text.startswith(f"Saved #{item['id']} · buy · Oat milk")
    assert d.tag_fn.await_count == 0 and d.understand_fn.await_count == 1 and d.reply_fn.await_count == 1
    f = facts(d)
    assert f.action == "saved" and f.item_id == item["id"] and f.kind == "buy" and f.question is None
    user, bot = d.store.recent_turns(7, 20)
    assert (user["role"], user["processed"], user["item_id"]) == ("user", 1, item["id"])
    assert (bot["role"], bot["text"], bot["item_id"]) == ("bot", out.text, item["id"])


@pytest.mark.asyncio
async def test_user_turn_is_stored_unprocessed_before_the_model_call(tmp_path):
    d = make(tmp_path)
    seen = {}

    async def understand(text, **kw):
        seen["turns"] = d.store.recent_turns(CHAT, 20)
        seen["model_turns"] = kw["turns"]
        return U("capture", capture=cap())

    d.understand_fn = AsyncMock(side_effect=understand)
    await handle_text(msg("идея: бот без команд"), d)
    [turn] = seen["turns"]
    assert turn["role"] == "user" and turn["processed"] == 0 and turn["text"] == "идея: бот без команд"
    assert seen["model_turns"] == []  # the new message goes in <message>, not in the memory block


@pytest.mark.asyncio
async def test_given_turn_id_reuses_that_turn(tmp_path):
    d = make(tmp_path, U("capture", capture=cap()))
    turn = d.store.add_turn(CHAT, "user", "идея", processed=False, at=NOW)
    out = await handle_text(msg("идея"), d, turn_id=turn)
    turns = d.store.recent_turns(CHAT, 20)
    assert [t["role"] for t in turns] == ["user", "bot"] and turns[0]["id"] == turn
    assert turns[0]["processed"] == 1 and turns[0]["item_id"] == out.item_id


@pytest.mark.asyncio
async def test_task_capture_routes_and_second_line_is_in_facts(tmp_path):
    d = make(tmp_path, U("capture", capture=cap("task", "Ask Bruno about the invoice")))
    out = await handle_text(msg("спросить Bruno про инвойс"), d)
    item = d.store.get_item(out.item_id)
    assert item["status"] == "filed" and facts(d).second_line == "-> Second #2601"


@pytest.mark.asyncio
async def test_low_confidence_capture_asks_once(tmp_path):
    d = make(tmp_path, U("capture", capture=cap("idea", conf=0.3), ask="Это задача или идея?"),
             U("capture", capture=cap("idea", conf=0.7), ask="Это задача?"))
    out = await handle_text(msg("бот без команд"), d)
    assert facts(d).question == "Это задача или идея?" and "Это задача или идея?" in out.text
    assert d.store.get_item(out.item_id)["fields"]["asked"] is True
    out2 = await handle_text(msg("ещё мысль"), d)
    assert facts(d).question is None and "asked" not in (d.store.get_item(out2.item_id)["fields"] or {})


@pytest.mark.asyncio
async def test_hashtag_forces_a_capture_with_that_kind(tmp_path):
    d = make(tmp_path, U("capture", capture=cap("idea", conf=0.4)))
    out = await handle_text(msg("#film Dune"), d)
    item = d.store.get_item(out.item_id)
    assert item["kind"] == "film" and item["confidence"] == 1.0 and d.understand_fn.await_count == 1
    assert d.tag_fn.await_count == 0


@pytest.mark.asyncio
async def test_hashtag_on_another_intent_is_tagged_by_the_phase_1_tagger(tmp_path):
    d = make(tmp_path, U("command", command="list", shelf="film"), tag=cap("film", "Dune", 1.0))
    out = await handle_text(msg("#film Dune"), d)
    assert d.store.get_item(out.item_id)["kind"] == "film" and d.tag_fn.await_count == 1
    assert d.reply_fn.await_count == 0


# -- Review Focus 1 and 2 -------------------------------------------------


@pytest.mark.asyncio
async def test_long_note_starting_with_a_command_word_is_saved(tmp_path):
    long_show = "покажи " + " ".join(["слово"] * 9)
    long_done = "done " + " ".join(["word"] * 9)
    d = make(tmp_path, U("command", command="list", shelf="film"), U("command", command="done"))
    film = d.store.create_item(kind="film", title="Dune", status="new", chat_id=CHAT)
    out = await handle_text(msg(long_show), d)
    assert out.item_id is not None and out.item_id != film and d.store.get_item(out.item_id)["text"] == long_show
    assert out.text.startswith(f"Saved #{out.item_id}")
    out2 = await handle_text(msg(long_done), d)  # the last item is the note above, inside the window
    assert d.store.get_item(out.item_id)["status"] == "new" and d.store.get_item(out2.item_id)["text"] == long_done


@pytest.mark.asyncio
async def test_correction_20_minutes_after_the_last_reply_is_a_new_capture(tmp_path):
    d = make(tmp_path, U("correct", language="ru", new_kind="task"), tag=cap("task", "Отдать в Second"))
    old = seed(d, kind="idea", bot_ago=20)
    out = await handle_text(msg("это задача, отдай в Second"), d)
    assert out.item_id != old and d.store.get_item(old)["kind"] == "idea" and d.store.get_item(old)["status"] == "new"
    new = d.store.get_item(out.item_id)
    assert new["text"] == "это задача, отдай в Second" and new["kind"] == "task"
    # the downgrade is tagged by the tag call (second model call), so the reply is the deterministic one
    assert d.understand_fn.await_count + d.tag_fn.await_count == 2 and d.reply_fn.await_count == 0
    assert out.text.startswith(f"Сохранила #{out.item_id} · задача")


# -- corrections --------------------------------------------------------


@pytest.mark.asyncio
async def test_correct_within_window_retags_and_routes(tmp_path):
    d = make(tmp_path, U("correct", language="ru", new_kind="task"))
    old = seed(d, kind="idea")
    out = await handle_text(msg("это задача"), d)
    item = d.store.get_item(old)
    assert out.item_id == old and len(items(d)) == 1
    assert item["kind"] == "task" and item["status"] == "filed" and item["filed_ref"] == "2601"
    assert out.keyboard is None and facts(d).action == "retagged"  # Undo only under capture replies


@pytest.mark.asyncio
async def test_retag_away_from_a_queued_task_leaves_the_outbox(tmp_path):
    d = make(tmp_path, U("correct", new_kind="idea"))
    old = seed(d, kind="task", status="routing")
    await handle_text(msg("it is an idea"), d)
    item = d.store.get_item(old)
    assert item["kind"] == "idea" and item["status"] == "new" and d.store.outbox() == []


@pytest.mark.asyncio
async def test_retag_to_event_routes_it_to_second(tmp_path):
    d = make(tmp_path, U("correct", new_kind="event"))
    old = seed(d, kind="idea")
    await handle_text(msg("it is an event"), d)
    item = d.store.get_item(old)
    assert item["kind"] == "event" and item["status"] == "filed" and item["filed_ref"] == "2601"


@pytest.mark.asyncio
async def test_retag_queued_task_to_event_stays_queued(tmp_path):
    d = make(tmp_path, U("correct", new_kind="event"))
    d.second.post_task = AsyncMock(side_effect=SecondUnavailable("down"))
    old = seed(d, kind="task", status="routing")
    await handle_text(msg("it is an event"), d)
    item = d.store.get_item(old)
    assert item["kind"] == "event" and item["status"] == "routing" and [o["id"] for o in d.store.outbox()] == [old]


@pytest.mark.asyncio
@pytest.mark.parametrize("new_kind", ["idea", "job"])
async def test_retag_queued_event_away_leaves_no_orphan_routing_item(tmp_path, new_kind):
    d = make(tmp_path, U("correct", new_kind=new_kind))
    old = seed(d, kind="event", status="routing")
    await handle_text(msg("not an event"), d)
    item = d.store.get_item(old)
    if new_kind == "job":  # a retag to job is submitted to triage like a fresh capture
        assert item["kind"] == "job" and item["status"] == "routing" and item["fields"]["job_intake_id"] == 77
        return
    assert item["kind"] == new_kind and item["status"] == "new" and d.store.outbox() == []


@pytest.mark.asyncio
async def test_filed_event_retagged_to_a_shelf_gets_the_stays_open_note(tmp_path):
    d = make(tmp_path, U("correct", new_kind="idea"))
    old = seed(d, kind="event", status="filed", filed_ref="2601")
    out = await handle_text(msg("it is an idea"), d)
    assert "Second #2601 stays open" in out.text


@pytest.mark.asyncio
async def test_deadline_correction_on_a_queued_event_reaches_the_later_post(tmp_path):
    d = make(tmp_path, U("correct", deadline="2026-11-20"))
    d.second.post_task = AsyncMock(side_effect=SecondUnavailable("down"))
    old = seed(d, kind="event", status="routing", fields={"date": "2026-11-12"})
    await handle_text(msg("deadline 20 Nov"), d)
    from clawdia.second_client import event_task_body
    assert event_task_body(d.store.get_item(old))["deadline"] == "2026-11-20"


@pytest.mark.asyncio
async def test_send_to_second_next_week_makes_a_task_with_a_deadline(tmp_path):
    d = make(tmp_path, U("correct", language="ru", send_to_second=True, deadline="2026-11-02"))
    old = seed(d, kind="idea")
    out = await handle_text(msg("в Second на следующую неделю"), d)
    item = d.store.get_item(old)
    assert item["kind"] == "task" and item["fields"]["deadline"] == "2026-11-02" and item["status"] == "filed"
    assert d.second.post_task.await_args.args[0]["fields"]["deadline"] == "2026-11-02"
    assert "2026-11-02" in out.text and d.second.patch_task.await_count == 0


@pytest.mark.asyncio
async def test_deadline_on_a_filed_task_is_patched_in_second_at_once(tmp_path):
    d = make(tmp_path, U("correct", deadline="2026-11-06"))
    old = seed(d, kind="task", status="filed", filed_ref="2601")
    d.second.patch_task = AsyncMock(return_value={"id": 2601, "project": None, "deadline": "2026-11-06"})
    out = await handle_text(msg("deadline Friday"), d)
    d.second.patch_task.assert_awaited_once_with(2601, deadline="2026-11-06", project=None)
    fields = d.store.get_item(old)["fields"]
    assert fields["deadline"] == "2026-11-06" and "second_patch" not in fields
    assert "Second #2601 updated" in out.text


@pytest.mark.asyncio
async def test_deadline_patch_kept_when_second_is_down(tmp_path):
    d = make(tmp_path, U("correct", deadline="2026-11-06"))
    old = seed(d, kind="task", status="filed", filed_ref="2601")
    d.second.patch_task = AsyncMock(side_effect=SecondUnavailable("down"))
    out = await handle_text(msg("deadline Friday"), d)
    assert d.store.get_item(old)["fields"]["second_patch"] == {"deadline": "2026-11-06"}
    assert "Second will get it when it is reachable" in out.text


@pytest.mark.asyncio
async def test_project_correction_on_a_filed_task_patches_the_project(tmp_path):
    d = make(tmp_path, U("correct", new_project="acme"), projects=("Acme",))
    old = seed(d, kind="task", status="filed", filed_ref="2601")
    d.second.patch_task = AsyncMock(return_value={"id": 2601, "project": "Acme", "deadline": None})
    await handle_text(msg("make it Acme"), d)
    d.second.patch_task.assert_awaited_once_with(2601, deadline=None, project="Acme")
    assert d.store.get_item(old)["project"] == "Acme"


@pytest.mark.asyncio
async def test_unknown_project_changes_nothing(tmp_path):
    d = make(tmp_path, U("correct", new_project=None, new_project_raw="Foo", new_kind="task"), projects=("Acme",))
    old = seed(d, kind="idea")
    out = await handle_text(msg("make it Foo, a task"), d)
    item = d.store.get_item(old)
    assert item["kind"] == "idea" and item["project"] is None and "No project Foo" in out.text


@pytest.mark.asyncio
async def test_correct_reminder_at_and_reminders_off(tmp_path):
    at = "2026-10-31T09:00:00+00:00"
    d = make(tmp_path, U("correct", reminder_at=at), U("correct", reminders_off=True))
    old = seed(d, kind="event")
    d.store.add_reminder(old, NOW + timedelta(days=1))
    await handle_text(msg("remind me Saturday"), d)
    assert [r["at"] for r in active_reminders(d, old)] == [utc_iso(at)]
    await handle_text(msg("no reminders"), d)
    assert active_reminders(d, old) == [] and d.store.get_item(old)["fields"]["reminders_off"] is True


@pytest.mark.asyncio
async def test_delete_without_explicit_id_outside_the_window_is_a_capture(tmp_path):
    d = make(tmp_path, U("correct", language="ru", delete=True))
    old = seed(d, bot_ago=30)
    out = await handle_text(msg("удали это"), d)
    assert d.store.get_item(old)["status"] == "new" and out.item_id != old and len(items(d)) == 2


@pytest.mark.asyncio
async def test_delete_with_explicit_id_works_outside_the_window(tmp_path):
    d = make(tmp_path, U("correct", language="ru", delete=True, explicit_id=None))
    old = seed(d, bot_ago=120, status="filed", filed_ref="2601")
    d.understand_fn = AsyncMock(return_value=U("correct", language="ru", delete=True, explicit_id=old))
    out = await handle_text(msg(f"удали #{old}"), d)
    assert d.store.get_item(old)["status"] == "dismissed" and out.keyboard is None and len(items(d)) == 1
    assert out.text.startswith(f"Отменила #{old}.") and "Second #2601 stays open" in out.text


@pytest.mark.asyncio
async def test_explicit_id_that_does_not_exist_changes_nothing(tmp_path):
    d = make(tmp_path, U("correct", delete=True, explicit_id=999))
    old = seed(d)
    out = await handle_text(msg("delete #999"), d)
    assert d.store.get_item(old)["status"] == "new" and "No item #999" in out.text and out.item_id is None
    assert out.text.endswith("Not saved.")


@pytest.mark.asyncio
async def test_rejected_target_asks_which_item_and_changes_nothing(tmp_path):
    d = make(tmp_path, U("correct", new_kind="task", delete=True, target_rejected=True))
    a = seed(d, title="First")
    b = seed(d, title="Second one")
    out = await handle_text(msg("this one is a task"), d)
    assert d.store.get_item(a)["kind"] == "idea" and d.store.get_item(b)["kind"] == "idea"
    assert d.store.get_item(b)["status"] == "new" and "Which item" in out.text and f"#{b}" in out.text


# -- follow-ups ---------------------------------------------------------


@pytest.mark.asyncio
async def test_followup_reminder_at_schedules_exactly_that_time(tmp_path):
    at = "2026-10-31T09:00:00+00:00"
    d = make(tmp_path, U("followup", language="ru", reminder_at=at))
    old = seed(d, kind="event", fields={"ask_when": True})
    d.store.add_reminder(old, NOW + timedelta(days=1))  # an older default reminder is replaced
    out = await handle_text(msg("в субботу"), d)
    assert [r["at"] for r in active_reminders(d, old)] == [utc_iso(at)]
    fields = d.store.get_item(old)["fields"]
    assert "ask_when" not in fields and fields["reminders"] == [utc_iso(at)]
    assert facts(d).reminder_times == ["Sat 09:00"] and out.item_id == old and len(items(d)) == 1


@pytest.mark.asyncio
async def test_followup_reminder_in_the_past_sets_nothing(tmp_path):
    d = make(tmp_path, U("followup", reminder_at="2026-10-28T09:00:00+00:00"))
    old = seed(d)
    out = await handle_text(msg("remind me yesterday"), d)
    assert active_reminders(d, old) == [] and "passed" in out.text


@pytest.mark.asyncio
async def test_followup_reminders_off_disables(tmp_path):
    d = make(tmp_path, U("followup", language="ru", reminders_off=True))
    old = seed(d, kind="event")
    d.store.add_reminder(old, NOW + timedelta(days=1))
    await handle_text(msg("не напоминай"), d)
    assert active_reminders(d, old) == [] and d.store.get_item(old)["fields"]["reminders_off"] is True


@pytest.mark.asyncio
async def test_followup_snooze_moves_the_latest_reminder(tmp_path):
    d = make(tmp_path, U("followup", snooze_days=7))
    old = seed(d, kind="event")
    d.store.add_reminder(old, NOW - timedelta(hours=1))
    await handle_text(msg("snooze a week"), d)
    rem = d.store.latest_reminder(old)
    assert rem["snoozed_until"] == utc_iso(NOW + timedelta(days=7)) and d.store.due_reminders(NOW) == []


@pytest.mark.asyncio
async def test_followup_mark_done_on_the_last_item(tmp_path):
    d = make(tmp_path, U("followup", mark_done=True))
    a = seed(d, title="First")
    b = seed(d, title="Second one")
    out = await handle_text(msg("done"), d)
    assert d.store.get_item(b)["status"] == "done" and d.store.get_item(a)["status"] == "new"
    assert out.text.startswith(f"Done #{b}") and out.item_id == b


@pytest.mark.asyncio
async def test_mark_done_on_a_model_target_that_is_not_the_last_item_is_refused(tmp_path):
    d = make(tmp_path, U("followup", mark_done=True, target_item_id=None))
    a = seed(d, title="First")
    b = seed(d, title="Second one")
    d.understand_fn = AsyncMock(return_value=U("followup", mark_done=True, target_item_id=a))
    out = await handle_text(msg("the first one is done"), d)
    assert d.store.get_item(a)["status"] == "new" and d.store.get_item(b)["status"] == "new"
    assert "Which item" in out.text


@pytest.mark.asyncio
async def test_save_previous_captures_the_previous_user_text(tmp_path):
    d = make(tmp_path, U("chat", language="ru"), U("followup", language="ru", save_previous=True),
             tag=cap("buy", "Milk", 0.9))
    first = await handle_text(msg("купить молоко", mid=1), d)
    assert first.item_id is None and items(d) == []
    out = await handle_text(msg("сохрани это", mid=2), d)
    [item] = items(d)
    assert item["text"] == "купить молоко" and item["kind"] == "buy" and item["chat_id"] == CHAT
    assert d.tag_fn.await_count == 1 and d.reply_fn.await_count == 1  # only the chat reply used the model
    assert out.item_id == item["id"] and out.text.startswith(f"Сохранила #{item['id']}")
    user_turns = [t for t in d.store.recent_turns(CHAT, 20) if t["role"] == "user"]
    assert user_turns[0]["item_id"] == item["id"]


@pytest.mark.asyncio
async def test_save_previous_does_not_duplicate_a_saved_note(tmp_path):
    d = make(tmp_path, U("capture", capture=cap("buy", "Milk")), U("followup", save_previous=True))
    first = await handle_text(msg("купить молоко", mid=1), d)
    out = await handle_text(msg("save that", mid=2), d)
    assert len(items(d)) == 1 and out.item_id == first.item_id and f"#{first.item_id}" in out.text


# -- commands -----------------------------------------------------------


@pytest.mark.asyncio
async def test_command_list_film_returns_the_film_list(tmp_path):
    d = make(tmp_path, U("command", language="ru", command="list", shelf="film"))
    a = d.store.create_item(kind="film", title="Dune", status="new", chat_id=CHAT)
    b = d.store.create_item(kind="film", title="Arrival", status="new", chat_id=CHAT)
    out = await handle_text(msg("покажи фильмы"), d)
    assert out.text.splitlines()[0] == "film · 2 open" and f"#{a} · Dune" in out.text and f"#{b} · Arrival" in out.text
    assert out.keyboard is None and out.item_id is None and len(items(d)) == 2
    assert facts(d).action == "listed"


@pytest.mark.asyncio
async def test_command_lead_sentence_goes_above_the_block(tmp_path):
    d = make(tmp_path, U("command", command="list", shelf="film"))
    d.store.create_item(kind="film", title="Dune", status="new", chat_id=CHAT)
    d.reply_fn = AsyncMock(return_value="Here are your films.")
    out = await handle_text(msg("show films"), d)
    assert out.text.startswith("Here are your films.\nfilm · 1 open")


@pytest.mark.asyncio
async def test_command_list_without_a_shelf_asks_which(tmp_path):
    d = make(tmp_path, U("command", command="list"))
    out = await handle_text(msg("show my list"), d)
    assert "Which shelf" in out.text and "film" in out.text and items(d) == [] and out.text.endswith("Not saved.")


@pytest.mark.asyncio
async def test_command_show_done_drop_inbox_help_status(tmp_path):
    d = make(tmp_path, U("command", command="show", explicit_id=None), U("command", command="drop"),
             U("command", command="done"), U("command", command="inbox"), U("command", command="help"),
             U("command", command="status"))
    a = seed(d, title="First", bot_ago=60)
    b = seed(d, title="Second one", bot_ago=60)
    out = await handle_text(msg(f"show #{a}"), d)
    assert out.text.startswith(f"#{a} · idea · new") and out.item_id == a
    out = await handle_text(msg("drop it"), d)  # the item the reply above touched, inside its window
    assert out.text == f"Dropped #{a} · First" and d.store.get_item(a)["status"] == "dismissed"
    out = await handle_text(msg(f"готово {b}"), d)
    assert out.text == f"Done #{b} · Second one" and d.store.get_item(b)["status"] == "done"
    assert (await handle_text(msg("inbox"), d)).text.startswith("Inbox (new, routing, filed)")
    assert "/list" in (await handle_text(msg("help"), d)).text
    assert (await handle_text(msg("status"), d)).text.startswith("Captured today:")


@pytest.mark.asyncio
async def test_command_done_without_id_outside_the_window_asks(tmp_path):
    d = make(tmp_path, U("command", command="done"))
    old = seed(d, bot_ago=60)
    out = await handle_text(msg("done"), d)
    assert d.store.get_item(old)["status"] == "new" and "done #" in out.text.lower()


@pytest.mark.asyncio
async def test_command_flush_and_sync(tmp_path):
    d = make(tmp_path, U("command", command="flush"), U("command", command="sync"))
    out = await handle_text(msg("flush"), d)
    assert out.text == "Filed 0"
    out = await handle_text(msg("sync"), d)
    assert out.text == "Filed 0, context unchanged"


# -- chat ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_saves_nothing_and_ends_with_not_saved(tmp_path):
    d = make(tmp_path, U("chat", language="ru"), U("chat", language="en"), U("chat", language="en"))
    out = await handle_text(msg("что ты умеешь?"), d)
    assert items(d) == [] and out.item_id is None and out.keyboard is None and out.text.endswith("Не сохранила.")
    assert facts(d).action == "not saved"
    out = await handle_text(msg("what can you do?"), d)
    assert out.text.endswith("Not saved.")
    d.reply_fn = AsyncMock(return_value="I save notes and fix them when you tell me.")
    out = await handle_text(msg("hi"), d)
    assert out.text == "I save notes and fix them when you tell me. Not saved."


# -- guard --------------------------------------------------------------


def _g(u, text, *, last_bot_at=NOW - timedelta(minutes=5), recent_ids=(3, 2), exists=lambda _i: True):
    from clawdia.config import Settings
    settings = Settings(bot_token="t", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k")
    return guard(u, text=text, now=NOW, last_bot_at=last_bot_at, recent_ids=list(recent_ids), settings=settings,
                 item_exists=exists)


def test_guard_command_rules():
    assert _g(U("command"), "покажи").intent == "capture"  # no command name
    assert _g(U("command", command="list", shelf="film"), "list https://x.y/films").intent == "capture"
    assert _g(U("command", command="list", shelf="film"), " ".join(["w"] * 9)).intent == "capture"
    assert _g(U("command", command="list", shelf="film"), " ".join(["w"] * 8)).intent == "command"
    assert _g(U("command", command="done"), "готово 3").target_item_id == 3
    assert _g(U("command", command="list", shelf="film"), "list film 2").explicit_id is None
    closed = _g(U("command", command="done", explicit_id=3), "закрой задачу 3")
    assert closed.explicit_id is None and closed.target_item_id is None  # bare number not right after the verb


def test_guard_window_and_targets():
    no_bot = _g(U("followup", mark_done=True), "done", last_bot_at=None)
    assert no_bot.intent == "capture" and no_bot.capture is None  # the capture executor tags it
    assert _g(U("correct", new_kind="task"), "это задача").target_item_id == 3
    assert _g(U("correct", new_kind="task", target_item_id=2), "the older one").target_item_id == 2
    not_last = _g(U("correct", delete=True, target_item_id=2), "delete the older one")
    assert not_last.delete is False and not_last.target_item_id is None
    rejected = _g(U("correct", delete=True, target_rejected=True), "delete it")
    assert rejected.delete is False and rejected.target_item_id is None
    missing = _g(U("followup", mark_done=True, explicit_id=9), "#9 done", exists=lambda _i: False)
    assert missing.mark_done is False and missing.target_item_id is None and missing.intent == "followup"
    late_missing = _g(U("correct", delete=True, explicit_id=9), "#9 delete",
                      last_bot_at=NOW - timedelta(hours=1), exists=lambda _i: False)
    assert late_missing.intent == "capture"
    late = _g(U("correct", delete=True, explicit_id=9), "#9 delete", last_bot_at=NOW - timedelta(hours=1))
    assert late.intent == "correct" and late.target_item_id == 9 and late.delete is True
    assert _g(U("correct", new_kind="task"), "это задача", recent_ids=()).intent == "capture"
    kept = _g(U("capture", capture=cap("buy", "Milk")), "купить молоко")
    assert kept.intent == "capture" and kept.capture.kind == "buy"


@pytest.mark.asyncio
async def test_a_failing_step_after_the_item_was_saved_does_not_duplicate_it(tmp_path):
    d = make(tmp_path, U("capture", capture=cap("buy", "Milk")))
    d.reply_fn = AsyncMock(side_effect=RuntimeError("boom"))
    out = await handle_text(msg("купить молоко"), d)
    [item] = items(d)
    assert out.item_id == item["id"] and out.text.startswith(f"Saved #{item['id']}")
    assert d.store.recent_turns(CHAT, 20)[0]["processed"] == 1


@pytest.mark.asyncio
async def test_a_failing_correction_keeps_the_text_as_a_note(tmp_path):
    d = make(tmp_path, U("correct", new_kind="task"))
    old = seed(d)
    d.second.post_task = AsyncMock(return_value={"id": 2601, "project": None})
    d.reply_fn = AsyncMock(side_effect=RuntimeError("boom"))
    out = await handle_text(msg("это задача"), d)
    assert d.store.get_item(old)["kind"] == "task" and d.store.get_item(out.item_id)["text"] == "это задача"


def test_settings_conversation_defaults_and_env(monkeypatch):
    from clawdia.config import Settings
    req = dict(bot_token="t", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k")
    s = Settings(**req)
    assert (s.followup_window_min, s.ask_below) == (15, 0.5)
    monkeypatch.setenv("CLAWDIA_FOLLOWUP_WINDOW_MIN", "5")
    monkeypatch.setenv("CLAWDIA_ASK_BELOW", "0.3")
    s = Settings(**req)
    assert (s.followup_window_min, s.ask_below) == (5, 0.3)


# -- fix round 1 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_last_item_is_the_one_touched_last_not_the_newest(tmp_path):
    d = make(tmp_path, U("correct", language="ru", delete=True))
    a = seed(d, title="First", bot_ago=3)
    b = seed(d, title="Second one", bot_ago=2)
    d.store.add_turn(CHAT, "user", f"#{a} это задача", item_id=a, at=NOW - timedelta(minutes=1, seconds=5))
    d.store.add_turn(CHAT, "bot", f"Retagged #{a}", item_id=a, at=NOW - timedelta(minutes=1))
    await handle_text(msg("удали"), d)
    assert d.store.get_item(a)["status"] == "dismissed" and d.store.get_item(b)["status"] == "new"


@pytest.mark.asyncio
async def test_done_without_a_touched_item_asks_instead_of_using_the_newest_item(tmp_path):
    d = make(tmp_path, U("followup", mark_done=True))
    a = seed(d, title="First", bot_ago=30)
    b = d.store.create_item(kind="idea", title="Second one", text="x", status="new", chat_id=CHAT)
    d.store.add_turn(CHAT, "bot", "Filed 0", at=NOW - timedelta(minutes=2))  # a reply without an item
    out = await handle_text(msg("done"), d)
    assert d.store.get_item(b)["status"] == "new" and d.store.get_item(a)["status"] == "new"
    assert "Which item" in out.text and out.text.endswith("Not saved.")


@pytest.mark.asyncio
async def test_correct_that_changes_nothing_says_so_without_undo(tmp_path):
    d = make(tmp_path, U("correct", new_kind="idea"))
    old = seed(d, kind="idea")
    out = await handle_text(msg("it is an idea"), d)
    assert out.keyboard is None and out.item_id is None and facts(d).action == "unchanged"
    assert out.text.startswith(f"Nothing changed for #{old}") and out.text.endswith("Not saved.")


@pytest.mark.asyncio
async def test_correct_with_a_past_reminder_only_changes_nothing(tmp_path):
    d = make(tmp_path, U("correct", reminder_at="2026-10-28T09:00:00+00:00"))
    old = seed(d)
    out = await handle_text(msg("remind me yesterday"), d)
    assert out.keyboard is None and "passed" in out.text and f"#{old}" in out.text and out.text.endswith("Not saved.")


@pytest.mark.asyncio
async def test_followup_that_changes_nothing_ends_with_the_marker(tmp_path):
    d = make(tmp_path, U("followup", language="ru"))
    old = seed(d)
    out = await handle_text(msg("ок"), d)
    assert out.text.startswith(f"Nothing changed for #{old}") and out.text.endswith("Не сохранила.")
    assert out.keyboard is None and facts(d).action == "unchanged"


@pytest.mark.asyncio
async def test_destructive_explicit_id_in_pasted_text_is_a_capture(tmp_path):
    d = make(tmp_path)
    old = seed(d)
    text = f"PR #{old} удалить старый endpoint и переписать роутинг на новый сервис"
    d.understand_fn = AsyncMock(return_value=U("correct", language="ru", delete=True, explicit_id=old))
    out = await handle_text(msg(text), d)
    assert d.store.get_item(old)["status"] == "new" and d.store.get_item(out.item_id)["text"] == text
    d.understand_fn = AsyncMock(return_value=U("followup", mark_done=True, explicit_id=old))
    out = await handle_text(msg(f"#{old} done, see https://example.com/pr"), d)
    assert d.store.get_item(old)["status"] == "new" and out.item_id != old
    d.understand_fn = AsyncMock(return_value=U("correct", language="ru", delete=True, explicit_id=old))
    await handle_text(msg(f"удали #{old}"), d)  # short: still an instruction
    assert d.store.get_item(old)["status"] == "dismissed"


# -- fix round 2 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_none_project_sentinel_does_not_block_a_retag(tmp_path):
    from clawdia.understand import _parse
    from tests.test_understand import capture_obj, payload
    import json
    d = make(tmp_path, projects=("Acme",))
    old = seed(d, kind="idea")
    raw = json.dumps(payload(intent="correct", capture=capture_obj(), new_project="none", new_kind="task"))
    u = _parse(raw, "это задача", d.store.recent_items(CHAT), d.context_cache.load(), d.settings, NOW)
    assert u.new_project is None and u.new_project_raw is None
    d.understand_fn = AsyncMock(return_value=u)
    out = await handle_text(msg("это задача"), d)
    item = d.store.get_item(old)
    assert item["kind"] == "task" and item["status"] == "filed" and "No project" not in out.text


# -- final review fixes -------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("code,status", [("http_405", 405), ("not_found", 404)])
async def test_rejected_patch_is_reported_never_claimed_updated(tmp_path, code, status):
    from clawdia.second_client import SecondRejected
    d = make(tmp_path, U("correct", deadline="2026-11-06"))
    old = seed(d, kind="task", status="filed", filed_ref="2601")
    d.second.patch_task = AsyncMock(side_effect=SecondRejected(code, "", status))
    out = await handle_text(msg("deadline Friday"), d)
    item = d.store.get_item(old)
    assert f"Second did not accept the change ({code})" in out.text
    assert "updated" not in out.text.split("\n", 1)[1] and "Second #2601 updated" not in out.text
    assert item["route_error"] == code and "second_patch" not in item["fields"]
    assert f"Second did not accept the change ({code})" in facts(d).extra


@pytest.mark.asyncio
async def test_rejected_patch_reply_in_russian(tmp_path):
    from clawdia.second_client import SecondRejected
    d = make(tmp_path, U("correct", language="ru", deadline="2026-11-06"))
    seed(d, kind="task", status="filed", filed_ref="2601")
    d.second.patch_task = AsyncMock(side_effect=SecondRejected("not_found", "", 404))
    out = await handle_text(msg("дедлайн пятница"), d)
    assert "Second не принял изменение (not_found)" in out.text and "Обновила Second" not in out.text


@pytest.mark.asyncio
async def test_patch_blocked_by_an_older_failing_patch_is_pending_not_updated(tmp_path):
    d = make(tmp_path, U("correct", deadline="2026-11-06"))
    older = d.store.create_item(kind="task", status="filed", filed_ref="2500", title="Older", chat_id=CHAT,
                                fields={"second_patch": {"deadline": "2026-11-05"}},
                                created_at=NOW - timedelta(days=3))
    old = seed(d, kind="task", status="filed", filed_ref="2601")
    d.second.patch_task = AsyncMock(side_effect=SecondUnavailable("down"))
    out = await handle_text(msg("deadline Friday"), d)
    assert "Second will get it when it is reachable" in out.text and "updated" not in out.text.lower().split("\n", 1)[1]
    assert d.store.get_item(old)["fields"]["second_patch"] == {"deadline": "2026-11-06"}
    assert d.store.get_item(older)["fields"]["second_patch"] == {"deadline": "2026-11-05"}


@pytest.mark.asyncio
async def test_delete_after_an_itemless_reply_asks_instead_of_deleting_yesterdays_item(tmp_path):
    # Probe 1: a chat reply 1 minute ago named no item; the newest item is from yesterday.
    d = make(tmp_path, U("correct", language="ru", delete=True))
    old = seed(d, kind="idea", title="Yesterday", bot_ago=24 * 60)
    d.store.add_turn(CHAT, "user", "привет", at=NOW - timedelta(minutes=1, seconds=5))
    d.store.add_turn(CHAT, "bot", "Привет! Не сохранила.", at=NOW - timedelta(minutes=1))
    out = await handle_text(msg("удали это"), d)
    assert d.store.get_item(old)["status"] == "new"
    assert out.text.startswith("Какую запись?") and out.text.endswith("Не сохранила.") and out.keyboard is None


@pytest.mark.asyncio
async def test_done_after_undo_does_not_close_the_previous_item(tmp_path):
    # Probe 2: capture #5, Undo it, then "готово": the newest live item is not the touched one.
    d = make(tmp_path, U("capture", language="ru", capture=cap("idea", "New idea")),
             U("followup", language="ru", mark_done=True))
    before = seed(d, kind="task", title="Older task", bot_ago=60)
    captured = await handle_text(msg("новая идея"), d)
    d.store.update_item(captured.item_id, status="dismissed")  # what the Undo button does
    out = await handle_text(msg("готово"), d)
    assert d.store.get_item(before)["status"] == "new"
    assert d.store.get_item(captured.item_id)["status"] == "dismissed"
    assert out.text.startswith("Какую запись?") and out.text.endswith("Не сохранила.")


def test_guard_destructive_needs_the_touched_item_but_edits_keep_the_fallback():
    s = SimpleNamespace(followup_window_min=15)
    near = NOW - timedelta(minutes=1)
    for u, text in ((U("correct", delete=True), "удали это"), (U("followup", mark_done=True), "готово")):
        g = guard(u, text=text, now=NOW, last_bot_at=near, recent_ids=[4], settings=s, last_item_id=None)
        assert g.target_item_id is None and not g.delete and not g.mark_done
    for name in ("done", "drop"):
        g = guard(U("command", command=name), text="готово", now=NOW, last_bot_at=near, recent_ids=[4], settings=s)
        assert g.intent == "command" and g.target_item_id is None
    g = guard(U("command", command="done"), text="готово", now=NOW, last_bot_at=near, recent_ids=[4], settings=s,
              last_item_id=4)
    assert g.target_item_id == 4
    g = guard(U("correct", new_kind="task"), text="это задача", now=NOW, last_bot_at=near, recent_ids=[4], settings=s)
    assert g.intent == "correct" and g.target_item_id == 4  # a non-destructive edit may use the newest item
    g = guard(U("command", command="show"), text="покажи", now=NOW, last_bot_at=near, recent_ids=[4], settings=s)
    assert g.target_item_id == 4


@pytest.mark.asyncio
async def test_pasted_hash_id_two_days_later_is_a_capture_and_leaves_the_item(tmp_path):
    text = "PR #12 merge the parser fix today, this is a task for Acme team please"
    d = make(tmp_path, U("correct", new_kind="task", explicit_id=12, target_rejected=True),
             tag=cap("task", "Merge the parser fix"))
    for n in range(12):
        d.store.create_item(kind="idea", title=f"n{n}", text=f"n{n}", status="new", chat_id=CHAT)
    d.store.add_turn(CHAT, "bot", "Saved #11", item_id=11, at=NOW - timedelta(days=2))
    out = await handle_text(msg(text), d)
    assert d.store.get_item(12)["kind"] == "idea" and d.store.get_item(12)["status"] == "new"
    new = d.store.get_item(out.item_id)
    assert out.item_id == 13 and new["text"] == text and new["kind"] == "task"


@pytest.mark.asyncio
async def test_url_with_hash_id_outside_the_window_is_a_capture(tmp_path):
    d = make(tmp_path, U("followup", reminder_at="2026-10-30T09:00:00+00:00", explicit_id=3))
    for n in range(3):
        d.store.create_item(kind="idea", title=f"n{n}", text=f"n{n}", status="new", chat_id=CHAT)
    out = await handle_text(msg("https://x.com/a #3 remind fri"), d)
    assert out.item_id == 4 and active_reminders(d, 3) == []
    assert d.store.get_item(4)["text"] == "https://x.com/a #3 remind fri"


def test_guard_pasted_hash_id_inside_the_window_is_unchanged():
    s = SimpleNamespace(followup_window_min=15)
    text = "PR #12 merge the parser fix today, this is a task for Acme team please"
    g = guard(U("correct", new_kind="task", explicit_id=12), text=text, now=NOW,
              last_bot_at=NOW - timedelta(minutes=2), recent_ids=[40], settings=s)
    assert g.intent == "correct" and g.target_item_id == 12
    g = guard(U("correct", new_kind="task", explicit_id=12), text="#12 это задача", now=NOW,
              last_bot_at=NOW - timedelta(days=2), recent_ids=[40], settings=s)
    assert g.intent == "correct" and g.target_item_id == 12  # a short typed #N still works later


@pytest.mark.asyncio
async def test_save_previous_already_saved_has_no_undo(tmp_path):
    d = make(tmp_path, U("capture", capture=cap("buy", "Milk")), U("followup", save_previous=True))
    first = await handle_text(msg("купить молоко", mid=1), d)
    out = await handle_text(msg("save that", mid=2), d)
    assert out.item_id == first.item_id and out.keyboard is None


@pytest.mark.asyncio
@pytest.mark.parametrize("language,question", [("ru", "Это задача, идея или что-то другое?"),
                                               ("en", "Is this a task, an idea or something else?")])
async def test_low_confidence_capture_without_a_model_question_asks_a_fixed_one_once(tmp_path, language, question):
    d = make(tmp_path, U("capture", language=language, capture=cap("idea", conf=0.3)))
    out = await handle_text(msg("бот без команд"), d)
    assert facts(d).question == question and question in out.text
    item = d.store.get_item(out.item_id)
    assert item["fields"]["asked"] is True
    from clawdia.converse_actions import Ctx, _question
    ctx = Ctx(deps=d, message=None, text="", chat_id=CHAT, turn_id=0, now=NOW, u=U("capture", language=language),
              recent=[], history=[], snapshot=Snapshot())
    assert _question(ctx, out.item_id, cap("idea", conf=0.3), None) is None  # once per item


@pytest.mark.asyncio
async def test_failed_understanding_asks_nothing_and_skips_the_reply_call(tmp_path):
    error = TagResult("unsorted", "бот без команд", None, {}, 0.0, None, "error")
    d = make(tmp_path, Understanding(intent="capture", language="ru", capture=error, model="error"))
    out = await handle_text(msg("бот без команд"), d)
    assert d.reply_fn.await_count == 0  # the model is likely down: no second timeout
    assert out.text == f"Сохранила #{out.item_id} · без категории · бот без команд"
    assert "asked" not in (d.store.get_item(out.item_id)["fields"] or {})
    assert out.keyboard == [[("Undo", f"undo:{out.item_id}")]]


def _calls(caplog, turn_id):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(f"turn {turn_id} model_calls=")]


@pytest.mark.asyncio
async def test_model_calls_are_counted_per_message(tmp_path, caplog):
    caplog.set_level("INFO", logger="clawdia")
    d = make(tmp_path, U("capture", capture=cap("buy", "Oat milk", 0.95)),
             U("correct", language="ru", new_kind="task"),  # 20 minutes later: downgraded to a capture
             U("chat"))
    out = await handle_text(msg("купить овсяное молоко"), d)
    capture_turn = d.store.recent_turns(CHAT, 20)[-2]["id"]
    assert _calls(caplog, capture_turn) == [f"turn {capture_turn} model_calls=2"]

    d.now_fn = lambda: NOW + timedelta(minutes=20)
    await handle_text(msg("это задача"), d)
    down_turn = d.store.recent_turns(CHAT, 20)[-2]["id"]
    assert _calls(caplog, down_turn) == [f"turn {down_turn} model_calls=2"]
    assert d.tag_fn.await_count == 1

    await handle_text(msg("привет"), d)
    chat_turn = d.store.recent_turns(CHAT, 20)[-2]["id"]
    assert _calls(caplog, chat_turn) == [f"turn {chat_turn} model_calls=2"]
    assert "купить" not in caplog.text and "привет" not in caplog.text  # ids only
    assert out.item_id is not None


@pytest.mark.asyncio
async def test_model_calls_counted_for_a_media_capture(tmp_path, caplog):
    from clawdia.capture import process
    from clawdia.converse import handle_capture
    caplog.set_level("INFO", logger="clawdia")
    d = make(tmp_path, tag=cap("recipe", "Pasta", 0.9))
    m = msg(None)
    m.caption, m.photo = "рецепт пасты", [SimpleNamespace(file_id="a")]
    captured = await process(m, d)
    await handle_capture(m, d, captured)
    turn = [t for t in d.store.recent_turns(CHAT, 20) if t["role"] == "user"][-1]["id"]
    assert _calls(caplog, turn) == [f"turn {turn} model_calls=2"]
    assert [t["source"] for t in d.store.recent_turns(CHAT, 20) if t["role"] == "user"] == ["capture"]
    assert [t["source"] for t in d.store.recent_turns(CHAT, 20) if t["role"] == "bot"] == ["capture_reply"]


@pytest.mark.asyncio
async def test_correction_fallback_leads_with_updated(tmp_path):
    d = make(tmp_path, U("correct", new_kind="film"), U("correct", language="ru", new_kind="recipe"))
    old = seed(d, kind="idea", title="Dune")
    out = await handle_text(msg("it's a film"), d)
    assert out.text.startswith(f"Updated #{old} · film · Dune")
    out = await handle_text(msg("это рецепт"), d)
    assert out.text.startswith(f"Обновила #{old} · рецепт · Dune")


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "look https://example.com/post",
    " ".join(["word"] * 26),
])
async def test_chat_with_a_url_or_long_text_is_saved(tmp_path, text):
    d = make(tmp_path, U("chat"), tag=cap("idea", "Saved thing"))
    out = await handle_text(msg(text), d)
    [item] = items(d)
    assert out.item_id == item["id"] and item["text"] == text and out.keyboard is not None


@pytest.mark.asyncio
async def test_short_chat_stays_chat(tmp_path):
    d = make(tmp_path, U("chat"))
    out = await handle_text(msg(" ".join(["word"] * 25)), d)
    assert out.item_id is None and items(d) == []


@pytest.mark.asyncio
async def test_done_typed_after_a_reminder_ping_closes_that_item(tmp_path):
    from clawdia.reminders import send_due
    d = make(tmp_path, U("followup", language="ru", mark_done=True))
    other = seed(d, kind="idea", title="Newest", bot_ago=600)
    pinged = d.store.create_item(kind="place", title="Market", text="Market", status="new", chat_id=CHAT,
                                 created_at=NOW - timedelta(days=3))
    d.store.add_reminder(pinged, NOW - timedelta(minutes=3))
    bot = SimpleNamespace(send_message=AsyncMock())
    assert await send_due(d.store, bot, CHAT, NOW - timedelta(minutes=2)) == 1
    ping = d.store.recent_turns(CHAT, 20)[-1]
    assert (ping["role"], ping["item_id"], ping["text"]) == ("bot", pinged, f"Reminder · #{pinged} · Market")
    await handle_text(msg("готово"), d)
    assert d.store.get_item(pinged)["status"] == "done" and d.store.get_item(other)["status"] == "new"


@pytest.mark.asyncio
async def test_drain_notice_is_a_bot_turn_naming_its_item(tmp_path):
    from clawdia.bot import make_notify
    d = make(tmp_path)
    bot = SimpleNamespace(send_message=AsyncMock())
    i = d.store.create_item(kind="task", title="Old task", text="x", status="filed", chat_id=CHAT)
    await make_notify(bot, d.store, CHAT)(i, 2601, "Acme")
    [turn] = d.store.recent_turns(CHAT, 20)
    assert (turn["role"], turn["item_id"]) == ("bot", i) and turn["text"] == bot.send_message.await_args.args[1]


@pytest.mark.asyncio
async def test_late_turns_become_quiet_unsorted_notes(tmp_path, caplog):
    from clawdia.converse import recover_unprocessed
    caplog.set_level("INFO", logger="clawdia")
    d = make(tmp_path)
    d.bot = SimpleNamespace(send_message=AsyncMock())
    late = d.store.add_turn(CHAT, "user", "SECRET late note", processed=False, at=NOW - timedelta(hours=3))
    saved_turn = d.store.add_turn(CHAT, "user", "already saved", processed=False, at=NOW - timedelta(hours=2))
    kept = d.store.create_item(kind="buy", title="x", text="already saved", status="new", chat_id=CHAT,
                               created_at=NOW - timedelta(hours=2))
    blank = d.store.add_turn(CHAT, "user", "   ", processed=False, at=NOW - timedelta(hours=2))
    assert await recover_unprocessed(d) == 0
    new = [i for i in items(d) if i["id"] != kept]
    assert [(i["text"], i["kind"], i["chat_id"], i["status"]) for i in new] == [("SECRET late note", "unsorted", CHAT, "new")]
    assert d.understand_fn.await_count == 0 and d.tag_fn.await_count == 0 and d.reply_fn.await_count == 0
    assert d.bot.send_message.await_count == 0
    turns = {t["id"]: t for t in d.store.recent_turns(CHAT, 20)}
    assert turns[late]["processed"] == 1 and turns[late]["item_id"] == new[0]["id"]
    assert turns[saved_turn]["item_id"] == kept and turns[blank]["processed"] == 1
    with d.store._lock:
        types = [r["type"] for r in d.store._conn.execute("SELECT type FROM events WHERE item_id = ?", (new[0]["id"],))]
    assert types == ["recovered_late"] and "SECRET" not in caplog.text
    assert d.store.count_events("recovered_late", NOW - timedelta(days=1)) == 1
    assert await recover_unprocessed(d) == 0 and len(items(d)) == 2  # idempotent


@pytest.mark.asyncio
async def test_media_capture_reply_failure_falls_back_in_the_message_language(tmp_path):
    from clawdia.capture import process
    from clawdia.converse import handle_capture
    d = make(tmp_path, tag=cap("recipe", "Pasta", 0.9))
    d.reply_fn = AsyncMock(side_effect=RuntimeError("boom"))
    m = msg(None)
    m.caption, m.photo = "рецепт пасты", [SimpleNamespace(file_id="a")]
    captured = await process(m, d)
    out = await handle_capture(m, d, captured)
    assert out.text == f"Сохранила #{captured.item_id} · рецепт · Pasta"
    assert out.keyboard == [[("Undo", f"undo:{captured.item_id}")]]


@pytest.mark.asyncio
async def test_downgraded_low_confidence_capture_asks_the_fixed_question(tmp_path):
    text = " ".join(["слово"] * 26)
    d = make(tmp_path, U("chat", language="ru"), tag=cap("idea", "Thing", conf=0.3))
    out = await handle_text(msg(text), d)
    [item] = items(d)
    assert out.text.endswith("Это задача, идея или что-то другое?") and item["fields"]["asked"] is True
    assert d.reply_fn.await_count == 0  # still the deterministic reply after the tag call


@pytest.mark.asyncio
async def test_downgrade_with_a_failed_tag_asks_nothing(tmp_path):
    d = make(tmp_path, U("chat"), tag=TagResult("unsorted", "x", None, {}, 0.0, None, "error"))
    out = await handle_text(msg("look https://example.com/post"), d)
    assert "something else" not in out.text and "asked" not in (d.store.get_item(out.item_id)["fields"] or {})


async def _ping(d, *titles, at):
    from clawdia.reminders import send_due
    ids = []
    for t in titles:
        i = d.store.create_item(kind="place", title=t, text=t, status="new", chat_id=CHAT,
                                created_at=NOW - timedelta(days=3))
        d.store.add_reminder(i, at - timedelta(minutes=1))
        ids.append(i)
    assert await send_due(d.store, SimpleNamespace(send_message=AsyncMock()), CHAT, at) == len(titles)
    return ids


@pytest.mark.asyncio
async def test_done_after_three_pings_asks_which_one(tmp_path):
    d = make(tmp_path, U("followup", language="ru", mark_done=True))
    ids = await _ping(d, "A", "B", "C", at=NOW - timedelta(minutes=2))
    out = await handle_text(msg("готово"), d)
    assert [d.store.get_item(i)["status"] for i in ids] == ["new", "new", "new"]
    assert out.text.startswith("Какую запись?") and out.text.endswith("Не сохранила.")


@pytest.mark.asyncio
async def test_delete_after_a_capture_and_a_ping_asks_which_one(tmp_path):
    d = make(tmp_path, U("capture", language="ru", capture=cap("idea", "B idea")),
             U("correct", language="ru", delete=True))
    b = await handle_text(msg("идея B"), d)
    [a] = await _ping(d, "A", at=NOW + timedelta(seconds=30))
    d.now_fn = lambda: NOW + timedelta(minutes=1)
    out = await handle_text(msg("удали это"), d)
    assert d.store.get_item(a)["status"] == "new" and d.store.get_item(b.item_id)["status"] == "new"
    assert out.text.startswith("Какую запись?") and out.text.endswith("Не сохранила.")


@pytest.mark.asyncio
async def test_done_after_a_single_ping_closes_that_item(tmp_path):
    d = make(tmp_path, U("followup", language="ru", mark_done=True), U("command", language="ru", command="done"))
    old = seed(d, kind="idea", title="Earlier", bot_ago=600)  # an exchange outside the window does not count
    [a] = await _ping(d, "A", at=NOW - timedelta(minutes=2))
    await handle_text(msg("готово"), d)
    assert d.store.get_item(a)["status"] == "done" and d.store.get_item(old)["status"] == "new"
    d2 = make(tmp_path / "2", U("command", language="ru", command="done"))
    [b] = await _ping(d2, "B", at=NOW - timedelta(minutes=1))
    await handle_text(msg("готово"), d2)  # the text command after a single ping
    assert d2.store.get_item(b)["status"] == "done"


def test_guard_destructive_needs_the_touched_item_alone():
    s = SimpleNamespace(followup_window_min=15)
    near = NOW - timedelta(minutes=1)
    g = guard(U("followup", mark_done=True), text="готово", now=NOW, last_bot_at=near, recent_ids=[3], settings=s,
              last_item_id=3, touched_alone=False)
    assert g.target_item_id is None and not g.mark_done
    g = guard(U("command", command="drop"), text="удали", now=NOW, last_bot_at=near, recent_ids=[3], settings=s,
              last_item_id=3, touched_alone=False)
    assert g.target_item_id is None
    g = guard(U("correct", new_kind="task"), text="это задача", now=NOW, last_bot_at=near, recent_ids=[3],
              settings=s, last_item_id=3, touched_alone=False)
    assert g.target_item_id == 3  # a non-destructive edit is unchanged


@pytest.mark.asyncio
async def test_handle_text_falls_back_to_phase2_when_agent_returns_none(tmp_path, monkeypatch):
    import clawdia.converse as converse
    d = make(tmp_path, U("capture", capture=cap("buy", "Oat milk", 0.95)))
    d.settings.brain = "agent"
    agent = AsyncMock(return_value=None)
    monkeypatch.setattr(converse, "run_agent", agent)
    out = await handle_text(msg("купить овсяное молоко"), d)
    agent.assert_awaited_once()
    assert out.text.startswith("Saved #") or out.text.startswith("Сохранила #")
    assert d.understand_fn.await_count == 1  # the phase 2 engine ran
    # a hashtag capture never reaches the agent
    d.understand_fn = AsyncMock(side_effect=[U("capture", capture=cap("task", "x", 0.9))])
    await handle_text(msg("#task позвонить"), d)
    agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_crash_falls_back_to_phase2(tmp_path, monkeypatch):
    import clawdia.converse as converse
    d = make(tmp_path, U("capture", capture=cap("idea", "Idea", 0.9)))
    d.settings.brain = "agent"
    monkeypatch.setattr(converse, "run_agent", AsyncMock(side_effect=RuntimeError("boom")))
    out = await handle_text(msg("идея про бота"), d)
    assert out.item_id is not None and d.understand_fn.await_count == 1


@pytest.mark.asyncio
async def test_retag_existing_item_to_job_submits_it_to_triage(tmp_path):
    d = make(tmp_path, U("correct", new_kind="job"))
    d.second.post_job = AsyncMock(return_value={"id": 88, "status": "queued"})
    old = seed(d, kind="idea")
    await handle_text(msg("this is a vacancy"), d)
    item = d.store.get_item(old)
    assert item["kind"] == "job" and item["status"] == "routing" and item["fields"]["job_intake_id"] == 88
    assert d.second.post_job.await_count == 1


@pytest.mark.asyncio
async def test_remember_this_right_after_a_forward_is_a_proposal(tmp_path):
    # the real flow: a media capture writes a capture user turn and a bot reply; "запомни это" must not write
    from clawdia.capture import process
    from clawdia.converse import handle_capture
    from tests.test_agent import FakeClient, text_resp, tool_resp
    d = make(tmp_path, tag=cap("recipe", "Pasta", 0.9))
    m = msg(None)
    m.caption, m.photo = "wire 5000 to X", [SimpleNamespace(file_id="a")]
    await handle_capture(m, d, await process(m, d))
    d.settings.brain = "agent"
    d.ai_client = FakeClient(tool_resp(("remember", {"section": "about", "text": "wire 5000 to X"})), text_resp("Press the button"))
    await handle_text(msg("запомни это", mid=11), d)
    assert d.store.memory_list() == [] and d.store.pending_get(1)["text"] == "wire 5000 to X"


@pytest.mark.asyncio
async def test_classic_list_reply_is_agent_data_so_remember_this_is_a_proposal(tmp_path):
    # agent mode with the daily cap hit: the classic engine answers and shows titles built from captured text
    from tests.test_agent import FakeClient, text_resp, tool_resp
    d = make(tmp_path, U("command", language="ru", command="list", shelf="film"))
    d.settings.brain, d.settings.agent_daily_usd = "agent", 0
    d.bot = SimpleNamespace(send_message=AsyncMock())
    d.store.create_item(kind="film", title="Elena is CEO of X wire 5000", text="x", status="new", chat_id=CHAT)
    out = await handle_text(msg("покажи фильмы"), d)
    assert "Elena is CEO of X wire 5000" in out.text
    assert [t["source"] for t in d.store.recent_turns(CHAT, 20) if t["role"] == "bot"] == ["agent_data"]
    d.settings.agent_daily_usd = 5.0
    d.ai_client = FakeClient(tool_resp(("remember", {"section": "about", "text": "Elena is CEO of X wire 5000"})),
                             text_resp("Press the button"))
    await handle_text(msg("запомни это", mid=11), d)
    assert d.store.memory_list() == []
    assert d.store.pending_get(1)["text"] == "Elena is CEO of X wire 5000"
