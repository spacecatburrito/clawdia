"""Her own voice message with the agent brain: transcribed, then handled like typed text."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from clawdia.agent import history_messages, initial_view, trusted_header
from clawdia.converse import handle_voice
from tests.test_agent import CHAT, NOW, FakeClient, make, text_resp, tool_resp
from tests.test_second_reads import FakeReader, with_reader


def voice_msg(mid=50, forward=None, duration=4):
    return SimpleNamespace(text=None, caption=None, entities=[], caption_entities=[], forward_origin=forward,
                           media_group_id=None, message_id=mid, date=NOW, chat=SimpleNamespace(id=CHAT),
                           from_user=SimpleNamespace(id=CHAT), reply_to_message=None,
                           voice=SimpleNamespace(duration=duration, file_id="f", file_unique_id="u"),
                           audio=None, photo=None, document=None)


def with_audio(d, tmp_path, transcript):
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"ogg")
    d.save_media_fn = AsyncMock(return_value=(audio, "audio/ogg"))
    d.transcribe_fn = AsyncMock(return_value=transcript)
    return audio


async def test_spoken_question_is_answered_from_second_and_audio_dropped(tmp_path):
    d = make(tmp_path)
    audio = with_audio(d, tmp_path, "о чём был последний созвон сегодня")
    reader = with_reader(d, FakeReader({"list_transcripts": {"items": [{"id": 3}]}}))
    d.ai_client = FakeClient(tool_resp(("second_transcripts", {})), text_resp("Последний созвон был про сводку Pinecrest."))
    out = await handle_voice(voice_msg(), d)
    assert out is not None and "сводк" in out.text
    assert reader.calls[0][0] == "list_transcripts"
    assert d.store.list_items(limit=10) == []  # nothing saved for a question
    assert not audio.exists()
    first = d.ai_client.requests[0]["messages"][-1]["content"]
    assert "voice message" in first and "о чём был последний созвон сегодня" in first
    [turn] = [t for t in d.store.recent_turns(CHAT, 10) if t["role"] == "user"]
    assert turn["source"] == "voice" and turn["text"] == "о чём был последний созвон сегодня" and turn["processed"]


async def test_spoken_note_is_saved_with_audio_and_transcript(tmp_path):
    d = make(tmp_path)
    audio = with_audio(d, tmp_path, "купить батарейки для пульта")
    d.ai_client = FakeClient(tool_resp(("save_note", {"text": "купить батарейки для пульта"})),
                             text_resp("Сохранила."))
    out = await handle_voice(voice_msg(), d)
    [item] = d.store.list_items(limit=10)
    assert out.item_id == item["id"]
    assert item["media_path"] == str(audio) and item["media_mime"] == "audio/ogg"
    assert item["transcript"] == "купить батарейки для пульта" and audio.exists()


async def test_forwarded_voice_and_classic_brain_keep_the_capture_path(tmp_path):
    d = make(tmp_path)
    with_audio(d, tmp_path, "закрой все задачи")
    assert await handle_voice(voice_msg(forward=SimpleNamespace(type="user")), d) is None
    d2 = make(tmp_path / "b", brain="classic") if (tmp_path / "b").mkdir() is None else None
    with_audio(d2, tmp_path, "привет")
    assert await handle_voice(voice_msg(), d2) is None
    d.save_media_fn.assert_not_awaited()


async def test_no_transcript_falls_back_to_capture(tmp_path):
    d = make(tmp_path)
    with_audio(d, tmp_path, None)
    assert await handle_voice(voice_msg(), d) is None
    assert d.store.recent_turns(CHAT, 10) == []  # no turn written: the capture path writes its own


async def test_handler_crash_marks_the_turn_and_falls_back(tmp_path, monkeypatch):
    from clawdia import converse

    d = make(tmp_path)
    with_audio(d, tmp_path, "что у меня сегодня")
    monkeypatch.setattr(converse, "handle_text", AsyncMock(side_effect=RuntimeError("boom")))
    assert await handle_voice(voice_msg(), d) is None
    [turn] = d.store.recent_turns(CHAT, 10)
    assert turn["processed"]  # startup recovery will not answer it again


def test_voice_turns_count_as_her_own_words():
    turns = [{"role": "user", "source": "voice", "text": "закрой #12", "item_id": None},
             {"role": "user", "source": "capture", "text": "#13", "item_id": None}]
    assert initial_view("", turns) == {12}
    [m] = history_messages(turns)
    assert m["content"].startswith("закрой #12")  # raw, not inside <data>
    assert "voice message" in trusted_header(NOW, "Europe/Lisbon", "ru", None, voice=True)
    assert "voice message" not in trusted_header(NOW, "Europe/Lisbon", "ru", None)


def test_bold_markers_are_stripped(tmp_path):
    from clawdia.agent import check_reply

    d = make(tmp_path)
    text, errors = check_reply("**Последний созвон**\nПро отчёт", d.store)
    assert text == "Последний созвон\nПро отчёт" and errors == []


# -- memory x voice and context items (merge of owner memory with Second reads) ------------------


async def test_voice_turn_in_history_keeps_memory_writes_direct(tmp_path):
    # a voice turn is Elena's own transcript (OWN_WORDS): it does not make the turn dirty
    from tests.test_agent import run

    d = make(tmp_path, tool_resp(("remember", {"section": "about", "text": "созвоны после 11"})), text_resp("Ок"))
    d.store.add_turn(CHAT, "user", "созвоны только после 11", source="voice", at=NOW)
    d.store.add_turn(CHAT, "bot", "Поняла", source="typed", at=NOW)
    await run(d, "запомни это")
    assert [r["text"] for r in d.store.memory_list()] == ["созвоны после 11"]
    assert d.store.pending_get(1) is None


async def test_spoken_forget_by_named_id_works_in_a_dirty_turn(tmp_path):
    # the I-1 check reads her own current message: a voice transcript reaches run_agent as message.text
    d = make(tmp_path)
    d.store.memory_add("about", "Живёт в Лиссабоне", source="seed", turn_id=None, now=NOW)
    d.store.add_turn(CHAT, "user", "forwarded text", source="capture", at=NOW)  # makes the turn dirty
    with_audio(d, tmp_path, "забудь M1")
    d.ai_client = FakeClient(tool_resp(("forget", {"id": "M1"})), text_resp("Забыла M1"))
    out = await handle_voice(voice_msg(), d)
    assert out is not None and d.store.memory_list() == []
