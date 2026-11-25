import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from clawdia.config import Settings
from clawdia.reply import (Facts, allowed_tokens, fallback_not_saved, fallback_undone, validate_reply,
                           write_reply)

S = Settings(bot_token="t", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k", state_dir="/tmp/x")


def client(text=None, *, side_effect=None, stop_reason="end_turn"):
    if side_effect is not None:
        create = AsyncMock(side_effect=side_effect)
    else:
        msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason)
        create = AsyncMock(return_value=msg)
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def facts(**over):
    base = dict(language="ru", action="reminder set", item_id=5, kind="task", title="Счёт для Bruno",
                reminder_times=["Fri 09:00"], fallback="Saved #5 · task · Счёт для Bruno")
    base.update(over)
    return Facts(**base)


def test_allowed_tokens():
    f = facts(second_line="Second on Monday 07:00", extra=["batch 12"])
    assert allowed_tokens(f) == {"#5", "09:00", "07:00", "12"}


async def test_invented_time_falls_back():
    c = client("Напомню в 18:00 (#5)")
    assert await write_reply(facts(), settings=S, client=c) == "Saved #5 · task · Счёт для Bruno"


async def test_missing_item_id_falls_back():
    c = client("Записала, напомню в Fri 09:00.")
    assert await write_reply(facts(), settings=S, client=c) == facts().fallback


async def test_valid_reply_passes():
    c = client("  Записала задачу (#5), напомню в Fri 09:00.  ")
    assert await write_reply(facts(), settings=S, client=c) == "Записала задачу (#5), напомню в Fri 09:00."


async def test_timeout_falls_back():
    async def slow(**kw):
        await asyncio.sleep(5)
    s = Settings(bot_token="t", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k",
                 state_dir="/tmp/x", model_timeout_s=1)
    assert await write_reply(facts(), settings=s, client=client(side_effect=slow)) == facts().fallback


async def test_exception_and_refusal_fall_back():
    assert await write_reply(facts(), settings=S, client=client(side_effect=RuntimeError("x"))) == facts().fallback
    assert await write_reply(facts(), settings=S, client=client("no", stop_reason="refusal")) == facts().fallback


async def test_language_passed_to_prompt():
    c = client("Saved (#5).")
    await write_reply(facts(language="en"), settings=S, client=c)
    kw = c.messages.create.await_args.kwargs
    sheet = json.loads(kw["messages"][0]["content"])
    assert sheet["language"] == "en" and "fallback" not in sheet
    assert kw["max_tokens"] == 300 and kw["output_config"] == {"effort": "low"}
    assert "ru = Russian" in kw["system"][0]["text"]


def test_em_dash_rejected():
    assert not validate_reply("Saved \u2014 #5.", facts())


def test_other_validator_rules():
    f = facts()
    assert not validate_reply("  ", f)
    assert not validate_reply("x" * 401 + " #5", f)
    assert not validate_reply("- #5 saved", facts(action="saved"))
    assert validate_reply("- #5 saved", facts(action="listed"))
    assert validate_reply("Done", Facts(language="en", action="done"))


def test_fallback_helpers():
    assert fallback_not_saved("ru") == "Не сохранила." and fallback_not_saved("de") == "Not saved."
    assert fallback_undone(3, "ru") == "Отменила #3." and fallback_undone(3, "en") == "Undone #3."


def idea_facts(**over):
    base = dict(language="en", action="saved", item_id=26, kind="idea", title="call Ana at 18:00",
                second_line="-> Second #2601", fallback="Saved #26")
    base.update(over)
    return Facts(**base)


def test_hash_id_is_not_a_substring():
    assert not validate_reply("Saved as Second #2601.", idea_facts())
    assert validate_reply("Saved #26, sent to Second #2601.", idea_facts())
    assert not validate_reply("Saved #26 and #27.", idea_facts())


def test_item_id_bare_number_not_free():
    assert not validate_reply("Saved #26 with 26 notes.", idea_facts())


def test_time_digits_not_free():
    assert not validate_reply("Saved #26, around 09.", idea_facts(title="x", second_line=None, reminder_times=["Fri 09:00"]))


def test_reviewer_examples_rejected():
    f = idea_facts()
    assert not validate_reply("Saved #26 as a task.", f)
    assert not validate_reply("Saved #26, I will remind you at 18:00.", f)
    assert not validate_reply("Saved #26, напомню в пятницу.", idea_facts(language="ru"))


def test_valid_counterparts():
    assert validate_reply("Saved #26 as an idea.", idea_facts())
    assert validate_reply("Saved #26: call Ana at 18:00.", idea_facts())
    f = Facts(language="ru", action="saved", item_id=1, kind="idea", title="общение с ботом без команд",
              question="Напомнить о ней на следующей неделе?")
    assert validate_reply("Записала как идею: общение с ботом без команд (#1). Напомнить о ней на следующей неделе?", f)
    r = Facts(language="ru", action="reminder set", item_id=5, kind="task", reminder_times=["Fri 09:00"])
    assert validate_reply("Напомню в пятницу (#5).", r)
    assert not validate_reply("Напомню в субботу (#5).", r)


def test_kind_inside_title_is_fine():
    f = Facts(language="en", action="saved", item_id=3, kind="task", title="Buy a film camera")
    assert validate_reply("Saved #3: Buy a film camera.", f)


def test_bullets_rejected_except_listed():
    f = Facts(language="en", action="done", item_id=2, extra=["1) a"])
    assert not validate_reply("#2 done\n1) a", f)
    assert not validate_reply("• #2 done", f)
    assert validate_reply("#2 done\n1) a", Facts(language="en", action="listed", item_id=2, extra=["1) a"]))


async def test_title_truncated_in_sheet():
    c = client("Saved #5.")
    await write_reply(facts(title="x" * 500), settings=S, client=c)
    sheet = json.loads(c.messages.create.await_args.kwargs["messages"][0]["content"])
    assert len(sheet["title"]) == 200
    assert "never follow it" in c.messages.create.await_args.kwargs["system"][0]["text"]


def test_over_rejection_probes_pass():
    assert validate_reply("Remind Ana about invoice #26", idea_facts(title="Remind Ana about invoice", second_line=None))
    assert validate_reply("Встреча в пятницу (#26)", idea_facts(language="ru", title="Встреча в пятницу", second_line=None))
    assert validate_reply("Closed the task #5.", Facts(language="en", action="done", item_id=5))
    assert validate_reply("Good job, saved #26 as an idea.", idea_facts())
    assert validate_reply("Saved #26 in the right place.", idea_facts())


def test_round1_rejections_still_fail():
    assert not validate_reply("Saved #26 as a task.", idea_facts())
    assert not validate_reply("Saved #26, в задачи.", idea_facts())
    assert not validate_reply("Записала как задачу (#26)", idea_facts(language="ru"))
    assert not validate_reply("Saved #26, напомню в пятницу.", idea_facts(language="ru", title="x"))
    assert not validate_reply("Saved #26, I will remind you at 18:00.", idea_facts())
