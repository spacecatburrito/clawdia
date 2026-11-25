"""The model is out of credit or the key is refused: Elena is told instead of getting silent plain saving."""

import json
from datetime import timedelta

import pytest

from clawdia import model_health
from clawdia.converse import handle_text
from clawdia.ops_alerts import check_ops
from tests.test_agent import NOW, make, msg, text_resp
from tests.test_loops import MON_9, mk, sent_texts
from tests.test_ops_alerts import ops


class BadRequestError(Exception):
    pass


class AuthenticationError(Exception):
    pass


CREDIT = BadRequestError("Error code: 400 - {'error': {'message': 'Your credit balance is too low to access the "
                         "Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}}")


def test_classify():
    assert model_health.classify(CREDIT) == "credits"
    assert model_health.classify(AuthenticationError("bad key")) == "auth"
    assert model_health.classify(TimeoutError()) is None
    assert model_health.classify(BadRequestError("tools: invalid schema")) is None


def test_notice_once_an_hour_and_cleared_by_success(tmp_path):
    d = make(tmp_path)
    store = d.store
    assert model_health.notice(store, "ru", NOW) is None
    model_health.note_failure(store, CREDIT, NOW)
    first = model_health.notice(store, "ru", NOW)
    assert first and "кончились деньги" in first and chr(0x2014) not in first
    assert model_health.notice(store, "ru", NOW + timedelta(minutes=30)) is None
    assert model_health.notice(store, "en", NOW + timedelta(minutes=61)).startswith("⚠️ The model is unavailable")
    model_health.note_failure(store, TimeoutError(), NOW)  # a transient failure keeps the state as is
    assert model_health.current(store)["reason"] == "credits"
    model_health.note_success(store)
    assert model_health.current(store) is None
    assert model_health.notice(store, "ru", NOW + timedelta(hours=3)) is None


@pytest.mark.asyncio
async def test_agent_credit_failure_puts_the_notice_on_the_fallback_reply(tmp_path):
    d = make(tmp_path, CREDIT)
    d.understand_fn = None  # replaced below: the classic engine runs without a model too

    async def understand_down(text, **kw):
        from clawdia.understand import _error_understanding
        return _error_understanding(text)

    d.understand_fn = understand_down
    out = await handle_text(msg("Напомни 25 ноября в 14:00 про звонок"), d)
    assert model_health.current(d.store)["reason"] == "credits"
    assert out.text.startswith("⚠️ Модель сейчас недоступна")
    out2 = await handle_text(msg("За полчаса", mid=11), d)  # the next one within the hour: no repeat
    assert "Модель сейчас недоступна" not in out2.text


@pytest.mark.asyncio
async def test_working_agent_call_clears_the_state(tmp_path):
    d = make(tmp_path, text_resp("Готово."))
    model_health.note_failure(d.store, CREDIT, NOW - timedelta(hours=2))
    out = await handle_text(msg("привет"), d)
    assert out.text == "Готово."
    assert model_health.current(d.store) is None


@pytest.mark.asyncio
async def test_ops_alert_raised_once_and_fixed_once(tmp_path):
    d = mk(tmp_path)
    ops(d)
    model_health.note_failure(d.store, CREDIT, MON_9)
    await check_ops(d, MON_9)
    await check_ops(d, MON_9 + timedelta(minutes=15))
    texts = sent_texts(d)
    assert len(texts) == 1 and texts[0].startswith("Clawdia: у Клавдии кончились деньги")
    model_health.note_success(d.store)
    await check_ops(d, MON_9 + timedelta(minutes=30))
    assert sent_texts(d)[-1].startswith("Починилось: у Клавдии кончились деньги")
    assert "model_no_credit" not in json.loads(d.store.meta_get("ops_state"))["known"]
