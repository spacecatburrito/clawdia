"""Whether the model is usable, as last seen by a real call.

A model call that fails because the Anthropic account has no credit or the key is refused
cannot recover by retrying, and every message then falls back to plain saving. Elena must
know: the fallback reply carries a notice (at most once per NOTICE_EVERY) and ops_alerts
raises one alert. Any successful agent call clears the state.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from clawdia.storage import utc_iso

KEY = "model_down"
TOLD_KEY = "model_down_told_at"
NOTICE_EVERY = timedelta(hours=1)
REASONS = ("credits", "auth")

NOTICE = {
    "ru": {
        "credits": "⚠️ Модель сейчас недоступна: на счёте Anthropic у Клавдии кончились деньги. Пока я только "
                   "сохраняю сообщения и не понимаю их. Пополни баланс: console.anthropic.com - Plans & Billing.",
        "auth": "⚠️ Модель сейчас недоступна: Anthropic не принимает ключ Клавдии. Пока я только сохраняю сообщения.",
    },
    "en": {
        "credits": "⚠️ The model is unavailable: Clawdia's Anthropic account is out of credit. Until it is topped up "
                   "I only save messages and do not understand them. Top up at console.anthropic.com - Plans & Billing.",
        "auth": "⚠️ The model is unavailable: Anthropic refuses Clawdia's key. Until it is fixed I only save messages.",
    },
}


def classify(exc: BaseException) -> str | None:
    """'credits' or 'auth' for failures that retrying cannot fix, else None."""
    if "credit balance" in str(exc).lower():
        return "credits"
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
        return "auth"
    return None


def current(store: Any) -> dict | None:
    raw = store.meta_get(KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) and data.get("reason") in REASONS else None


def note_failure(store: Any, exc: BaseException, now: datetime) -> None:
    reason = classify(exc)
    if reason is None:
        return
    old = current(store)
    if old and old["reason"] == reason:
        return
    store.meta_set(KEY, json.dumps({"reason": reason, "since": utc_iso(now)}))


def note_success(store: Any) -> None:
    if store.meta_get(KEY):
        store.meta_set(KEY, "")
        store.meta_set(TOLD_KEY, "")


def notice(store: Any, lang: str, now: datetime) -> str | None:
    """The notice line for a fallback reply, at most once per NOTICE_EVERY while the model is down."""
    down = current(store)
    if down is None:
        return None
    told = store.meta_get(TOLD_KEY)
    if told:
        try:
            if now - datetime.fromisoformat(told) < NOTICE_EVERY:
                return None
        except ValueError:
            pass
    store.meta_set(TOLD_KEY, utc_iso(now))
    return NOTICE["ru" if lang == "ru" else "en"][down["reason"]]
