"""Relay Second's ops problems to the primary owner, once per problem.

`check_ops` polls GET /v1/health/ops (one check over the whole chain). A new problem code
sends one message with the problem and its fix command; a code that disappears sends one
"fixed" line. Outside work time (nights, quiet days) messages wait in meta `ops_pending` and
go out merged at the first check after 07:00 of the next work day.

State lives in one meta JSON blob `ops_state`, written once per check:
`known` {code: first_seen_iso}, `texts` {code: {message, fix}} (server text kept for the
"fixed" line), `pending` (events waiting for a send window), `clean` {code: consecutive clean
checks} (flap damping: "fixed" only after 2), `first_failure` {at, code} (a GET that keeps
failing). A failed GET leaves the real problems untouched; after 20 minutes it raises a
synthetic problem (`second_api_unreachable` / `second_api_rejects_key`) through the same
machinery, unless the stuck warning state in loops.py is already set. The old separate keys
(`ops_known`, `ops_texts`, `ops_pending`) are read once when `ops_state` is absent.
Fix commands are shown verbatim, never translated.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from clawdia.capture import Deps
from clawdia.model_health import current as model_current
from clawdia.reminders import record_bot_turn
from clawdia.second_client import SecondAuthFailed
from clawdia.storage import utc_iso

log = logging.getLogger(__name__)

# code -> short Russian lead (no numbers: the server message carries them in English)
LEADS_RU = {
    "ingest_failed": "утренний импорт (таймер в 08:00) упал",
    "ingest_stale": "импорт сообщений давно не проходил",
    "ingest_incomplete": "импорт идёт, но часть источников не читается",
    "jobs_stuck": "вакансии зависли в очереди разбора",
    "jobs_failed": "разбор вакансий падает с ошибкой",
    "clawdia_heartbeat_stale": "Second не слышит Клавдию",
    "bundle_missing": "набор правил для разбора вакансий не загружен или повреждён",
    "bundle_stale": "набор правил для разбора вакансий давно не обновляли",
    "catalog_stale": "каталог CV давно не обновляли",
    "db_unreachable": "база данных недоступна",
    "second_api_unreachable": "Second API не отвечает",
    "second_api_rejects_key": "Second не принимает ключ Клавдии",
    "model_no_credit": "у Клавдии кончились деньги на счёте Anthropic: она только сохраняет, не понимает",
    "model_key_refused": "Anthropic не принимает ключ Клавдии: она только сохраняет, не понимает",
}

UNREACHABLE = "second_api_unreachable"
REJECTS_KEY = "second_api_rejects_key"
SYNTHETIC = {
    UNREACHABLE: ("Second API is not responding", "check that the Second API is running and reachable at CLAWDIA_SECOND_API_URL"),
    REJECTS_KEY: ("Second API is not accepting Clawdia's key", "check the Second API logs and CLAWDIA_SECOND_API_TOKEN"),
}
# raised from Clawdia's own model calls (clawdia.model_health), not from Second's health check
MODEL_CODES = {
    "credits": ("model_no_credit", "Clawdia's Anthropic account is out of credit",
                "console.anthropic.com - Plans & Billing (the account of CLAWDIA_ANTHROPIC_API_KEY)"),
    "auth": ("model_key_refused", "Anthropic refuses Clawdia's API key",
             "put a valid CLAWDIA_ANTHROPIC_API_KEY into the service env file and restart clawdia"),
}
_MODEL_CODE_SET = {code for code, _, _ in MODEL_CODES.values()}
FAILURE_AFTER = timedelta(minutes=20)
CLEAN_CHECKS = 2  # consecutive clean checks before a "fixed" line
NIGHT_FROM_HOUR = 22  # 22:00-07:00 local is night
MAX_MESSAGE = 3500
MAX_LINE = 1000
STATE_KEY = "ops_state"


def _load(raw: str | None, kind: type) -> Any:
    try:
        value = json.loads(raw) if raw else kind()
    except ValueError:
        return kind()
    return value if isinstance(value, kind) else kind()


def load_state(store) -> dict:
    """The ops state blob; migrates the three old meta keys once when the blob is absent."""
    state = _load(store.meta_get(STATE_KEY), dict)
    if not state:
        state = {"known": _load(store.meta_get("ops_known"), dict), "texts": _load(store.meta_get("ops_texts"), dict),
                 "pending": _load(store.meta_get("ops_pending"), list)}
    for key, kind in (("known", dict), ("texts", dict), ("pending", list), ("clean", dict)):
        if not isinstance(state.get(key), kind):
            state[key] = kind()
    if not isinstance(state.get("first_failure"), dict):
        state["first_failure"] = None
    return state


def _lead(code: str, message: str, lang: str) -> str:
    if lang != "ru":
        return message
    if code.startswith("unit_down:"):
        return f"сервис {code.split(':', 1)[1]} остановлен"
    return LEADS_RU.get(code, message)


def _line(event: dict, lang: str) -> str:
    lead = _lead(event["code"], event["message"], lang)
    if event["kind"] == "fixed":
        text = ("Починилось: " if lang == "ru" else "Fixed: ") + lead
    else:
        fix = event.get("fix")
        text = f"{'Clawdia' if event['code'] in _MODEL_CODE_SET else 'Second'}: {lead}."
        if fix:
            text += f" Починка: `{fix}`" if lang == "ru" else f" Fix: `{fix}`"
    return text[:MAX_LINE]


def render(events: list[dict], lang: str) -> str:
    """Merged message, capped at MAX_MESSAGE characters; the tail becomes a "+N more" line."""
    lines = [_line(e, lang) for e in events]
    text = "\n\n".join(lines)
    if len(text) <= MAX_MESSAGE:
        return text
    kept: list[str] = []
    size = 0
    for i, line in enumerate(lines):
        more = f"\n\n+{len(lines) - i} more" if lang != "ru" else f"\n\nи ещё {len(lines) - i}"
        if size + len(line) + 2 + len(more) > MAX_MESSAGE:
            return "\n\n".join(kept) + more
        kept.append(line)
        size += len(line) + 2
    return text  # unreachable


def _problems(payload: Any) -> list[dict] | None:
    """Valid problems from the response, or None when the payload is not the contract."""
    if not isinstance(payload, dict) or not isinstance(payload.get("problems"), list):
        return None
    out = []
    for p in payload["problems"]:
        if isinstance(p, dict) and isinstance(p.get("code"), str) and p["code"]:
            out.append({"code": p["code"], "message": str(p.get("message") or p["code"]),
                        "fix": str(p.get("fix") or "")})
    return out


def _parse(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _quiet(deps: Deps, now: datetime) -> bool:
    """No messages: quiet day, before 07:00, or night (22:00 and later)."""
    return now < deps.calendar.next_flush_at(now) or now.astimezone(deps.calendar.tz).hour >= NIGHT_FROM_HOUR


def _raise(state: dict, now: datetime, code: str, message: str, fix: str) -> None:
    state["known"][code] = utc_iso(now)
    state["texts"][code] = {"message": message, "fix": fix}
    state["pending"].append({"kind": "new", "code": code, "message": message, "fix": fix})


def _clear(state: dict, code: str) -> None:
    old = state["texts"].get(code) or {}
    state["known"].pop(code, None)
    state["texts"].pop(code, None)
    state["clean"].pop(code, None)
    waiting = [e for e in state["pending"] if e.get("kind") == "new" and e.get("code") == code]
    if waiting:  # raised and cleared while the message was still waiting: nobody needs either
        state["pending"] = [e for e in state["pending"] if e not in waiting]
    else:
        state["pending"].append({"kind": "fixed", "code": code, "message": old.get("message") or code})


def _on_failure(deps: Deps, state: dict, now: datetime, exc: Exception) -> None:
    code = REJECTS_KEY if isinstance(exc, SecondAuthFailed) else UNREACHABLE
    first = state["first_failure"]
    since = _parse(first.get("at")) if first else None
    if since is None:
        state["first_failure"] = {"at": utc_iso(now), "code": code}
        return
    state["first_failure"]["code"] = code
    if now - since < FAILURE_AFTER or code in state["known"]:
        return
    if _parse(deps.store.meta_get("second_down_since")) is not None:
        return  # the stuck warning already reports it: no double report
    _raise(state, now, code, *SYNTHETIC[code])


def _on_success(state: dict, now: datetime, problems: list[dict]) -> None:
    state["first_failure"] = None
    current = {p["code"]: p for p in problems}
    for code, p in current.items():
        state["clean"].pop(code, None)
        if code not in state["known"]:
            _raise(state, now, code, p["message"], p["fix"])
    for code in [c for c in state["known"] if c not in current and c not in _MODEL_CODE_SET]:
        if code not in SYNTHETIC:  # flap damping: real problems need consecutive clean checks
            state["clean"][code] = state["clean"].get(code, 0) + 1
            if state["clean"][code] < CLEAN_CHECKS:
                continue
        _clear(state, code)


def _sync_model(state: dict, now: datetime, down: dict | None) -> None:
    """One alert when the model becomes unusable, one "fixed" line when a call works again."""
    want = MODEL_CODES[down["reason"]][0] if down else None
    for reason, (code, message, fix) in MODEL_CODES.items():
        if code == want and code not in state["known"]:
            _raise(state, now, code, message, fix)
        elif code != want and code in state["known"]:
            _clear(state, code)


async def check_ops(deps: Deps, now: datetime) -> None:
    from clawdia.converse import chat_language  # lazy: converse imports loops-level modules

    store = deps.store
    state = load_state(store)
    try:
        payload = await deps.second.get_ops_health()
    except Exception as exc:  # noqa: BLE001 - unreachable or rejected
        log.info("ops health not read: %s", type(exc).__name__)
        _on_failure(deps, state, now, exc)
    else:
        problems = _problems(payload)
        if problems is None:
            log.info("ops health: malformed payload")
            return
        _on_success(state, now, problems)

    _sync_model(state, now, model_current(store))

    if state["pending"] and not _quiet(deps, now):
        lang = "ru" if chat_language(deps, deps.settings.owner_id) == "ru" else "en"
        text = render(state["pending"], lang)
        try:
            await deps.bot.send_message(deps.settings.owner_id, text=text)  # primary owner only
        except Exception as exc:  # noqa: BLE001 - kept pending, retried on the next check
            log.warning("ops message not sent: %s", type(exc).__name__)
        else:
            state["pending"] = []
            record_bot_turn(store, deps.settings.owner_id, text, None, now)
    store.meta_set(STATE_KEY, json.dumps(state))
