"""Conversation engine (phase 2): one typed message in, a saved or changed item or a command
result out, plus a natural reply.

Only plain text reaches `handle_text`. Forwarded messages (anything with `forward_origin`),
photos, documents and audio are routed by the bot to the phase 1 capture path and never yield
intents; slash commands keep their own handlers and make no model call. Exception: with the agent
brain, Elena's own voice message (not forwarded) is transcribed and handled like typed text
(`handle_voice`), so a spoken question gets an answer; it falls back to the capture path when it
cannot be transcribed.

Order (spec 6.3): the user turn is stored unprocessed before any model call, then one
understanding call, the code guards (`guard`, spec 6.4a), one executor and one reply call; at
the end the turn is marked processed, the bot turn is stored and old turns are pruned. The
executors live in `converse_actions`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from clawdia.agent import run_agent, writes_outcome
from clawdia.capture import Captured, Deps, _add_media, _duration, _step, apply_tag, build_reply, keyboard_for
from clawdia.context import Snapshot
from clawdia.converse_actions import (EXECUTORS, Ctx, Outcome, _question, model_client, saved_outcome,
                                      unsorted_result)
from clawdia.keyboards import markup
from clawdia.kinds import kind_from_hashtags
from clawdia.model_health import notice as model_notice
from clawdia.reminders import TURNS_KEPT
from clawdia.render import MAX_ID, find_item
from clawdia.tagger import TITLE_MAX
from clawdia.understand import Understanding, guess_language

__all__ = ["Outcome", "chat_language", "guard", "handle_capture", "handle_text", "recover_unprocessed",
           "save_late_turns"]

log = logging.getLogger(__name__)

COMMAND_MAX_WORDS = 8
CHAT_MAX_WORDS = 25
TURNS_TO_MODEL = 8
RECENT_ITEMS = 10
RECOVER_WINDOW = timedelta(hours=1)
DESTRUCTIVE_COMMANDS = ("done", "drop")
_URL_RE = re.compile(r"https?://|\bwww\.\S|\bt\.me/\S", re.IGNORECASE)
_HASH_ID_RE = re.compile(r"(?<![\w#])#(\d+)(?!\w)", re.ASCII)
_BARE_NUMBER_RE = re.compile(r"(?<!\S)\d+(?!\S)", re.ASCII)


# -- guards (spec 6.4a) ---------------------------------------------------


def _as_capture(u: Understanding, text: str) -> Understanding:
    """A downgraded message is a note. It keeps the model's capture when it gave one; capture None
    tells the capture executor to tag the text (the second model call, deterministic reply)."""
    keep = u.capture is not None
    return Understanding(intent="capture", language=u.language, capture=u.capture,
                         ask=u.ask if keep else None, model=u.model)


def _pasted(text: str, max_words: int = COMMAND_MAX_WORDS) -> bool:
    """Longer than a command or carrying a URL: pasted content rather than an instruction (6.4a)."""
    return len(text.split()) > max_words or _URL_RE.search(text) is not None


def _valid_id(raw: str) -> int | None:
    value = int(raw)
    return value if 0 < value <= MAX_ID else None


def _command_explicit_id(text: str, command: str) -> int | None:
    """A "#N" anywhere; a bare number only for show/done/drop and only right after the verb ("готово 3")."""
    match = _HASH_ID_RE.search(text)
    if match:
        return _valid_id(match.group(1))
    tokens = text.split()
    if command in ("show", *DESTRUCTIVE_COMMANDS) and len(tokens) >= 2 and tokens[1].isascii() and tokens[1].isdigit():
        return _valid_id(tokens[1])
    return None


def _in_window(now: datetime, last_bot_at: datetime | None, settings: Any) -> bool:
    return last_bot_at is not None and now - last_bot_at <= timedelta(minutes=settings.followup_window_min)


def _no_destruction(u: Understanding) -> Understanding:
    return replace(u, delete=False, mark_done=False)


def guard(u: Understanding, *, text: str, now: datetime, last_bot_at: datetime | None, recent_ids: Sequence[int],
          settings: Any, item_exists: Callable[[int], bool] = lambda _id: True,
          last_item_id: int | None = None, touched_alone: bool = True) -> Understanding:
    """Apply the 6.4a guards to what the model understood; returns a new Understanding.

    `recent_ids` are this chat's recent items, most recent first. The last item is
    `last_item_id` (the item the latest bot reply in the window created or touched, see
    `last_touched`) or, without one, `recent_ids[0]`; destructive effects without an explicit
    id (delete, mark_done, done/drop) need `last_item_id` itself and never use that fallback,
    and only when `touched_alone` (no other item was named since Elena's last message, see
    `touched_is_alone`): after three reminder pings "готово" asks which one.
    `item_exists` checks an explicit id against all items. On the result,
    `target_item_id` is the item to act on (also for commands), or None: then `explicit_id`
    set means "No item #N", otherwise the executor asks which item.
    """
    if u.intent == "command":
        if u.command is None or _pasted(text):
            return _as_capture(u, text)
        u = replace(u, explicit_id=_command_explicit_id(text, u.command))
    elif u.intent == "chat" and _pasted(text, CHAT_MAX_WORDS):  # anti-goal 3: when in doubt, save
        return _as_capture(u, text)
    elif u.intent not in ("correct", "followup"):
        return replace(u)
    in_window = _in_window(now, last_bot_at, settings)
    explicit = u.explicit_id
    explicit_ok = explicit is not None and item_exists(explicit)
    touched = last_item_id  # the item a bot reply in the window really named
    sole = touched if touched_alone else None  # the only item a destructive effect may take without an id
    last = touched if touched is not None else (recent_ids[0] if recent_ids else None)

    if u.intent == "command":
        if explicit is not None:
            return replace(u, target_item_id=explicit if explicit_ok else None, target_rejected=False)
        if u.command not in ("show", *DESTRUCTIVE_COMMANDS) or u.target_rejected:
            return replace(u, target_item_id=None)
        # A number we did not accept as an id ("закрой задачу 2") must not fall back to the last item.
        stray = _BARE_NUMBER_RE.search(text) is not None
        if u.command == "show":
            return replace(u, target_item_id=u.item_id or (None if stray else last))
        # done/drop: only the touched item, inside the window only (never the newest-item fallback)
        candidate = None if stray else (u.item_id or sole)
        ok = candidate is not None and candidate == sole and in_window
        return replace(u, target_item_id=candidate if ok else None)

    # correct / followup: outside the window only an existing explicit id in a short typed
    # message keeps the intent; pasted-looking text ("PR #12 ...", a URL) is a note.
    if not in_window and (not explicit_ok or _pasted(text)):
        return _as_capture(u, text)
    if explicit is not None and (u.delete or u.mark_done) and _pasted(text):  # "PR #12 удалить старый endpoint ..."
        return _as_capture(u, text)
    if u.intent == "followup" and u.save_previous:
        return replace(u, target_item_id=None)
    if explicit is not None:
        if explicit_ok:
            return replace(u, target_item_id=explicit, target_rejected=False)
        return _no_destruction(replace(u, target_item_id=None))
    if u.target_rejected:
        return _no_destruction(replace(u, target_item_id=None))
    if last is None:  # nothing to correct in this chat: the message is a note
        return _as_capture(u, text)
    target = u.target_item_id or last
    if (u.delete or u.mark_done) and (sole is None or target != sole):  # destructive: the touched item only
        return _no_destruction(replace(u, target_item_id=None))
    return replace(u, target_item_id=target)


def _hashtag_capture(u: Understanding, kind: str, text: str) -> Understanding:
    """A kind hashtag keeps its phase 1 precedence: always a capture of that kind. Without a model
    capture the tag call runs, and the phase 1 tagger applies the hashtag itself."""
    capture = replace(u.capture, kind=kind, confidence=1.0) if u.capture is not None else None
    return Understanding(intent="capture", language=u.language, capture=capture, model=u.model)


def last_touched(history: list[dict], now: datetime, settings: Any, item_exists: Callable[[int], bool]) -> int | None:
    """The item of the latest bot reply in this chat that named one, if that reply is inside the
    follow-up window and the item still exists: the item the chat created or touched last."""
    for turn in reversed(history):
        if turn["role"] != "bot" or turn.get("item_id") is None:
            continue
        try:
            at = datetime.fromisoformat(turn["at"])
        except (TypeError, ValueError):
            return None
        return turn["item_id"] if _in_window(now, at, settings) and item_exists(turn["item_id"]) else None
    return None


def touched_is_alone(history: list[dict], touched: int | None, now: datetime, settings: Any) -> bool:
    """Whether every bot turn in the follow-up window since Elena's latest user turn names
    `touched`: a reminder ping or drain notice for another item in between makes "this" ambiguous."""
    if touched is None:
        return False
    for turn in reversed(history):
        if turn["role"] == "user":
            break
        if turn["role"] != "bot":
            continue
        try:
            at = datetime.fromisoformat(turn["at"])
        except (TypeError, ValueError):
            return False
        if _in_window(now, at, settings) and turn.get("item_id") != touched:
            return False
    return True


# -- entry point --------------------------------------------------------


async def _rescue(ctx: Ctx) -> Outcome:
    """An executor failed: keep the note. Reuse the item this turn created, else save the text unsorted."""
    if ctx.item_id is None:
        ctx.item_id = ctx.deps.store.create_item(status="new", kind="unsorted", text=ctx.text, chat_id=ctx.chat_id)
        await _step(ctx.deps, ctx.item_id, "tag", apply_tag(ctx.deps, ctx.item_id, unsorted_result(ctx.text), ctx.now))
    return await saved_outcome(ctx, ctx.item_id, use_model=False)


def _finish(deps: Deps, chat_id: int, turn_id: int, outcome: Outcome) -> None:
    store = deps.store
    try:
        store.mark_turn(turn_id, processed=True, item_id=outcome.item_id)
        store.add_turn(chat_id, "bot", outcome.text, item_id=outcome.item_id, at=deps.now_fn(), source=outcome.source)
        store.prune_turns(chat_id, TURNS_KEPT)
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not cost the owner her reply
        log.warning("turn %s bookkeeping failed: %s", turn_id, type(exc).__name__)


def _last_bot_at(history: list[dict]) -> datetime | None:
    """When the latest bot turn before this message was written (a reply written later, e.g. to
    another recovered turn, does not count)."""
    for turn in reversed(history):
        if turn["role"] == "bot":
            try:
                return datetime.fromisoformat(turn["at"])
            except (TypeError, ValueError):
                return None
    return None


async def handle_text(message: Any, deps: Deps, *, turn_id: int | None = None, now: datetime | None = None) -> Outcome:
    """Handle one plain text message (never a forward or media, see the module docstring).

    With `turn_id` None the user turn is written first (processed=0); a given `turn_id`
    (startup recovery, or the bot handler that wrote it) is reused as is. `now` overrides the
    clock for the follow-up window and relative dates: recovery passes the turn's send time.
    """
    store, settings = deps.store, deps.settings
    chat_id = message.chat.id
    text = getattr(message, "text", None) or ""
    now = now or deps.now_fn()
    if turn_id is None:
        turn_id = store.add_turn(chat_id, "user", text, processed=False, at=now)
    hashtag = kind_from_hashtags(text)
    if hashtag is None and getattr(settings, "brain", "classic") == "agent":  # hashtag captures stay deterministic
        try:
            outcome = await run_agent(message, deps, turn_id=turn_id, now=now)
        except Exception as exc:  # noqa: BLE001 - the phase 2 engine below is the fallback
            log.warning("turn %s agent crashed: %s", turn_id, type(exc).__name__)
            outcome = writes_outcome(deps, chat_id, turn_id, text)  # a write was made: no second pass
        if outcome is not None:
            _finish(deps, chat_id, turn_id, outcome)
            return outcome
    history = [t for t in store.recent_turns(chat_id, TURNS_KEPT) if t["id"] < turn_id]
    recent = store.recent_items(chat_id, RECENT_ITEMS)
    snapshot = deps.context_cache.load()
    u = await deps.understand_fn(text, turns=history[-TURNS_TO_MODEL:], recent_items=recent, snapshot=snapshot,
                                 settings=settings, now=now, client=model_client(deps))
    if hashtag is not None:
        u = _hashtag_capture(u, hashtag, text)
    else:
        def live(item_id: int) -> bool:
            item = find_item(store, item_id)
            return item is not None and item["status"] != "dismissed"

        touched = last_touched(history, now, settings, live)
        u = guard(u, text=text, now=now, last_bot_at=_last_bot_at(history),
                  recent_ids=[i["id"] for i in recent], settings=settings,
                  item_exists=lambda i: find_item(store, i) is not None,
                  last_item_id=touched, touched_alone=touched_is_alone(history, touched, now, settings))
    ctx = Ctx(deps=deps, message=message, text=text, chat_id=chat_id, turn_id=turn_id, now=now, u=u,
              recent=recent, history=history, snapshot=snapshot, model_calls=1)  # the understanding call
    try:
        outcome = await EXECUTORS[u.intent](ctx)
    except Exception as exc:  # noqa: BLE001 - a failed action must not lose the note
        log.warning("turn %s intent %s failed: %s", turn_id, u.intent, type(exc).__name__)
        outcome = await _rescue(ctx)
    log.info("turn %s model_calls=%s", turn_id, ctx.model_calls)
    _prepend_model_notice(deps, outcome, u.language, now)
    outcome.source = "agent_data"  # lists, inbox and answers show titles built from captured text, whatever the brain
    _finish(deps, chat_id, turn_id, outcome)
    return outcome


# -- her own voice messages (agent brain) -----------------------------------------


class _Spoken:
    """Her voice message seen as typed text: `text` is the transcript, the rest is the message."""

    def __init__(self, message: Any, text: str) -> None:
        self._message = message
        self.text = text
        self.caption = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._message, name)


def _attach_voice(deps: Deps, turn_id: int, outcome: Outcome, text: str, saved: tuple, since: datetime) -> bool:
    """Give the items this turn created the audio and the transcript; True when the file was used."""
    store = deps.store
    ids = {w["item_id"] for w in store.agent_writes(turn_id) if w.get("tool") == "save_note" and w.get("item_id")}
    if outcome.item_id is not None:
        ids.add(outcome.item_id)
    used = False
    for item_id in sorted(ids):
        item = store.get_item(item_id)
        try:
            created = datetime.fromisoformat(item["created_at"]) if item else None
        except (TypeError, ValueError):
            created = None
        if item is None or created is None or created < since or item.get("media_path"):
            continue  # an existing item the turn only changed keeps what it had
        _add_media(store, item_id, *saved)
        store.update_item(item_id, transcript=text)
        used = True
    return used


async def handle_voice(message: Any, deps: Deps) -> Outcome | None:
    """Elena's own voice message with the agent brain: transcribe it, then handle the transcript as
    her typed words (a question gets an answer, a note is saved with the audio attached). None when
    this path does not apply or the voice could not be transcribed: the caller then saves it through
    the capture path, as before. A forwarded voice is someone else's words and never comes here."""
    if (getattr(deps.settings, "brain", "classic") != "agent" or not getattr(message, "voice", None)
            or getattr(message, "forward_origin", None) is not None):
        return None
    chat_id = message.chat.id
    since = deps.now_fn()
    try:
        saved = await deps.save_media_fn(deps.bot, message, deps.settings.state_dir / "media",
                                         deps.settings.max_document_mb, stem=f"voice-{chat_id}-{message.message_id}")
        text = (await deps.transcribe_fn(saved[0], _duration(message), deps.settings) or "").strip() if saved else ""
    except Exception as exc:  # noqa: BLE001 - the capture path below still saves the voice
        log.warning("voice not transcribed: %s", type(exc).__name__)
        return None
    if not text:
        return None
    turn_id = deps.store.add_turn(chat_id, "user", text, processed=False, at=since, source="voice")
    try:
        outcome = await handle_text(_Spoken(message, text), deps, turn_id=turn_id, now=since)
    except Exception as exc:  # noqa: BLE001 - never lose it: mark the turn and let the capture path save it
        log.warning("voice turn %s not handled: %s", turn_id, type(exc).__name__)
        deps.store.mark_turn(turn_id, processed=True)
        return None
    try:
        if not _attach_voice(deps, turn_id, outcome, text, saved, since):
            saved[0].unlink(missing_ok=True)  # a question: the audio is not kept
    except Exception as exc:  # noqa: BLE001 - the answer is already there
        log.warning("voice turn %s audio not attached: %s", turn_id, type(exc).__name__)
    return outcome


# -- media and forwards (phase 1 capture, phase 2 reply) --------------------------


def chat_language(deps: Deps, chat_id: int) -> str:
    """Language of the latest typed user turn in this chat; "ru" when there is none (Elena
    writes Russian). Used where a message has no words of its own."""
    from clawdia.converse_actions import MEDIA_MARKERS
    try:
        turns = deps.store.recent_turns(chat_id, TURNS_KEPT)
    except Exception as exc:  # noqa: BLE001
        log.warning("chat %s turns not read: %s", chat_id, type(exc).__name__)
        return "ru"
    for turn in reversed(turns):
        if turn["role"] == "user" and (turn.get("text") or "").strip() and turn["text"] not in MEDIA_MARKERS:
            return guess_language(turn["text"])
    return "ru"


def _media_marker(message: Any) -> str:
    if getattr(message, "photo", None):
        return "[photo]"
    if getattr(message, "voice", None) or getattr(message, "audio", None):
        return "[voice]"
    if getattr(message, "document", None):
        return "[document]"
    if any(getattr(message, a, None) for a in ("video", "video_note", "animation", "sticker")):
        return "[media]"
    return "[forward]"


async def handle_capture(message: Any, deps: Deps, captured: Captured, *, turn_id: int | None = None) -> Outcome:
    """Reply to an item the phase 1 pipeline already saved from a forward or media message.

    The tag call was the first model call; the reply call is the second (the deterministic reply
    on any failure). Both turns are stored, so a follow-up ("это задача") can refer to the item.
    A given `turn_id` is Elena's already stored typed comment (a comment glued to a forward): it is
    marked processed with the item instead of writing a second user turn.
    """
    store, item_id = deps.store, captured.item_id
    chat_id = message.chat.id
    now = deps.now_fn()
    item = store.get_item(item_id) or {}
    words = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    spoken = " ".join(t for t in (words, item.get("transcript") or "") if t)
    language = guess_language(spoken) if spoken else chat_language(deps, chat_id)  # before this turn is written
    turn_text = words or _media_marker(message)
    glued = turn_id is not None
    try:
        if turn_id is None:
            turn_id = store.add_turn(chat_id, "user", turn_text, item_id=item_id, processed=True, at=now, source="capture")
        else:  # Elena's own typed comment glued to a forward stays a typed turn
            store.mark_turn(turn_id, processed=True, item_id=item_id)
    except Exception as exc:  # noqa: BLE001 - the item is saved; memory is a convenience
        log.warning("item %s user turn not written: %s", item_id, type(exc).__name__)
    turn_id = turn_id or 0
    ctx = Ctx(deps=deps, message=message, text=turn_text, chat_id=chat_id, turn_id=turn_id, now=now,
              u=Understanding(intent="capture", language=language), recent=[], history=[], snapshot=Snapshot(),
              item_id=item_id, model_calls=1)  # the phase 1 tag call
    try:
        outcome = await saved_outcome(ctx, item_id, _question(ctx, item_id, None, None))
    except Exception as exc:  # noqa: BLE001 - the phase 1 reply is always available
        log.warning("item %s reply failed: %s", item_id, type(exc).__name__)
        try:
            text = build_reply(store.get_item(item_id), deps.calendar, now, language)
        except Exception:  # noqa: BLE001
            text = captured.reply_text
        outcome = Outcome(text, captured.keyboard, item_id)
    log.info("turn %s model_calls=%s", turn_id, ctx.model_calls)
    own = _own_words(message, glued)
    if own and asks_something(own) and turn_id and getattr(deps.settings, "brain", "classic") == "agent":
        outcome = await _answer_with_item(message, deps, own, item_id, turn_id, outcome, language)
    _prepend_model_notice(deps, outcome, language, now)
    try:
        store.add_turn(chat_id, "bot", outcome.text, item_id=item_id, at=deps.now_fn(), source="capture_reply")
        store.prune_turns(chat_id, TURNS_KEPT)
    except Exception as exc:  # noqa: BLE001
        log.warning("item %s bot turn not written: %s", item_id, type(exc).__name__)
    return outcome


_REQUEST_WORDS = re.compile(
    r"(?<!\w)(?:посмотри|глянь|оцени|ответь|скажи|подскажи|проверь|разбери|сравни|объясни|расскажи|переведи|"
    r"что думаешь|как тебе|как считаешь|стоит ли|надо ли|look|check|tell me|answer|review|explain|"
    r"what do you think|should i|is it worth)", re.IGNORECASE)


def asks_something(own: str) -> bool:
    """Her own words hold a question mark or a request word: the agent should answer, not only save."""
    return "?" in own or bool(_REQUEST_WORDS.search(own))


def _prepend_model_notice(deps: Deps, outcome: Outcome, lang: str, now: datetime) -> None:
    """The model is out (no credit, key refused): the plain-saving reply says so, at most once an hour."""
    try:
        line = model_notice(deps.store, lang, now)
    except Exception as exc:  # noqa: BLE001 - the notice must never cost the reply
        log.warning("model notice failed: %s", type(exc).__name__)
        return
    if line:
        outcome.text = f"{line}\n\n{outcome.text}"


def _own_words(message: Any, glued: bool) -> str:
    """Elena's own words that came with a capture: her typed comment glued to a forward, or the caption
    of her own (not forwarded) photo or file. A forward's text or caption is someone else's."""
    if glued:
        return (getattr(message, "text", None) or "").strip()
    if getattr(message, "forward_origin", None) is None:
        return (getattr(message, "caption", None) or "").strip()
    return ""


async def _answer_with_item(message: Any, deps: Deps, own: str, item_id: int, turn_id: int,
                            saved: Outcome, language: str) -> Outcome:
    """Her question about what she just sent ("что на картинке?", "эту вакансию уже смотрели?"): the
    agent answers it with the new item in view, and the reply is just the answer. The capture stays
    quietly; on any failure the capture reply is kept as it was."""
    try:
        answer = await run_agent(_Spoken(message, own), deps, turn_id=turn_id, now=deps.now_fn(),
                                 context_item=item_id)
    except Exception as exc:  # noqa: BLE001 - the item is saved and has its reply already
        log.warning("item %s question not answered: %s", item_id, type(exc).__name__)
        return saved
    if answer is None:
        return saved
    # Only the answer: she asked a question, the save is not news (the item stays, /list shows it). The
    # answer keeps its source and its keyboard (memory proposal buttons or Undo).
    _close_answered(deps, item_id, turn_id)
    return Outcome(answer.text, answer.keyboard, item_id, answer.source)


def _close_answered(deps: Deps, item_id: int, turn_id: int) -> None:
    """An unsorted capture that came with a question and got its answer is done: nothing is left to
    sort, so it leaves the inbox (/list "all" still shows it). It stays open when tagging failed (no
    "tagged" event), when the answer turn wrote to it, or when it still waits on a reminder (an enabled,
    unsent one, or an "ask_when" question). A classified capture (job, recipe, film...) stays open."""
    try:
        item = deps.store.get_item(item_id)
        if item is None or item.get("kind") != "unsorted" or item.get("status") != "new":
            return
        fields = dict(item.get("fields") or {})
        if fields.get("ask_when") or "tagged" not in deps.store.event_types(item_id):
            return
        if any(w.get("item_id") == item_id for w in deps.store.agent_writes(turn_id)):
            return
        if any(not r.get("disabled") and not r.get("sent_at") for r in deps.store.reminders_of(item_id)):
            return
        fields["answered"] = True
        deps.store.update_item(item_id, status="done", fields=fields)
        deps.store.add_event(item_id, "answered", {})
    except Exception as exc:  # noqa: BLE001 - the answer is ready; a stale inbox row is the worst case
        log.warning("item %s not closed after its answer: %s", item_id, type(exc).__name__)


# -- startup recovery (spec 6.3) -------------------------------------------------


def _saved_by(deps: Deps, turn: dict, at: datetime) -> dict | None:
    """The item a turn already created before the process stopped (same chat and text, created
    after the turn), so a recovered turn never makes a second one."""
    for item in deps.store.recent_items(turn["chat_id"], RECENT_ITEMS):
        try:
            created = datetime.fromisoformat(item["created_at"])
        except (TypeError, ValueError):
            continue
        if item.get("text") == turn["text"] and created >= at:
            return item
    return None


async def _recover_turn(deps: Deps, turn: dict) -> Outcome:
    at = datetime.fromisoformat(turn["at"])
    chat_id = turn["chat_id"]
    saved = _saved_by(deps, turn, at)
    if saved is not None:
        reply = build_reply(saved, deps.calendar, deps.now_fn(), guess_language(turn["text"] or ""))
        outcome = Outcome(reply, keyboard_for(saved), saved["id"])
        _finish(deps, chat_id, turn["id"], outcome)
        return outcome
    wrote = writes_outcome(deps, chat_id, turn["id"], turn["text"] or "")
    if wrote is not None:  # the agent already wrote for this turn before the restart: do not run it again
        _finish(deps, chat_id, turn["id"], wrote)
        return wrote
    message = SimpleNamespace(
        text=turn["text"], caption=None, entities=[], caption_entities=[], forward_origin=None, media_group_id=None,
        message_id=None, date=at, chat=SimpleNamespace(id=chat_id, type="private"), from_user=SimpleNamespace(id=chat_id))
    return await handle_text(message, deps, turn_id=turn["id"], now=at)


def _save_late(deps: Deps, turn: dict) -> None:
    """A turn too old to answer is still Elena's note: saved unsorted, no model call, no message."""
    store = deps.store
    text = turn.get("text") or ""
    saved = _saved_by(deps, turn, datetime.fromisoformat(turn["at"])) if text.strip() else None
    item_id = saved["id"] if saved is not None else None
    if item_id is None and text.strip():
        item_id = store.create_item(status="new", kind="unsorted", text=text, title=text[:TITLE_MAX],
                                    chat_id=turn["chat_id"], confidence=0.0)
        store.add_event(item_id, "recovered_late", {"turn_id": turn["id"]})
    store.mark_turn(turn["id"], processed=True, item_id=item_id)


def save_late_turns(deps: Deps) -> int:
    """Unprocessed user turns older than the recovery window become unsorted notes (never
    dropped, never answered: a reply hours later would only confuse). Returns how many."""
    try:
        turns = deps.store.unprocessed_turns(before=deps.now_fn() - RECOVER_WINDOW)
    except Exception as exc:  # noqa: BLE001
        log.warning("late turns not read: %s", type(exc).__name__)
        return 0
    saved = 0
    for turn in turns:
        try:
            _save_late(deps, turn)
        except Exception as exc:  # noqa: BLE001 - one broken turn must not block the others
            log.warning("turn %s not saved late: %s", turn["id"], type(exc).__name__)
            continue
        saved += 1
    if saved:
        log.info("saved %s late turns as notes", saved)
    return saved


async def recover_unprocessed(deps: Deps) -> int:
    """Handle the user turns of the last hour that never got an answer (a restart between the turn
    write and the reply). Called once before polling; a handled turn is marked processed, so a
    second call does nothing. Returns how many turns were handled; each turn is independent.
    Older unprocessed turns are saved as unsorted notes first (`save_late_turns`)."""
    save_late_turns(deps)
    try:
        turns = deps.store.unprocessed_turns(deps.now_fn() - RECOVER_WINDOW)
    except Exception as exc:  # noqa: BLE001
        log.warning("unprocessed turns not read: %s", type(exc).__name__)
        return 0
    handled = 0
    for turn in turns:
        try:
            outcome = await _recover_turn(deps, turn)
        except Exception as exc:  # noqa: BLE001 - one broken turn must not block the others
            log.warning("turn %s not recovered: %s", turn["id"], type(exc).__name__)
            continue
        handled += 1
        try:
            await deps.bot.send_message(turn["chat_id"], text=outcome.text,
                                        reply_markup=markup(outcome.keyboard) if outcome.keyboard else None)
        except Exception as exc:  # noqa: BLE001
            log.warning("turn %s reply not sent: %s", turn["id"], type(exc).__name__)
    if handled:
        log.info("recovered %s unprocessed turns", handled)
    return handled
