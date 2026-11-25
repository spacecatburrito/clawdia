"""A typed comment and a forward that arrive one right after the other become one item.

Telegram sends "forward with comment" as the comment, then the forward, within about a second. A typed
text waits up to `pair_window_s` for forwards from the same chat; a forward waits the same time for a
typed text. The text handler always owns the result: it builds the glued item and the single reply.
Without a counterpart each message is handled exactly as before. Media albums are never paired.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from clawdia.capture import Deps, process_glued
from clawdia.converse import Outcome, handle_capture

SETTLE_S = 1.0  # after the first forward, how long a waiting text collects further forwards (capped at window/4)


@dataclass
class _Slot:
    kind: str  # "text" or "fwd"
    messages: list[Any] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    claimed: bool = False


def _drop(deps: Deps, chat_id: int, slot: _Slot) -> None:
    slots = deps.pair_slots.get(chat_id, [])
    if slot in slots:
        slots.remove(slot)
    if not slots:
        deps.pair_slots.pop(chat_id, None)


def _window(deps: Deps) -> float:
    return float(getattr(deps.settings, "pair_window_s", 0) or 0)


async def hold_text(message: Any, deps: Deps) -> list[Any]:
    """Wait for forwards to glue to this typed text; the forwards to glue, or [] to handle it alone."""
    window = _window(deps)
    if window <= 0:
        return []
    chat_id = message.chat.id
    waiting = [s for s in deps.pair_slots.get(chat_id, []) if s.kind == "fwd" and not s.claimed]
    if waiting:  # the forward came first
        for slot in waiting:
            slot.claimed = True
            slot.event.set()
        return [m for slot in waiting for m in slot.messages]
    slot = _Slot("text")
    deps.pair_slots.setdefault(chat_id, []).append(slot)
    try:
        await asyncio.wait_for(slot.event.wait(), window)
        await asyncio.sleep(min(SETTLE_S, window / 4))  # a second forward of the same share
    except asyncio.TimeoutError:
        pass
    finally:
        _drop(deps, chat_id, slot)
    return list(slot.messages)


async def hold_forward(message: Any, deps: Deps) -> bool:
    """True when a typed text took this forward (it replies, not the caller); False to capture it alone.
    Only real forwards pair: her own photo or file (no forward_origin) is captured at once."""
    window = _window(deps)
    if window <= 0 or getattr(message, "forward_origin", None) is None or getattr(message, "media_group_id", None):
        return False
    chat_id = message.chat.id
    texts = [s for s in deps.pair_slots.get(chat_id, []) if s.kind == "text"]
    if texts:  # the comment came first
        texts[-1].messages.append(message)
        texts[-1].event.set()
        return True
    slot = _Slot("fwd", [message])
    deps.pair_slots.setdefault(chat_id, []).append(slot)
    try:
        await asyncio.wait_for(slot.event.wait(), window)
    except asyncio.TimeoutError:
        pass
    finally:
        _drop(deps, chat_id, slot)
    return slot.claimed


async def handle_glued(message: Any, deps: Deps, forwards: list[Any], turn_id: int) -> Outcome:
    """Her comment is the instruction (never a command); the forwards are context data."""
    captured = await process_glued(message.text or "", forwards, deps)
    return await handle_capture(message, deps, captured, turn_id=turn_id)
