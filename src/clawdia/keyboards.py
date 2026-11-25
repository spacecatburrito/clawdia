"""Inline keyboard helper shared by the handlers and the reminder sender."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def markup(keyboard: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=data) for label, data in row] for row in keyboard])


def undo_keyboard(item_id: int) -> list[list[tuple[str, str]]]:
    """The only button on a capture reply."""
    return [[("Undo", f"undo:{item_id}")]]


def undo_turn_keyboard(turn_id: int) -> list[list[tuple[str, str]]]:
    """The one button on an agent reply that wrote something: reverts every write of the turn."""
    return [[("Undo", f"undoturn:{turn_id}")]]


def memory_keyboard(pending_ids: list[int], lang: str) -> list[list[tuple[str, str]]]:
    """One numbered Remember button per open memory proposal, in the order of the proposal lines the
    reply shows ("Remember 1", "Remember 2"); only the owner's account can press it."""
    label = "Запомнить" if lang == "ru" else "Remember"
    return [[(f"{label} {n}", f"memok:{pid}")] for n, pid in enumerate(pending_ids, 1)]


def call_wait_keyboard(wait_id: int, lang: str) -> list[list[tuple[str, str]]]:
    """The button under a reply that opened a call wait: stops waiting for the transcript."""
    return [[("Отменить ожидание" if lang == "ru" else "Cancel", f"callwait:cancel:{wait_id}")]]


def focus_request_keyboard(lang: str) -> list[list[tuple[str, str]]]:
    """The button under "No day map yet": leaves a request for the laptop."""
    return [[("Собрать на ноутбуке" if lang == "ru" else "Build on laptop", "focusreq:open")]]
