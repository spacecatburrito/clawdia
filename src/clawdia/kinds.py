"""Shelf kinds and hashtag detection."""

from __future__ import annotations

import re
from typing import Literal, get_args

Kind = Literal["task", "job", "event", "recipe", "film", "place", "buy", "idea", "unsorted"]
KINDS: tuple[str, ...] = get_args(Kind)

# Kinds that are handed to Second (everything else stays on personal shelves).
WORK_KINDS: set[str] = {"task", "job", "event"}

# Kinds posted to Second's task outbox at any hour (jobs go to /v1/jobs, not here).
SECOND_TASK_KINDS: set[str] = {"task", "event"}

HASHTAGS: dict[str, list[str]] = {
    "task": ["#task", "#work", "#задача"],
    "job": ["#job", "#vacancy", "#вакансия"],
    "event": ["#event", "#ивент"],
    "recipe": ["#recipe", "#рецепт"],
    "film": ["#film", "#movie", "#фильм"],
    "place": ["#place", "#место"],
    "buy": ["#buy", "#shop", "#купить"],
    "idea": ["#idea", "#идея"],
}

_ALIAS_TO_KIND: dict[str, str] = {
    tag.lstrip("#"): kind for kind, tags in HASHTAGS.items() for tag in tags
}

# A hashtag is '#' not glued to a preceding word character, then word chars.
_HASHTAG_RE = re.compile(r"(?<![\w#])#(\w+)")


def kind_from_hashtags(text: str) -> Kind | None:
    """Return the kind of the first known hashtag in `text` (left to right), or None."""
    for match in _HASHTAG_RE.finditer(text or ""):
        kind = _ALIAS_TO_KIND.get(match.group(1).casefold())
        if kind is not None:
            return kind  # type: ignore[return-value]
    return None
