"""The owner's daily focus map for the agent: the newest `YYYY-MM-DD.md` that another job placed in
`settings.focus_dir` (read-only here; the job that writes the files is not part of this package).

The map is her priority list: Top-N in order, the queue, what others owe her, the calendar. It is
written by Claude from her messages and calls, so it holds other people's words: the agent treats it
as data, and reading it stops writes for the rest of the turn (like a Second read).
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

MAP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")
# section heading words -> key and character budget in the tool result
SECTIONS = (
    ("decisions", ("decisions for you",), 800),
    ("top", ("top-",), 6000),
    ("queue", ("queue",), 3500),
    ("their_move", ("their move",), 2500),
    ("calendar", ("calendar",), 1500),
    ("not_in_calendar", ("не в календаре", "not in calendar"), 1000),
)
_LOCAL_LINK_RE = re.compile(r"\s*\[[^\]]*\]\(<?/[^)>]*>?\)")  # links to files on her laptop: useless on the phone
_LINK_RE = re.compile(r"\[([^\]]*)\]\(<?(https?://[^)>\s]+)>?\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def newest_map(folder: Path, today: date) -> tuple[date, Path] | None:
    """The newest map dated today or earlier (a map from the future is ignored)."""
    best: tuple[date, Path] | None = None
    try:
        entries = list(Path(folder).iterdir())
    except OSError:
        return None
    for path in entries:
        m = MAP_RE.match(path.name)
        if not m or not path.is_file():
            continue
        try:
            day = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if day <= today and (best is None or day > best[0]):
            best = (day, path)
    return best


def today_map(folder: Path, today: date) -> Path | None:
    """The map dated exactly today, or None."""
    found = newest_map(folder, today)
    return found[1] if found is not None and found[0] == today else None


def is_auto(text: str) -> bool:
    """True for a map the laptop built by itself (its first line starts with "Built automatically")."""
    return text.lstrip().startswith("Built automatically")


def _clean(text: str) -> str:
    text = _LOCAL_LINK_RE.sub("", text)
    text = _LINK_RE.sub(lambda m: f"{m.group(1)}: {m.group(2)}" if m.group(1) not in ("source", "") else m.group(2), text)
    return _BOLD_RE.sub(r"\1", text).strip()


def _cut(text: str, n: int) -> tuple[str, bool]:
    if len(text) <= n:
        return text, False
    head = text[:n]
    cut = head.rfind("\n")
    return (head[:cut] if cut > n // 2 else head).rstrip() + "\n...[more in the map]", True


def split_sections(text: str) -> dict[str, str]:
    """`## heading` blocks keyed by their lowercased heading (emoji and counts kept in the key)."""
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip().casefold()
            out[current] = ""
        elif current is not None:
            out[current] += line + "\n"
    return out


def load_plan(folder: Path, today: date) -> dict:
    """The parts of the newest map the agent needs, each cut to its budget."""
    found = newest_map(folder, today)
    if found is None:
        return {"error": "no daily focus map on the server yet"}
    day, path = found
    sections = split_sections(path.read_text(encoding="utf-8", errors="replace"))
    out: dict = {"map_date": day.isoformat(), "is_today": day == today}
    for key, words, budget in SECTIONS:
        heading = next((h for h in sections if any(w in h for w in words)), None)
        if heading is None:
            continue
        text, cut = _cut(_clean(sections[heading]), budget)
        out[key] = text
        if cut:
            out[f"{key}_cut"] = True
    return out


_TOP_LINE_RE = re.compile(r"^\s*\d+[.)]\s+(.*\S)")


def top_items(text: str, n: int) -> list[str]:
    """The first n numbered lines of the Top section, cleaned."""
    sections = split_sections(text)
    heading = next((h for h in sections if "top-" in h), None)
    if heading is None:
        return []
    items = []
    for line in sections[heading].splitlines():
        m = _TOP_LINE_RE.match(line)
        if m:
            items.append(_clean(m.group(1)))
            if len(items) == n:
                break
    return items
