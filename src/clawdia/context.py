"""Context snapshot from Second: on-disk cache and the stable prompt block."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Snapshot:
    projects: list[dict] = field(default_factory=list)
    people: list[dict] = field(default_factory=list)
    open_tasks: list[dict] = field(default_factory=list)
    generated_at: str | None = None
    etag: str | None = None


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clean(value: Any) -> str:
    """Collapse newlines and whitespace runs so a value cannot add lines to the prompt."""
    return " ".join(str(value).split())


def _list_of_dicts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, dict)]


class ContextCache:
    """Last good /v1/context body kept on disk so tagging works while the API is down."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, body: dict, etag: str) -> None:
        """Write {"body", "etag"} atomically (temp file in the same dir, then os.replace), mode 0600."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"body": body, "etag": etag}, fh, ensure_ascii=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self) -> Snapshot:
        """Return the cached snapshot, or an empty one when the file is missing or corrupt."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            body = data["body"]
            if not isinstance(body, dict):
                return Snapshot()
            etag = data.get("etag")
            generated_at = body.get("generated_at")
            return Snapshot(
                projects=_list_of_dicts(body.get("projects")),
                people=_list_of_dicts(body.get("people")),
                open_tasks=_list_of_dicts(body.get("open_tasks")),
                generated_at=generated_at if isinstance(generated_at, str) else None,
                etag=etag if isinstance(etag, str) else None,
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return Snapshot()

    def age_seconds(self, now: datetime) -> int | None:
        """Seconds since the snapshot was generated, None when there is no usable stamp."""
        stamp = self.load().generated_at
        if not stamp:
            return None
        try:
            return int((now - _parse_iso(stamp)).total_seconds())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def prompt_block(snapshot: Snapshot) -> str:
        """Deterministic text for the cached prompt prefix; order is the API's order."""
        lines: list[str] = []
        if snapshot.projects:
            lines.append("Projects:")
            for p in snapshot.projects:
                line = _clean(p.get("name", ""))
                if p.get("description"):
                    line += f" - {_clean(p['description'])}"
                aliases = [_clean(a) for a in (p.get("aliases") or []) if a]
                if aliases:
                    line += f" (aliases: {', '.join(aliases)})"
                lines.append(line)
        if snapshot.people:
            lines.append("People:")
            for person in snapshot.people:
                line = _clean(person.get("name", ""))
                if person.get("project"):
                    line += f" ({_clean(person['project'])})"
                lines.append(line)
        if snapshot.open_tasks:
            lines.append("Open tasks:")
            for task in snapshot.open_tasks:
                line = _clean(task.get("title", ""))
                if task.get("project"):
                    line += f" [{_clean(task['project'])}]"
                lines.append(line)
        return "\n".join(lines) if lines else "(no context available)"
