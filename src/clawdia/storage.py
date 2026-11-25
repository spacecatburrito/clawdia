"""SQLite store for Clawdia (sync sqlite3, WAL).

All timestamps are stored as ISO 8601 UTC strings in one fixed format, so
string comparison in SQL matches chronological order. JSON columns are
encoded on write and decoded on read; rows come back as plain dicts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 6

ITEM_STATUSES = ("new", "routing", "filed", "done", "dismissed")

ITEM_COLUMNS = (
    "created_at", "updated_at", "kind", "status", "title", "text", "caption",
    "transcript", "ocr_text", "urls", "forward_from", "tg_message_id", "tg_date",
    "media_path", "media_paths", "media_mime", "media_group_id", "fields",
    "project", "confidence", "model", "filed_ref", "filed_at", "route_error",
    "resurface_at", "notes", "chat_id",
)
JSON_COLUMNS = ("fields", "forward_from", "urls", "media_paths")
TIME_COLUMNS = ("created_at", "updated_at", "tg_date", "filed_at", "resurface_at")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'unsorted'
        CHECK (kind IN ('task','job','event','recipe','film','place','buy','idea','unsorted')),
    status TEXT NOT NULL DEFAULT 'new'
        CHECK (status IN ('new','routing','filed','done','dismissed')),
    title TEXT,
    text TEXT,
    caption TEXT,
    transcript TEXT,
    ocr_text TEXT,
    urls TEXT,
    forward_from TEXT,
    tg_message_id INTEGER,
    tg_date TEXT,
    media_path TEXT,
    media_paths TEXT,
    media_mime TEXT,
    media_group_id TEXT,
    fields TEXT,
    project TEXT,
    confidence REAL,
    model TEXT,
    filed_ref TEXT,
    filed_at TEXT,
    route_error TEXT,
    resurface_at TEXT,
    notes TEXT,
    chat_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_items_kind_status ON items (kind, status);
CREATE INDEX IF NOT EXISTS idx_items_created ON items (created_at);
CREATE INDEX IF NOT EXISTS idx_items_media_group ON items (media_group_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items (id) ON DELETE CASCADE,
    at TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_item ON events (item_id);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items (id) ON DELETE CASCADE,
    at TEXT NOT NULL,
    sent_at TEXT,
    snoozed_until TEXT,
    disabled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reminders_item ON reminders (item_id);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (sent_at, disabled, at);
"""

# v1 -> v2: chat_id on items (in _SCHEMA for fresh databases, ALTER on old ones), turns table.
# Fresh databases are stamped version 1 and take this same path, so both end identical.
_MIGRATE_V2 = (
    "CREATE INDEX IF NOT EXISTS idx_items_chat_id ON items (chat_id)",
    """CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    at TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user','bot')),
    text TEXT NOT NULL,
    item_id INTEGER,
    processed INTEGER NOT NULL DEFAULT 1
)""",
    "CREATE INDEX IF NOT EXISTS idx_turns_chat ON turns (chat_id, id)",
)

# v2 -> v3: the agent brain (spec 6.14): one row per handled message, one row per write for turn Undo.
_MIGRATE_V3 = (
    """CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    at TEXT NOT NULL,
    model TEXT NOT NULL,
    calls INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    tools TEXT NOT NULL,
    error TEXT
)""",
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_at ON agent_runs (at)",
    """CREATE TABLE IF NOT EXISTS agent_writes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL,
    item_id INTEGER,
    tool TEXT NOT NULL,
    at TEXT NOT NULL,
    before TEXT NOT NULL,
    after TEXT NOT NULL,
    undone INTEGER NOT NULL DEFAULT 0
)""",
    "CREATE INDEX IF NOT EXISTS idx_agent_writes_turn ON agent_writes (turn_id)",
)

# v3 -> v4: owner memory (durable facts about the owner) and pending proposals awaiting a button.
_MIGRATE_V4 = (
    """CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section TEXT NOT NULL CHECK (section IN ('about','context')),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    source TEXT NOT NULL CHECK (source IN ('chat','button','seed','cli')),
    turn_id INTEGER
)""",
    """CREATE TABLE IF NOT EXISTS memory_pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','stored','expired')),
    memory_id INTEGER,
    closed_at TEXT
)""",
)

# v4 -> v5: call waits (Elena asked about a call whose transcript is not in Second yet).
_MIGRATE_V5 = (
    """CREATE TABLE IF NOT EXISTS call_waits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    turn_id INTEGER NOT NULL,
    message_id INTEGER,
    question TEXT NOT NULL,
    title_hint TEXT,
    baseline_id INTEGER NOT NULL,
    opened_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open','answering','answered','expired','cancelled')),
    last_check_at TEXT,
    transcript_id INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0
)""",
)

MISS_REASONS = ("no_tool", "no_data", "corrected", "undone")
MISS_STATUSES = ("open", "backlog", "dismissed")

# v5 -> v6: the memory CHECK gains 'rules' (SQLite cannot alter a CHECK, so the table is rebuilt in
# _migrate with ids kept) and misses: requests the bot could not serve, for the weekly review.
_MIGRATE_V6_MEMORY = (
    """CREATE TABLE memory_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section TEXT NOT NULL CHECK (section IN ('about','context','rules')),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    source TEXT NOT NULL CHECK (source IN ('chat','button','seed','cli')),
    turn_id INTEGER
)""",
    "INSERT INTO memory_new (id, section, text, created_at, updated_at, deleted_at, source, turn_id) "
    "SELECT id, section, text, created_at, updated_at, deleted_at, source, turn_id FROM memory",
    "DROP TABLE memory",
    "ALTER TABLE memory_new RENAME TO memory",
)
_MIGRATE_V6 = (
    """CREATE TABLE IF NOT EXISTS misses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    turn_id INTEGER,
    at TEXT NOT NULL,
    topic TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    request TEXT NOT NULL,
    reason TEXT NOT NULL CHECK (reason IN ('no_tool','no_data','corrected','undone')),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','backlog','dismissed')),
    closed_at TEXT
)""",
    "CREATE INDEX IF NOT EXISTS idx_misses_status_at ON misses (status, at)",
)


def utc_iso(value: datetime | str | None = None) -> str:
    """Format a datetime (or ISO string) as a fixed-width UTC ISO string.

    None means now. Naive datetimes are rejected for datetimes (ambiguous);
    naive ISO strings are rejected too.
    """
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        raise ValueError("naive datetime: timezone-aware value required")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _as_list(value: str | Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._enable_wal()
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _enable_wal(self) -> None:
        """Switch to WAL. The switch needs an exclusive lock that the busy timeout
        does not cover, so retry while another process holds the database."""
        deadline = time.monotonic() + 10
        while True:
            if str(self._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal":
                return
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    # -- schema ---------------------------------------------------------

    def ensure_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1')",
            )
        with self._lock:
            self._migrate()

    def _migrate(self) -> None:
        """Forward migrations, keyed on the stored schema_version. Idempotent and
        safe when two processes open the same database: BEGIN IMMEDIATE takes the
        write lock first (the busy timeout makes the loser wait), then the version
        is re-read so the loser sees the winner's result and returns."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            version = int(self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0])
            if version < 2:
                cols = [r[1] for r in self._conn.execute("PRAGMA table_info(items)")]
                if "chat_id" not in cols:
                    self._conn.execute("ALTER TABLE items ADD COLUMN chat_id INTEGER")
                for stmt in _MIGRATE_V2:
                    self._conn.execute(stmt)
                self._conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
            if version < 3:
                for stmt in _MIGRATE_V3:
                    self._conn.execute(stmt)
                if "source" not in [r[1] for r in self._conn.execute("PRAGMA table_info(turns)")]:
                    # 'typed' = Elena typed it (or a plain bot reply); 'capture' = a forward or media she sent;
                    # 'triage' = a vacancy triage message; 'legacy' = written before this column (data, never instructions)
                    self._conn.execute("ALTER TABLE turns ADD COLUMN source TEXT NOT NULL DEFAULT 'typed'")
                    # turns written before the column existed may hold captured text under either role
                    self._conn.execute("UPDATE turns SET source = 'legacy'")
                self._conn.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
            if version < 4:
                for stmt in _MIGRATE_V4:
                    self._conn.execute(stmt)
                # bot turns written before the memory release were 'typed' even when they echoed captured
                # text; the dirty check must not trust them (only new typed bot replies are clean)
                self._conn.execute("UPDATE turns SET source = 'legacy' WHERE role = 'bot' AND source = 'typed'")
                self._conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
            if version < 5:
                for stmt in _MIGRATE_V5:
                    self._conn.execute(stmt)
                self._conn.execute("UPDATE meta SET value = '5' WHERE key = 'schema_version'")
            if version < 6:
                seq = self._conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'memory'").fetchone()
                for stmt in _MIGRATE_V6_MEMORY:
                    self._conn.execute(stmt)
                if seq is not None:  # keep ids of deleted rows from being reused
                    self._conn.execute(
                        "UPDATE sqlite_sequence SET seq = MAX(seq, ?) WHERE name = 'memory'", (seq[0],))
                    self._conn.execute(
                        "INSERT INTO sqlite_sequence (name, seq) SELECT 'memory', ? "
                        "WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'memory')", (seq[0],))
                for stmt in _MIGRATE_V6:
                    self._conn.execute(stmt)
                self._conn.execute(
                    "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))
            self._conn.execute("COMMIT")
        except BaseException:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- row helpers ----------------------------------------------------

    @staticmethod
    def _decode_item(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for col in JSON_COLUMNS:
            if item.get(col) is not None:
                item[col] = json.loads(item[col])
        return item

    @staticmethod
    def _encode_fields(fields: dict[str, Any]) -> dict[str, Any]:
        unknown = set(fields) - set(ITEM_COLUMNS)
        if unknown:
            raise ValueError(f"unknown item field(s): {', '.join(sorted(unknown))}")
        out = dict(fields)
        for col in JSON_COLUMNS:
            if out.get(col) is not None:
                out[col] = json.dumps(out[col], ensure_ascii=False)
        for col in TIME_COLUMNS:
            if out.get(col) is not None:
                out[col] = utc_iso(out[col])
        return out

    # -- items ----------------------------------------------------------

    def create_item(self, **fields: Any) -> int:
        data = self._encode_fields(fields)
        now = utc_iso()
        data.setdefault("created_at", now)
        data["updated_at"] = now
        data.setdefault("kind", "unsorted")
        data.setdefault("status", "new")
        cols = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"INSERT INTO items ({cols}) VALUES ({marks})", tuple(data.values())
            )
            return int(cur.lastrowid)

    def update_item(self, id: int, **fields: Any) -> None:
        data = self._encode_fields(fields)
        data["updated_at"] = utc_iso()
        sets = ", ".join(f"{c} = ?" for c in data)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE items SET {sets} WHERE id = ?", (*data.values(), id)
            )

    def get_item(self, id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM items WHERE id = ?", (id,)).fetchone()
        return self._decode_item(row) if row else None

    def list_items(
        self,
        kind: str | Iterable[str] | None = None,
        status: str | Iterable[str] | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        for col, value in (("kind", kind), ("status", status)):
            values = _as_list(value)
            if values is not None:
                where.append(f"{col} IN ({', '.join('?' for _ in values)})")
                params.extend(values)
        if since is not None:
            where.append("created_at >= ?")
            params.append(utc_iso(since))
        sql = "SELECT * FROM items"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode_item(r) for r in rows]

    def outbox(self) -> list[dict[str, Any]]:
        from clawdia.kinds import SECOND_TASK_KINDS

        kinds = sorted(SECOND_TASK_KINDS)
        sql = (
            f"SELECT * FROM items WHERE status = 'routing' "
            f"AND kind IN ({', '.join('?' for _ in kinds)}) "
            "ORDER BY created_at ASC, id ASC"
        )
        with self._lock:
            rows = self._conn.execute(sql, kinds).fetchall()
        return [self._decode_item(r) for r in rows]

    def recent_items(self, chat_id: int, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM items WHERE chat_id = ? AND status != 'dismissed' "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [self._decode_item(r) for r in rows]

    # -- conversation turns ---------------------------------------------

    def add_turn(
        self,
        chat_id: int,
        role: str,
        text: str,
        *,
        item_id: int | None = None,
        processed: bool = True,
        at: datetime | None = None,
        source: str = "typed",
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO turns (chat_id, at, role, text, item_id, processed, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chat_id, utc_iso(at), role, text, item_id, 1 if processed else 0, source),
            )
            return int(cur.lastrowid)

    def mark_turn(
        self, turn_id: int, *, processed: bool = True, item_id: int | None = None
    ) -> None:
        with self._lock, self._conn:
            if item_id is None:
                self._conn.execute(
                    "UPDATE turns SET processed = ? WHERE id = ?",
                    (1 if processed else 0, turn_id),
                )
            else:
                self._conn.execute(
                    "UPDATE turns SET processed = ?, item_id = ? WHERE id = ?",
                    (1 if processed else 0, item_id, turn_id),
                )

    def recent_turns(self, chat_id: int, limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM turns WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def turn_get(self, turn_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
        return dict(row) if row else None

    def last_bot_turn_at(self, chat_id: int) -> datetime | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT at FROM turns WHERE chat_id = ? AND role = 'bot' "
                "ORDER BY at DESC, id DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
        return datetime.fromisoformat(row["at"]) if row else None

    def unprocessed_turns(
        self, since: datetime | None = None, *, before: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Unprocessed user turns with `since <= at` and `at < before` (either bound optional)."""
        sql = "SELECT * FROM turns WHERE role = 'user' AND processed = 0 AND source != 'followup'"
        args: list[Any] = []
        if since is not None:
            sql += " AND at >= ?"
            args.append(utc_iso(since))
        if before is not None:
            sql += " AND at < ?"
            args.append(utc_iso(before))
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY at ASC, id ASC", args).fetchall()
        return [dict(r) for r in rows]

    def count_events(self, type: str, since: datetime) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE type = ? AND at >= ?", (type, utc_iso(since))
            ).fetchone()
        return int(row[0])

    def prune_turns(self, chat_id: int, keep: int = 20) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM turns WHERE chat_id = ? AND processed = 1 AND id NOT IN "
                "(SELECT id FROM turns WHERE chat_id = ? ORDER BY id DESC LIMIT ?)",
                (chat_id, chat_id, keep),
            )

    # -- events and meta ------------------------------------------------

    def add_event(self, item_id: int, type: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO events (item_id, at, type, payload) VALUES (?, ?, ?, ?)",
                (item_id, utc_iso(), type, json.dumps(payload or {}, ensure_ascii=False)),
            )

    def max_event_id(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(id), 0) AS id FROM events").fetchone()
        return int(row["id"])

    def events_after(self, after_id: int, types: Iterable[str]) -> list[dict[str, Any]]:
        """Events newer than `after_id` of the given types, oldest first, payload decoded."""
        kinds = list(types)
        marks = ",".join("?" * len(kinds))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, item_id, at, type, payload FROM events WHERE id > ? AND type IN ({marks}) ORDER BY id",
                (after_id, *kinds),
            ).fetchall()
        out = []
        for r in rows:
            try:
                payload = json.loads(r["payload"] or "{}")
            except ValueError:
                payload = {}
            out.append({**dict(r), "payload": payload if isinstance(payload, dict) else {}})
        return out

    def event_types(self, item_id: int) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT type FROM events WHERE item_id = ?", (item_id,)).fetchall()
        return {r["type"] for r in rows}

    def meta_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- laptop day-map request (meta keys, no table) -------------------

    def focus_request_open(self, now: datetime) -> None:
        """Leave one request for the laptop; a second press while one is waiting keeps the first."""
        if self.meta_get("focus_request"):
            return
        stamp = utc_iso(now)
        self.meta_set("focus_request", json.dumps({"requested_at": stamp}))
        self.meta_set("focus_requested_at", stamp)  # survives take: the one-hour reminder reads it

    def focus_request_take(self) -> dict | None:
        """Return and clear the pending request."""
        raw = self.meta_get("focus_request")
        if not raw:
            return None
        self.meta_set("focus_request", "")
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def focus_report(self, status: str, reason: str, now: datetime) -> None:
        self.meta_set("focus_report", json.dumps({"status": status, "reason": reason, "at": utc_iso(now)}))

    def focus_report_get(self) -> dict | None:
        raw = self.meta_get("focus_report")
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return None

    # -- reminders ------------------------------------------------------

    def add_reminder(self, item_id: int, at: datetime) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO reminders (item_id, at) VALUES (?, ?)", (item_id, utc_iso(at))
            )
            return int(cur.lastrowid)

    def get_reminder(self, id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM reminders WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None

    def due_reminders(self, now: datetime) -> list[dict[str, Any]]:
        now_s = utc_iso(now)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE at <= ? AND sent_at IS NULL AND disabled = 0 "
                "AND (snoozed_until IS NULL OR snoozed_until <= ?) ORDER BY at ASC, id ASC",
                (now_s, now_s),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_reminder_sent(self, id: int, now: datetime) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE reminders SET sent_at = ? WHERE id = ?", (utc_iso(now), id))

    def snooze_reminder(self, id: int, until: datetime) -> None:
        """Move one reminder to `until`. Snoozing is an explicit ask, so it also re-enables that reminder."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE reminders SET snoozed_until = ?, sent_at = NULL, disabled = 0 WHERE id = ?",
                (utc_iso(until), id),
            )

    def latest_reminder(self, item_id: int) -> dict[str, Any] | None:
        """The item's reminder with the latest effective time (snoozed_until or at), or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reminders WHERE item_id = ? "
                "ORDER BY COALESCE(snoozed_until, at) DESC, id DESC LIMIT 1",
                (item_id,),
            ).fetchone()
        return dict(row) if row else None

    def disable_reminders(self, item_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE reminders SET disabled = 1 WHERE item_id = ?", (item_id,))

    def reminders_of(self, item_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE item_id = ? ORDER BY id ASC", (item_id,)).fetchall()
        return [dict(r) for r in rows]

    def restore_reminders(self, item_id: int, rows: list[dict[str, Any]]) -> None:
        """Put an item's reminders back to a `reminders_of` snapshot; reminders made since are turned off."""
        keep = {int(r["id"]) for r in rows}
        with self._lock, self._conn:
            for r in rows:
                self._conn.execute(
                    "UPDATE reminders SET at = ?, sent_at = ?, snoozed_until = ?, disabled = ? WHERE id = ?",
                    (r["at"], r["sent_at"], r["snoozed_until"], r["disabled"], r["id"]))
            for r in self.reminders_of(item_id):
                if int(r["id"]) not in keep:
                    self._conn.execute("UPDATE reminders SET disabled = 1 WHERE id = ?", (r["id"],))

    # -- agent runs and writes (spec 6.14) --------------------------------

    def add_agent_run(self, *, turn_id: int, chat_id: int, at: datetime, model: str, calls: int,
                      input_tokens: int, output_tokens: int, cost_usd: float, tools: list[str],
                      error: str | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO agent_runs (turn_id, chat_id, at, model, calls, input_tokens, output_tokens, "
                "cost_usd, tools, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (turn_id, chat_id, utc_iso(at), model, calls, input_tokens, output_tokens, cost_usd,
                 json.dumps(tools), error))

    def agent_spend_since(self, start: datetime) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM agent_runs WHERE at >= ?", (utc_iso(start),)).fetchone()
        return float(row[0])

    def add_agent_write(self, turn_id: int, item_id: int | None, tool: str, before: dict, after: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO agent_writes (turn_id, item_id, tool, at, before, after) VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, item_id, tool, utc_iso(), json.dumps(before, ensure_ascii=False),
                 json.dumps(after, ensure_ascii=False)))

    def agent_writes(self, turn_id: int) -> list[dict[str, Any]]:
        """The turn's writes in insertion order; `before` and `after` decoded."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agent_writes WHERE turn_id = ? ORDER BY id ASC", (turn_id,)).fetchall()
        out = []
        for r in rows:
            w = dict(r)
            w["before"], w["after"] = json.loads(w["before"]), json.loads(w["after"])
            out.append(w)
        return out

    def mark_write_undone(self, write_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE agent_writes SET undone = 1 WHERE id = ?", (write_id,))

    def mark_writes_undone(self, turn_id: int) -> bool:
        """False when the turn has no write left to undo (already undone, or none)."""
        with self._lock, self._conn:
            cur = self._conn.execute("UPDATE agent_writes SET undone = 1 WHERE turn_id = ? AND undone = 0", (turn_id,))
            return cur.rowcount > 0

    # -- owner memory -----------------------------------------------------

    def memory_add(self, section: str, text: str, *, source: str, turn_id: int | None, now: datetime) -> int:
        """Insert a fact (cleaned defensively). No validation: callers use memory.check_add."""
        from clawdia.memory import clean

        at = utc_iso(now)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO memory (section, text, created_at, updated_at, source, turn_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (section, clean(text), at, at, source, turn_id))
            return int(cur.lastrowid)

    # -- misses -----------------------------------------------------------

    def miss_add(self, chat_id: int, turn_id: int | None, topic: str, request: str, reason: str,
                 now: datetime) -> int:
        from clawdia.memory import clean

        if reason not in MISS_REASONS:
            raise ValueError(f"unknown miss reason {reason!r}")
        topic = clean(topic)[:60]
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO misses (chat_id, turn_id, at, topic, topic_key, request, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chat_id, turn_id, utc_iso(now), topic, topic.casefold(), clean(request)[:300], reason))
            return int(cur.lastrowid)

    def misses_since(self, since: datetime, status: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM misses WHERE at >= ?", [utc_iso(since)]
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY at DESC, id DESC", args).fetchall()
        return [dict(r) for r in rows]

    def miss_get(self, miss_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM misses WHERE id = ?", (miss_id,)).fetchone()
        return dict(row) if row else None

    def misses_close_topic(self, miss_id: int, status: str, now: datetime, until: datetime | None = None) -> int:
        """Move every open miss sharing this miss's topic (case-insensitive) to backlog or dismissed.
        With until, only misses logged at or before it (the review's window)."""
        if status not in ("backlog", "dismissed"):
            raise ValueError(f"cannot close a miss as {status!r}")
        with self._lock, self._conn:
            row = self._conn.execute("SELECT topic_key FROM misses WHERE id = ?", (miss_id,)).fetchone()
            if row is None:
                return 0
            sql = "UPDATE misses SET status = ?, closed_at = ? WHERE topic_key = ? AND status = 'open'"
            params: list[Any] = [status, utc_iso(now), row["topic_key"]]
            if until is not None:
                sql += " AND at <= ?"
                params.append(utc_iso(until))
            cur = self._conn.execute(sql, params)
            return cur.rowcount

    def memory_list(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        """Facts ordered by section (about, context, rules), then id."""
        where = "" if include_deleted else "WHERE deleted_at IS NULL "
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memory {where}"
                "ORDER BY CASE section WHEN 'about' THEN 0 WHEN 'context' THEN 1 ELSE 2 END, id").fetchall()
        return [dict(r) for r in rows]

    def memory_get(self, memory_id: int) -> dict[str, Any] | None:
        """One fact by id, deleted rows included."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM memory WHERE id = ?", (memory_id,)).fetchone()
        return dict(row) if row else None

    def memory_delete(self, memory_id: int, now: datetime) -> bool:
        """Soft delete. False when the fact is missing or already deleted."""
        at = utc_iso(now)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE memory SET deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
                (at, at, memory_id))
            return cur.rowcount > 0

    def memory_restore(self, memory_id: int, now: datetime) -> bool:
        """Undo a soft delete. False when the fact is missing or not deleted."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE memory SET deleted_at = NULL, updated_at = ? WHERE id = ? AND deleted_at IS NOT NULL",
                (utc_iso(now), memory_id))
            return cur.rowcount > 0

    def memory_total(self, section: str) -> int:
        """Characters over the section's live facts."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(LENGTH(text)), 0) FROM memory WHERE section = ? AND deleted_at IS NULL",
                (section,)).fetchone()
        return int(row[0])

    def pending_add(self, section: str, text: str, *, turn_id: int, now: datetime) -> int:
        """A proposal for Elena's button. The same open (turn, section, text) is returned, not repeated."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id FROM memory_pending WHERE turn_id = ? AND section = ? AND text = ? AND status = 'open' "
                "ORDER BY id LIMIT 1", (turn_id, section, text)).fetchone()
            if row is not None:
                return int(row[0])
            cur = self._conn.execute(
                "INSERT INTO memory_pending (section, text, turn_id, created_at) VALUES (?, ?, ?, ?)",
                (section, text, turn_id, utc_iso(now)))
            return int(cur.lastrowid)

    def pending_get(self, pending_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memory_pending WHERE id = ?", (pending_id,)).fetchone()
        return dict(row) if row else None

    def pending_close(self, pending_id: int, status: str, memory_id: int | None, now: datetime) -> bool:
        """Close an open proposal once. False when it was not open (or does not exist)."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE memory_pending SET status = ?, memory_id = ?, closed_at = ? "
                "WHERE id = ? AND status = 'open'",
                (status, memory_id, utc_iso(now), pending_id))
            return cur.rowcount > 0

    # -- call waits -------------------------------------------------------

    def wait_open(self, chat_id: int, turn_id: int, message_id: int | None, title_hint: str | None,
                  baseline_id: int, now: datetime, ttl: timedelta, question: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO call_waits (chat_id, turn_id, message_id, question, title_hint, baseline_id, "
                "opened_at, expires_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')",
                (chat_id, turn_id, message_id, question, title_hint, baseline_id, utc_iso(now), utc_iso(now + ttl)))
            return int(cur.lastrowid)

    def wait_fail(self, wait_id: int) -> int:
        """Count one failed answer attempt; returns the new total."""
        with self._lock, self._conn:
            self._conn.execute("UPDATE call_waits SET attempts = attempts + 1 WHERE id = ?", (wait_id,))
            row = self._conn.execute("SELECT attempts FROM call_waits WHERE id = ?", (wait_id,)).fetchone()
        return int(row["attempts"]) if row else 0

    def waits_open(self, chat_id: int | None = None) -> list[dict[str, Any]]:
        """Open waits, oldest first, optionally for one chat."""
        sql = "SELECT * FROM call_waits WHERE status = 'open'"
        args: tuple = ()
        if chat_id is not None:
            sql += " AND chat_id = ?"
            args = (chat_id,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY id", args).fetchall()
        return [dict(r) for r in rows]

    def wait_get(self, wait_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM call_waits WHERE id = ?", (wait_id,)).fetchone()
        return dict(row) if row else None

    def wait_claim(self, wait_id: int, transcript_id: int, now: datetime) -> bool:
        """Atomic open -> answering. False when the wait was not open (someone else holds it)."""
        at = utc_iso(now)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE call_waits SET status = 'answering', transcript_id = ?, last_check_at = ? "
                "WHERE id = ? AND status = 'open'", (transcript_id, at, wait_id))
            return cur.rowcount > 0

    def wait_set_status(self, wait_id: int, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE call_waits SET status = ? WHERE id = ?", (status, wait_id))

    def wait_cancel(self, wait_id: int) -> bool:
        """Atomic open -> cancelled. False when the wait was not open."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE call_waits SET status = 'cancelled' WHERE id = ? AND status = 'open'", (wait_id,))
            return cur.rowcount > 0

    def wait_cancel_chat(self, chat_id: int) -> int:
        """Cancel every open wait of a chat; returns how many."""
        return len(self.wait_cancel_chat_ids(chat_id))

    def wait_cancel_chat_ids(self, chat_id: int) -> list[int]:
        """Cancel every open wait of a chat; returns the ids actually cancelled."""
        with self._lock, self._conn:
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM call_waits WHERE chat_id = ? AND status = 'open' ORDER BY id", (chat_id,))]
            self._conn.execute(
                "UPDATE call_waits SET status = 'cancelled' WHERE chat_id = ? AND status = 'open'", (chat_id,))
            return ids

    def wait_reopen(self, wait_id: int, now: datetime) -> bool:
        """Atomic cancelled -> open, only while the wait has not expired. False otherwise."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE call_waits SET status = 'open' WHERE id = ? AND status = 'cancelled' AND expires_at > ?",
                (wait_id, utc_iso(now)))
            return cur.rowcount > 0

    def waits_reset_answering(self, now: datetime, *, older_than: timedelta | None = timedelta(minutes=15)) -> int:
        """Answering waits go back to open: those claimed `older_than` or more before now, or every
        one when `older_than` is None (startup: nothing can be in flight)."""
        sql, args = "UPDATE call_waits SET status = 'open' WHERE status = 'answering'", ()
        if older_than is not None:
            sql += " AND last_check_at <= ?"
            args = (utc_iso(now - older_than),)
        with self._lock, self._conn:
            cur = self._conn.execute(sql, args)
            return cur.rowcount

    # -- backup ---------------------------------------------------------

    def backup_to(self, path: Path) -> None:
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(dest_path)
        try:
            with self._lock:
                self._conn.backup(dest)
        finally:
            dest.close()
