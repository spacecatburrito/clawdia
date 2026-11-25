"""Read access to Second for the agent, through second-api `POST /v1/read/{tool}`.

The Second API (CLAWDIA_SECOND_API_URL) serves the read tools of Second's MCP server in-process:
the same validation, limits, read-only transaction, secret masking and `untrusted` envelope as
the MCP server. Clawdia calls it with the token it already uses for tasks and vacancies, so this
adds no privilege, no process and no dependency. Only allowlisted clients may read (403 otherwise).

One `SecondReader` per agent turn: the first read also asks `freshness`, and every result carries
it, so the reply can say how old the data is. Errors are the second_client ones
(`SecondUnavailable`, `SecondAuthFailed`, `SecondRejected`); the caller bounds the time.

Everything a read returns is third-party data. The trust rule (no writes for the rest of a turn
that read Second) is enforced in `agent_tools.call_tool`, not here.
"""

from __future__ import annotations

import logging
from typing import Any

from clawdia.second_client import SecondRejected

log = logging.getLogger(__name__)

FRESHNESS_KEYS = ("latest_message_at", "latest_message_age_minutes")
CHATS_FOR_LINKS = 200  # the server's maximum; blacklisted chats are not listed and get no link


class SecondReader:
    """The reads of one turn. `client` is a SecondClient (or anything with `read(tool, args)`)."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.freshness: dict | None = None
        self._chats: dict[int, tuple[int | None, str | None]] | None = None

    async def call(self, name: str, args: dict) -> dict:
        """Run one read tool; returns its result with `freshness` added. Raises the second_client errors."""
        if self.freshness is None:
            try:
                fresh = await self._client.read("freshness", {})
                self.freshness = {k: fresh.get(k) for k in FRESHNESS_KEYS}
            except SecondRejected as exc:  # the read itself may still work; the reply says freshness is unknown
                log.warning("second reads: freshness refused: %s", exc.code)
                self.freshness = {"error": "freshness unknown"}
        data = await self._client.read(name, args)
        return {**data, "freshness": self.freshness}

    async def chats(self) -> dict[int, tuple[int | None, str | None]]:
        """Second's chat id -> (tg_chat_id, type), read once per turn, for links to a task's source."""
        if self._chats is None:
            data = await self._client.read("list_chats", {"limit": CHATS_FOR_LINKS})
            self._chats = {row["id"]: (row.get("tg_chat_id"), row.get("type"))
                           for row in data.get("items") or [] if isinstance(row, dict) and row.get("id") is not None}
        return self._chats

    async def close(self) -> None:
        """Nothing to release over HTTP; kept so the loop ends every turn the same way."""
        return None
