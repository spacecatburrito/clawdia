"""Client for second-api (http://127.0.0.1:8091) and the outbox drain.

Error mapping: Second being down or throttling (connection errors, timeouts,
5xx, 429) is SecondUnavailable - keep the item in `routing` and stop. Any
other 4xx is SecondRejected - the item will not succeed by retrying.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx

from clawdia import __version__
from clawdia.storage import Store

log = logging.getLogger(__name__)

TIMEOUT_S = 10.0
REFRESH_TIMEOUT_S = 70.0  # the server bounds a refresh at 60 s
TITLE_MAX = 500
TITLE_FALLBACK_CHARS = 80
EXCERPT_CHARS = 300
PROJECT_MIN_CONFIDENCE = 0.6  # spec 6.2 step 4: below this the task lands in Second's Inbox
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class SecondUnavailable(Exception):
    """Second is unreachable, failing (5xx) or throttling (429)."""


class SecondRejected(Exception):
    """Second answered with a client error that retrying will not fix."""

    def __init__(self, code: str, message: str = "", status: int | None = None):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.status = status


class SecondAuthFailed(SecondUnavailable):
    """401/403 on a task post: our token is wrong, so every item would fail the same way."""


def source_text(item: dict[str, Any]) -> str:
    """What Second gets as the task body: text, caption, transcript, then the forward sender."""
    parts = [p for p in (item.get("text"), item.get("caption")) if p and p.strip()]
    transcript = item.get("transcript")
    if transcript and transcript.strip():
        parts.append(f"Transcript: {transcript}")
    origin = item.get("forward_from")
    name = origin.get("name") if isinstance(origin, dict) else None
    if name:
        parts.append(f"Forwarded from: {name}")
    link = origin.get("link") if isinstance(origin, dict) else None
    if link and not any(link in p for p in parts):
        parts.append(f"Link: {link}")
    return "\n".join(parts)


def routed_project(item: dict[str, Any]) -> str | None:
    """The project sent to Second: the tagged one only when the tagger was sure enough, else None (Inbox)."""
    project = item.get("project")
    confidence = item.get("confidence")
    if not project or not isinstance(confidence, (int, float)) or confidence < PROJECT_MIN_CONFIDENCE:
        return None
    return project


def idempotency_key(item: dict[str, Any]) -> str:
    """`clawdia-<id>-<created_at in epoch microseconds>`: stable across retries of one row, and a
    restored or reset database that reuses an id gets a new key instead of an old task replayed."""
    created = item["created_at"]
    stamp = created if isinstance(created, datetime) else datetime.fromisoformat(str(created))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return f"clawdia-{item['id']}-{(stamp - _EPOCH) // timedelta(microseconds=1)}"


def job_idempotency_key(item: dict[str, Any]) -> str:
    """`clawdia-job-<id>-<created_at epoch us>`: like `idempotency_key`, in the jobs namespace."""
    return "clawdia-job-" + idempotency_key(item).removeprefix("clawdia-")


def _title(item: dict[str, Any]) -> str:
    title = (item.get("title") or "").strip()
    if not title:
        first = next((item.get(k) for k in ("text", "caption", "transcript") if (item.get(k) or "").strip()), "")
        title = first.strip()[:TITLE_FALLBACK_CHARS]
    return (title or "(untitled)")[:TITLE_MAX]


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_date(value: Any) -> str | None:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return value


def event_task_body(item: dict[str, Any]) -> dict[str, Any]:
    """An event as a dated Second task: "Event: <title>", deadline = an explicit correction or the event date, body = time and place."""
    fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
    head = " · ".join(str(fields[k]).strip() for k in ("time", "place") if str(fields.get(k) or "").strip())
    text = source_text(item)
    body = "\n".join(p for p in (head, text) if p)
    return {"title": f"Event: {_title(item)}"[:TITLE_MAX], "deadline": _valid_date(fields.get("deadline")) or _valid_date(fields.get("date")), "body": body}


class SecondClient:
    def __init__(self, base_url: str, token: str, http: httpx.AsyncClient | None = None):
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT_S)
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _request(
        self, method: str, path: str, *, headers: dict[str, str] | None = None, json: Any = None,
        timeout: float = TIMEOUT_S,
    ) -> httpx.Response:
        # An injected client may have no base_url, so build the full URL ourselves.
        url = path if str(self._http.base_url) else self._base_url + path
        try:
            resp = await self._http.request(
                method, url, headers={**self._headers, **(headers or {})}, json=json, timeout=timeout
            )
        except httpx.HTTPError as exc:
            raise SecondUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 500 or resp.status_code == 429:
            raise SecondUnavailable(f"http_{resp.status_code}")
        if resp.status_code >= 400:
            code, message = f"http_{resp.status_code}", ""
            try:
                err = resp.json()["error"]
                code = str(err["code"])
                message = str(err.get("message", ""))
            except (ValueError, KeyError, TypeError):
                pass
            raise SecondRejected(code, message, resp.status_code)
        return resp

    async def post_task(self, item: dict[str, Any]) -> dict[str, Any]:
        """POST one work item. The project goes out only through `routed_project` (confidence gate);
        the item keeps the raw tagged project for /show."""
        text = source_text(item)
        event = event_task_body(item) if item.get("kind") == "event" else None
        body = {
            "title": event["title"] if event else _title(item),
            "body": event["body"] if event else text,
            "project": routed_project(item),
            "urgency": "medium",
            "source": {
                "client": "clawdia",
                "item_id": item["id"],
                "kind": item["kind"],
                "captured_at": item["created_at"],
                "text_excerpt": text[:EXCERPT_CHARS],
            },
        }
        fields = item.get("fields")
        if event:
            deadline = event["deadline"]
        else:
            deadline = fields.get("deadline") if isinstance(fields, dict) else None
        if isinstance(deadline, str) and deadline:
            body["deadline"] = deadline
        try:
            resp = await self._request(
                "POST", "/v1/tasks", headers={"Idempotency-Key": idempotency_key(item)}, json=body
            )
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            data = resp.json()
            return {"id": int(data["id"]), "project": data.get("project")}
        except (ValueError, KeyError, TypeError) as exc:
            raise SecondUnavailable("malformed response from /v1/tasks") from exc

    async def patch_task(
        self, second_id: int, *, deadline: str | None = None, project: str | None = None
    ) -> dict[str, Any]:
        """PATCH a task this client created. Only the given keys are sent."""
        body = {k: v for k, v in (("deadline", deadline), ("project", project)) if v is not None}
        if not body:
            raise ValueError("patch_task needs a deadline or a project")
        try:
            resp = await self._request("PATCH", f"/v1/tasks/{int(second_id)}", json=body)
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            data = resp.json()
            return {"id": int(data["id"]), "project": data.get("project"), "deadline": data.get("deadline")}
        except (ValueError, KeyError, TypeError) as exc:
            raise SecondUnavailable("malformed response from PATCH /v1/tasks") from exc

    async def post_job(self, item: dict[str, Any], images: list[tuple[str, bytes]], lang: str) -> dict[str, Any]:
        """POST one vacancy to the triage endpoint. `images` are (mime, bytes), already capped by the caller."""
        parts = [source_text(item)]
        if (item.get("ocr_text") or "").strip():
            parts.append(item["ocr_text"])
        urls = [u for u in item.get("urls") or [] if str(u).lower().startswith(("http://", "https://")) and len(u) <= 2000]
        created = item["created_at"]
        body = {
            "source": {"client": "clawdia", "item_id": item["id"],
                       "captured_at": created if isinstance(created, str) else created.isoformat()},
            "text": "\n".join(p for p in parts if p)[:20000],
            "urls": list(dict.fromkeys(urls))[:10],
            "images": [{"mime": mime, "data_b64": base64.b64encode(data).decode("ascii")} for mime, data in images],
            "lang": lang if lang in ("ru", "en") else "other",
        }
        try:
            resp = await self._request("POST", "/v1/jobs", headers={"Idempotency-Key": job_idempotency_key(item)}, json=body)
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            data = resp.json()
            return {"id": int(data["id"]), "status": data.get("status")}
        except (ValueError, KeyError, TypeError) as exc:
            raise SecondUnavailable("malformed response from /v1/jobs") from exc

    async def get_job(self, intake_id: int) -> dict[str, Any]:
        try:
            resp = await self._request("GET", f"/v1/jobs/{int(intake_id)}")
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            data = resp.json()
            data["status"]
            return data
        except (ValueError, KeyError, TypeError) as exc:
            raise SecondUnavailable("malformed response from GET /v1/jobs") from exc

    async def list_jobs(self, since: datetime, limit: int = 50) -> list[dict[str, Any]]:
        """Vacancy intakes of this client captured since `since` (newest first), each with `item_id`."""
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        query = urlencode({"since": since.astimezone(timezone.utc).isoformat(), "limit": max(1, min(int(limit), 100))})
        try:
            resp = await self._request("GET", f"/v1/jobs?{query}")
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            jobs = resp.json()["jobs"]
            if not isinstance(jobs, list):
                raise TypeError("jobs is not a list")
            return [j for j in jobs if isinstance(j, dict)]
        except (ValueError, KeyError, TypeError) as exc:
            raise SecondUnavailable("malformed response from GET /v1/jobs") from exc

    async def patch_job(self, intake_id: int, decision: str | None, note: str | None) -> dict[str, Any]:
        """Record Elena's decision ("apply", "skip" or None to clear). The decision key is always sent."""
        body: dict[str, Any] = {"decision": decision}
        if note is not None:
            body["note"] = note
        try:
            resp = await self._request("PATCH", f"/v1/jobs/{int(intake_id)}", json=body)
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            return resp.json()
        except ValueError as exc:
            raise SecondUnavailable("malformed response from PATCH /v1/jobs") from exc

    async def read(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """One read tool of Second's MCP server, served in-process by second-api: `POST /v1/read/{tool}`
        with the arguments as the body. Returns the tool's JSON unchanged (third-party text sits under
        `untrusted`). 400 bad_arguments and 404 unknown_tool/not_found raise SecondRejected."""
        try:
            resp = await self._request("POST", f"/v1/read/{tool}", json=arguments)
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise TypeError("not an object")
            return data
        except (ValueError, TypeError) as exc:
            raise SecondUnavailable(f"malformed response from /v1/read/{tool}") from exc

    async def pull_transcripts(self) -> dict[str, Any]:
        """Ask Second to fetch new call transcripts now: `POST /v1/transcripts/pull`. Returns the JSON
        (`status` is ok, cooldown or busy). 403 forbidden and 5xx/429/network raise like the other calls."""
        try:
            resp = await self._request("POST", "/v1/transcripts/pull")
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise TypeError("not an object")
            return data
        except (ValueError, TypeError) as exc:
            raise SecondUnavailable("malformed response from /v1/transcripts/pull") from exc

    async def refresh_sources(self) -> dict[str, Any]:
        """Ask Second to bring its cheap sources up to date: `POST /v1/sources/refresh` (bounded on the
        server). Returns the JSON (refreshed, skipped, failed, timed_out). Errors raise like the other calls."""
        try:
            resp = await self._request("POST", "/v1/sources/refresh", timeout=REFRESH_TIMEOUT_S)
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise TypeError("not an object")
            return data
        except (ValueError, TypeError) as exc:
            raise SecondUnavailable("malformed response from /v1/sources/refresh") from exc

    async def get_ops_health(self) -> dict[str, Any]:
        try:
            resp = await self._request("GET", "/v1/health/ops")
        except SecondRejected as exc:
            if exc.status in (401, 403):
                raise SecondAuthFailed(exc.code) from exc
            raise
        try:
            return resp.json()
        except ValueError as exc:
            raise SecondUnavailable("malformed response from /v1/health/ops") from exc

    async def get_context(self, etag: str | None) -> tuple[dict[str, Any], str] | None:
        headers = {"If-None-Match": etag} if etag else None
        resp = await self._request("GET", "/v1/context", headers=headers)
        if resp.status_code == 304:
            return None
        try:
            return resp.json(), resp.headers.get("ETag", "")
        except ValueError as exc:
            raise SecondUnavailable("malformed response from /v1/context") from exc

    async def heartbeat(self, stats: dict[str, Any]) -> None:
        await self._request("POST", "/v1/heartbeat", json={"version": __version__, "stats": stats})


Notify = Callable[[int, int, str], Awaitable[None]]


def _status(store: Store, item_id: int) -> str | None:
    item = store.get_item(item_id)
    return item["status"] if item else None


async def _post(store: Store, client: SecondClient, item: dict[str, Any]) -> dict[str, Any]:
    """Post an item; a 422 unknown_project (renamed or deleted project) is retried once at once
    without a project, so the task lands in Second's Inbox instead of stranding here."""
    try:
        return await client.post_task(item)
    except SecondRejected as exc:
        dropped = routed_project(item)
        if exc.code != "unknown_project" or dropped is None:
            raise
    store.add_event(item["id"], "project_dropped", {"project": dropped})
    return await client.post_task({**item, "project": None})


async def drain_outbox(
    store: Store,
    client: SecondClient,
    now: datetime,
    notify: Notify,
) -> int:
    """File routing work items in Second, oldest first. Returns how many were filed.

    Work items go out at any hour (spec 6.9); quiet days gate only reminders and the stuck
    warning. One SecondUnavailable stops the drain: Second is down or
    throttling, so the rest would fail the same way. A 401/403 is systemic too:
    items stay `routing` with route_error="unauthorized" and the drain stops.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    filed = 0
    for item in store.outbox():
        item_id = item["id"]
        if _status(store, item_id) != "routing":  # deleted or retagged since the snapshot was taken
            continue
        try:
            result = await _post(store, client, item)
        except SecondRejected as exc:
            if _status(store, item_id) == "routing":
                store.update_item(item_id, status="new", route_error=exc.code)
            store.add_event(item_id, "error", {"code": exc.code})
            continue
        except SecondAuthFailed:
            if _status(store, item_id) == "routing":
                store.update_item(item_id, route_error="unauthorized")
            store.add_event(item_id, "error", {"code": "unauthorized"})
            break
        except SecondUnavailable:
            if _status(store, item_id) == "routing":
                store.update_item(item_id, route_error="unavailable")
            break
        if _status(store, item_id) == "routing":
            store.update_item(
                item_id, status="filed", filed_ref=str(result["id"]), filed_at=now, route_error=None
            )
            store.add_event(item_id, "filed", {"second_id": result["id"], "project": result["project"]})
            filed += 1
        else:  # the owner changed the item while the post was in flight: keep their status
            store.add_event(item_id, "routed", {"second_id": result["id"], "project": result["project"]})
        try:
            await notify(item_id, result["id"], result["project"])
        except Exception:  # the item is already filed; a failed message must not stop the drain
            log.exception("notify failed for item %s", item_id)
    return filed


def _fresh_fields(store: Store, item_id: int, sent: dict[str, Any]) -> dict[str, Any] | None:
    """Re-read the item after a request: its fields without the patch we sent (a newer patch stays).
    None when the item is gone."""
    fresh = store.get_item(item_id)
    if fresh is None:
        return None
    fields = dict(fresh["fields"]) if isinstance(fresh.get("fields"), dict) else {}
    if fields.get("second_patch") == sent:
        fields.pop("second_patch", None)
    return fields


async def apply_pending_patches(store: Store, client: SecondClient,
                                outcomes: dict[int, str] | None = None) -> int:
    """Send corrections (`fields["second_patch"]`) for items already filed in Second, oldest first.

    Returns how many were applied. Second down or auth failure keeps the patch and stops;
    a rejection drops that patch (retrying cannot help) and moves on. Patches go out even for
    dismissed or done items: the Second task is still open and the correction was explicit.
    Fields are re-read after each request so edits made while it was in flight survive.

    `outcomes`, when given, is filled per item id: "patched", "rejected:<code>" or "pending"
    (kept for the next tick: Second down, auth failure, or not reached because of either).
    """
    outcomes = {} if outcomes is None else outcomes
    candidates = [
        it for it in store.list_items(limit=1_000_000)
        if isinstance(it.get("fields"), dict) and isinstance(it["fields"].get("second_patch"), dict)
        and str(it.get("filed_ref") or "").isascii() and str(it.get("filed_ref") or "").isdigit()
    ]
    candidates.sort(key=lambda it: (it["created_at"], it["id"]))
    for item in candidates:
        outcomes[item["id"]] = "pending"
    applied = 0
    for item in candidates:
        item_id = item["id"]
        patch = item["fields"]["second_patch"]
        second_id = int(item["filed_ref"])
        deadline, project = patch.get("deadline"), patch.get("project")
        try:
            result = await client.patch_task(second_id, deadline=deadline, project=project)
        except ValueError:  # empty patch: nothing to send
            outcomes[item_id] = "rejected:empty"
            fields = _fresh_fields(store, item_id, patch)
            if fields is not None:
                store.update_item(item_id, fields=fields)
            continue
        except SecondRejected as exc:
            outcomes[item_id] = f"rejected:{exc.code}"
            fields = _fresh_fields(store, item_id, patch)
            if fields is not None:
                store.update_item(item_id, fields=fields, route_error=exc.code)
                store.add_event(item_id, "error", {"code": exc.code})
            continue
        except SecondAuthFailed:
            store.update_item(item_id, route_error="unauthorized")
            break
        except SecondUnavailable:
            store.update_item(item_id, route_error="unavailable")
            break
        outcomes[item_id] = "patched"
        fields = _fresh_fields(store, item_id, patch)
        if fields is None:
            continue
        if result.get("deadline"):
            fields["deadline"] = result["deadline"]
        cols: dict[str, Any] = {"fields": fields, "route_error": None}
        if result.get("project"):
            cols["project"] = result["project"]
        store.update_item(item_id, **cols)
        store.add_event(item_id, "patched", {"keys": sorted(patch)})
        applied += 1
    return applied
