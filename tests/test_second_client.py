import json
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from clawdia.second_client import SecondAuthFailed, SecondClient, SecondRejected, SecondUnavailable, drain_outbox
from clawdia.storage import Store

ITEM = {"id": 42, "title": "Ask Bruno", "text": "напомнить Bruno про инвойс", "project": "Acme", "confidence": 0.9,
        "kind": "task", "created_at": "2026-10-30T10:00:00Z"}
KEY_42 = "clawdia-42-1793354400000000"  # 2026-10-30T10:00:00Z in epoch microseconds
OK = {"id": 2601, "project": "Acme", "created_at": "2026-10-30T10:00:01Z"}
WORKDAY = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)  # Monday
SATURDAY = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def s(tmp_path):
    store = Store(tmp_path / "c.sqlite")
    store.ensure_schema()
    yield store
    store.close()


def transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t")


def client_for(handler):
    return SecondClient("http://t", "tok", http=transport(handler))


@pytest.mark.asyncio
async def test_post_task_sends_idempotency_key_and_parses_id():
    seen = {}

    def h(req):
        seen["key"] = req.headers["idempotency-key"]
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(201, json=OK)

    r = await client_for(h).post_task(ITEM)
    assert r["id"] == 2601 and seen["key"] == KEY_42 and seen["auth"] == "Bearer tok"


@pytest.mark.asyncio
async def test_post_task_body_matches_contract():
    seen = {}

    def h(req):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=OK)

    r = await client_for(h).post_task(ITEM)
    assert r == {"id": 2601, "project": "Acme"}
    assert seen["url"] == "http://t/v1/tasks"
    b = seen["body"]
    assert b["title"] == "Ask Bruno" and b["project"] == "Acme" and b["urgency"] == "medium"
    assert b["body"] == ITEM["text"]
    assert b["source"] == {
        "client": "clawdia", "item_id": 42, "kind": "task",
        "captured_at": "2026-10-30T10:00:00Z", "text_excerpt": ITEM["text"],
    }


@pytest.mark.asyncio
async def test_post_task_title_fallbacks_and_truncation():
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(201, json=OK)

    c = client_for(h)
    long_text = "x" * 400
    await c.post_task({**ITEM, "title": None, "text": long_text})
    await c.post_task({**ITEM, "title": "", "text": None})
    await c.post_task({**ITEM, "title": "t" * 900})
    await c.post_task({**ITEM, "project": None, "text": long_text})
    assert bodies[0]["title"] == "x" * 80
    assert bodies[0]["source"]["text_excerpt"] == "x" * 300 and bodies[0]["body"] == long_text
    assert bodies[1]["title"] == "(untitled)" and bodies[1]["body"] == "" and bodies[1]["source"]["text_excerpt"] == ""
    assert len(bodies[2]["title"]) == 500
    assert bodies[3]["project"] is None


@pytest.mark.asyncio
async def test_post_task_maps_errors():
    with pytest.raises(SecondRejected) as e:
        await client_for(lambda r: httpx.Response(422, json={"error": {"code": "unknown_project", "message": "x"}})).post_task(ITEM)
    assert e.value.code == "unknown_project" and e.value.message == "x"
    with pytest.raises(SecondUnavailable):
        await client_for(lambda r: httpx.Response(503)).post_task(ITEM)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 502])
async def test_throttle_and_5xx_are_unavailable(status):
    with pytest.raises(SecondUnavailable):
        await client_for(lambda r: httpx.Response(status, headers={"Retry-After": "5"})).post_task(ITEM)


@pytest.mark.asyncio
async def test_4xx_without_json_falls_back_to_http_code():
    with pytest.raises(SecondRejected) as e:
        await client_for(lambda r: httpx.Response(409, text="nope")).post_task(ITEM)
    assert e.value.code == "http_409"
    with pytest.raises(SecondAuthFailed) as a:
        await client_for(lambda r: httpx.Response(401, json={"error": {"code": "unauthorized", "message": "bad"}})).post_task(ITEM)
    assert isinstance(a.value, SecondUnavailable)


@pytest.mark.asyncio
async def test_connection_error_and_timeout_are_unavailable():
    def boom(req):
        raise httpx.ConnectError("refused")

    def slow(req):
        raise httpx.ReadTimeout("slow")

    for h in (boom, slow):
        with pytest.raises(SecondUnavailable):
            await client_for(h).post_task(ITEM)
        with pytest.raises(SecondUnavailable):
            await client_for(h).get_context(None)
        with pytest.raises(SecondUnavailable):
            await client_for(h).heartbeat({})


@pytest.mark.asyncio
async def test_get_context_200_304_and_etag_header():
    seen = []

    def h(req):
        seen.append(req.headers.get("if-none-match"))
        if req.headers.get("if-none-match") == '"abc"':
            return httpx.Response(304)
        return httpx.Response(200, json={"projects": []}, headers={"ETag": '"abc"'})

    c = client_for(h)
    assert await c.get_context(None) == ({"projects": []}, '"abc"')
    assert await c.get_context('"abc"') is None
    assert seen == [None, '"abc"']


@pytest.mark.asyncio
async def test_get_context_error_mapping():
    with pytest.raises(SecondUnavailable):
        await client_for(lambda r: httpx.Response(500)).get_context(None)
    with pytest.raises(SecondRejected) as e:
        await client_for(lambda r: httpx.Response(401, json={"error": {"code": "unauthorized", "message": "m"}})).get_context(None)
    assert e.value.code == "unauthorized"


@pytest.mark.asyncio
async def test_heartbeat_posts_version_and_stats():
    seen = {}

    def h(req):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(204)

    assert await client_for(h).heartbeat({"outbox": 3}) is None
    from clawdia import __version__
    assert seen["url"] == "http://t/v1/heartbeat" and seen["auth"] == "Bearer tok"
    assert seen["body"] == {"version": __version__, "stats": {"outbox": 3}}
    with pytest.raises(SecondUnavailable):
        await client_for(lambda r: httpx.Response(503)).heartbeat({})


@pytest.mark.asyncio
async def test_aclose_closes_only_its_own_client():
    own = SecondClient("http://127.0.0.1:1", "tok")
    await own.aclose()
    assert own._http.is_closed
    shared = transport(lambda r: httpx.Response(204))
    c = SecondClient("http://t", "tok", http=shared)
    await c.aclose()
    assert not shared.is_closed
    await shared.aclose()


@pytest.mark.asyncio
async def test_task_captured_sunday_2300_is_posted_at_once(s):
    from zoneinfo import ZoneInfo
    sunday_2300 = datetime(2026, 11, 1, 23, 0, tzinfo=ZoneInfo("Europe/Lisbon"))
    i = s.create_item(kind="task", status="routing", title="Ask Bruno", text="x", project="Acme", route_error="unavailable")
    notify = AsyncMock()
    assert await drain_outbox(s, client_for(lambda r: httpx.Response(201, json=OK)), sunday_2300, notify) == 1
    it = s.get_item(i)
    assert it["status"] == "filed" and it["filed_ref"] == "2601" and it["route_error"] is None
    assert notify.await_args.args == (i, 2601, "Acme")


@pytest.mark.asyncio
async def test_drain_posts_on_saturday_and_before_0700(s):
    from zoneinfo import ZoneInfo
    s.create_item(kind="task", status="routing", title="a")
    client = client_for(lambda r: httpx.Response(201, json=OK))
    assert await drain_outbox(s, client, SATURDAY, AsyncMock()) == 1
    s.create_item(kind="task", status="routing", title="b")
    assert await drain_outbox(s, client, datetime(2026, 11, 2, 0, 5, tzinfo=ZoneInfo("Europe/Lisbon")), AsyncMock()) == 1


@pytest.mark.asyncio
async def test_event_is_posted_as_dated_task(s):
    seen = {}

    def h(r):
        seen["body"] = json.loads(r.content)
        return httpx.Response(201, json=OK)

    i = s.create_item(kind="event", status="routing", title="Dinner", text="dinner with Bruno",
                      fields={"date": "2026-11-12", "time": "19:30", "place": "Lisbon"})
    assert await drain_outbox(s, client_for(h), SATURDAY, AsyncMock()) == 1
    assert seen["body"]["title"] == "Event: Dinner" and seen["body"]["deadline"] == "2026-11-12"
    assert seen["body"]["body"] == "19:30 · Lisbon\ndinner with Bruno"
    assert seen["body"]["source"]["kind"] == "event" and s.get_item(i)["status"] == "filed"


@pytest.mark.asyncio
async def test_event_without_date_posts_without_deadline(s):
    seen = {}

    def h(r):
        seen["body"] = json.loads(r.content)
        return httpx.Response(201, json=OK)

    s.create_item(kind="event", status="routing", title="Party", text="somewhere", fields={"date": "next week", "place": "Porto"})
    assert await drain_outbox(s, client_for(h), SATURDAY, AsyncMock()) == 1
    assert "deadline" not in seen["body"] and seen["body"]["body"] == "Porto\nsomewhere"


def test_event_explicit_deadline_wins_over_date():
    from clawdia.second_client import event_task_body
    body = event_task_body({"title": "P", "text": "t", "fields": {"date": "2026-11-12", "deadline": "2026-11-20"}})
    assert body["deadline"] == "2026-11-20"
    assert event_task_body({"title": "P", "text": "t", "fields": {"date": "2026-11-12", "deadline": "bad"}})["deadline"] == "2026-11-12"


def test_event_task_body_omits_missing_time_and_place():
    from clawdia.second_client import event_task_body
    body = event_task_body({"title": "Party", "text": "bring wine", "fields": {"date": "2026-02-30"}})
    assert body == {"title": "Event: Party", "deadline": None, "body": "bring wine"}
    assert event_task_body({"title": "P", "text": "t", "fields": {"date": "2026-11-12", "time": "10:00"}})["body"] == "10:00\nt"


@pytest.mark.asyncio
async def test_jobs_never_enter_task_outbox(s):
    s.create_item(kind="job", status="routing", title="Role")
    posts = []
    client = client_for(lambda r: posts.append(r) or httpx.Response(201, json=OK))
    assert s.outbox() == [] and await drain_outbox(s, client, WORKDAY, AsyncMock()) == 0 and posts == []


@pytest.mark.asyncio
async def test_drain_naive_now_raises(s):
    with pytest.raises(ValueError):
        await drain_outbox(s, client_for(lambda r: httpx.Response(201, json=OK)), datetime(2026, 11, 2, 8, 0), AsyncMock())


@pytest.mark.asyncio
async def test_drain_stops_when_second_is_down(s):
    a = s.create_item(kind="task", status="routing", title="a")
    b = s.create_item(kind="task", status="routing", title="b")
    calls = []

    def h(r):
        calls.append(1)
        return httpx.Response(503)

    notify = AsyncMock()
    assert await drain_outbox(s, client_for(h), WORKDAY, notify) == 0
    assert len(calls) == 1 and s.get_item(a)["route_error"] == "unavailable" and s.get_item(b)["status"] == "routing"
    assert s.get_item(b)["route_error"] is None
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_drain_rejected_item_returns_to_new_and_drain_continues(s):
    a = s.create_item(kind="task", status="routing", title="a")
    b = s.create_item(kind="event", status="routing", title="b")
    seen = []

    def h(r):
        seen.append(json.loads(r.content)["source"]["item_id"])
        if seen[-1] == a:
            return httpx.Response(400, json={"error": {"code": "validation_error", "message": "x"}})
        return httpx.Response(201, json=OK)

    notify = AsyncMock()
    assert await drain_outbox(s, client_for(h), WORKDAY, notify) == 1
    ia = s.get_item(a)
    assert ia["status"] == "new" and ia["route_error"] == "validation_error"
    assert s.get_item(b)["status"] == "filed" and seen == [a, b]  # no retry for other rejections
    events = s._conn.execute("SELECT type, payload FROM events WHERE item_id = ?", (a,)).fetchall()
    assert [(e["type"], json.loads(e["payload"])) for e in events] == [("error", {"code": "validation_error"})]
    assert s.outbox() == []


def _events(s, item_id):
    rows = s._conn.execute("SELECT type, payload FROM events WHERE item_id = ? ORDER BY id", (item_id,)).fetchall()
    return [(e["type"], json.loads(e["payload"])) for e in rows]


@pytest.mark.asyncio
async def test_unknown_project_is_retried_once_without_project(s):
    a = s.create_item(kind="task", status="routing", title="a", project="Old name", confidence=0.9)
    sent = []

    def h(r):
        body = json.loads(r.content)
        sent.append((body["project"], r.headers["idempotency-key"]))
        if body["project"] is not None:
            return httpx.Response(422, json={"error": {"code": "unknown_project", "message": "unknown project: Old name"}})
        return httpx.Response(201, json={"id": 77, "project": None})

    notify = AsyncMock()
    assert await drain_outbox(s, client_for(h), WORKDAY, notify) == 1
    assert [p for p, _ in sent] == ["Old name", None] and sent[0][1] == sent[1][1]  # same key on the retry
    ia = s.get_item(a)
    assert ia["status"] == "filed" and ia["filed_ref"] == "77" and ia["project"] == "Old name"  # raw kept for /show
    assert _events(s, a) == [("project_dropped", {"project": "Old name"}), ("filed", {"second_id": 77, "project": None})]
    notify.assert_awaited_once_with(a, 77, None)


@pytest.mark.asyncio
async def test_unknown_project_retry_rejected_again_parks_the_item(s):
    a = s.create_item(kind="task", status="routing", title="a", project="Old name", confidence=0.9)
    calls = []

    def h(r):
        calls.append(json.loads(r.content)["project"])
        return httpx.Response(422, json={"error": {"code": "unknown_project", "message": "x"}})

    assert await drain_outbox(s, client_for(h), WORKDAY, AsyncMock()) == 0
    assert calls == ["Old name", None]  # exactly one retry
    assert s.get_item(a)["status"] == "new" and s.get_item(a)["route_error"] == "unknown_project"
    assert [t for t, _ in _events(s, a)] == ["project_dropped", "error"]


@pytest.mark.asyncio
async def test_unknown_project_retry_that_finds_second_down_keeps_routing(s):
    a = s.create_item(kind="task", status="routing", title="a", project="Old name", confidence=0.9)
    responses = iter([httpx.Response(422, json={"error": {"code": "unknown_project", "message": "x"}}), httpx.Response(503)])
    assert await drain_outbox(s, client_for(lambda r: next(responses)), WORKDAY, AsyncMock()) == 0
    assert s.get_item(a)["status"] == "routing" and s.get_item(a)["route_error"] == "unavailable"


# -- project confidence gate (spec 6.2 step 4) ------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("confidence,expected", [(0.59, None), (0.6, "Acme"), (0.95, "Acme"), (None, None), (0.0, None)])
async def test_project_is_posted_only_at_confidence_06_or_more(confidence, expected):
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(201, json=OK)

    await client_for(h).post_task({**ITEM, "confidence": confidence})
    assert bodies[0]["project"] == expected


@pytest.mark.asyncio
async def test_low_confidence_item_keeps_its_raw_project(s):
    a = s.create_item(kind="task", status="routing", title="a", project="Acme", confidence=0.4)
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(201, json={"id": 5, "project": None})

    await drain_outbox(s, client_for(h), WORKDAY, AsyncMock())
    assert bodies[0]["project"] is None and s.get_item(a)["project"] == "Acme"
    assert [t for t, _ in _events(s, a)] == ["filed"]  # no unknown_project dance for a gated project


# -- source text for voice and caption-only items ---------------------------

@pytest.mark.asyncio
async def test_voice_only_item_sends_its_transcript():
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(201, json=OK)

    long = "please send the invoice " * 20
    item = {**ITEM, "text": None, "transcript": long, "forward_from": {"name": "Bruno", "chat_title": None, "link": None}}
    await client_for(h).post_task(item)
    b = bodies[0]
    assert b["body"] == f"Transcript: {long}\nForwarded from: Bruno"
    assert b["source"]["text_excerpt"] == b["body"][:300] and b["source"]["text_excerpt"].startswith("Transcript: please")


@pytest.mark.asyncio
async def test_caption_only_item_and_part_order():
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(201, json=OK)

    c = client_for(h)
    await c.post_task({**ITEM, "text": None, "caption": "invoice photo for Bruno"})
    await c.post_task({**ITEM, "text": "t", "caption": "c", "transcript": "v", "forward_from": {"name": "Ann"}})
    await c.post_task({**ITEM, "title": None, "text": None, "caption": "  ", "transcript": "call Bob"})
    assert bodies[0]["body"] == "invoice photo for Bruno" == bodies[0]["source"]["text_excerpt"]
    assert bodies[1]["body"] == "t\nc\nTranscript: v\nForwarded from: Ann"
    assert bodies[2]["title"] == "call Bob" and bodies[2]["body"] == "Transcript: call Bob"


# -- idempotency key -----------------------------------------------------------

def test_idempotency_key_is_stable_per_row_and_unique_per_created_at():
    from clawdia.second_client import idempotency_key
    assert idempotency_key(ITEM) == KEY_42 == idempotency_key({**ITEM, "created_at": "2026-10-30T10:00:00.000000+00:00"})
    assert idempotency_key({**ITEM, "created_at": "2026-10-30T11:00:00+01:00"}) == KEY_42  # same instant
    assert idempotency_key({**ITEM, "created_at": "2026-10-30T10:00:00.000001+00:00"}) == "clawdia-42-1793354400000001"
    assert idempotency_key({**ITEM, "created_at": "2026-10-30T10:00:00"}) == KEY_42  # naive is UTC


@pytest.mark.asyncio
async def test_reused_id_after_a_db_reset_gets_a_new_key(tmp_path):
    keys = []

    def h(req):
        keys.append(req.headers["idempotency-key"])
        return httpx.Response(201, json=OK)

    for n in range(2):  # the same item id in two databases, created at different times
        store = Store(tmp_path / f"{n}.sqlite"); store.ensure_schema()
        i = store.create_item(kind="task", status="routing", title="x",
                              created_at=datetime(2026, 10, 30 + n, 9, 0, tzinfo=timezone.utc))
        await drain_outbox(store, client_for(h), WORKDAY, AsyncMock())
        await drain_outbox(store, client_for(h), WORKDAY, AsyncMock())  # filed: no second post
        store.close()
        assert i == 1
    assert keys == ["clawdia-1-1793350800000000", "clawdia-1-1793437200000000"]


@pytest.mark.asyncio
async def test_drain_adds_filed_event_with_second_id(s):
    i = s.create_item(kind="task", status="routing", title="a")
    await drain_outbox(s, client_for(lambda r: httpx.Response(201, json=OK)), WORKDAY, AsyncMock())
    events = s._conn.execute("SELECT type, payload FROM events WHERE item_id = ?", (i,)).fetchall()
    assert [e["type"] for e in events] == ["filed"]
    assert json.loads(events[0]["payload"])["second_id"] == 2601


@pytest.mark.asyncio
async def test_drain_notify_failure_does_not_undo_filing(s):
    a = s.create_item(kind="task", status="routing", title="a")
    b = s.create_item(kind="task", status="routing", title="b")
    notify = AsyncMock(side_effect=RuntimeError("telegram down"))
    n = await drain_outbox(s, client_for(lambda r: httpx.Response(201, json=OK)), WORKDAY, notify)
    assert n == 2 and s.get_item(a)["status"] == "filed" and s.get_item(b)["status"] == "filed"


@pytest.mark.asyncio
async def test_injected_client_without_base_url_uses_constructor_url():
    seen = {}

    def h(req):
        seen["url"] = str(req.url)
        return httpx.Response(201, json=OK)

    http = httpx.AsyncClient(transport=httpx.MockTransport(h))
    await SecondClient("http://127.0.0.1:8091/", "tok", http=http).post_task(ITEM)
    await http.aclose()
    assert seen["url"] == "http://127.0.0.1:8091/v1/tasks"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_drain_auth_failure_keeps_items_routing_and_stops(s, status):
    a = s.create_item(kind="task", status="routing", title="a")
    b = s.create_item(kind="task", status="routing", title="b")
    calls = []

    def h(r):
        calls.append(1)
        return httpx.Response(status, json={"error": {"code": "unauthorized", "message": "bad token"}})

    notify = AsyncMock()
    assert await drain_outbox(s, client_for(h), WORKDAY, notify) == 0
    assert len(calls) == 1
    assert s.get_item(a)["status"] == "routing" and s.get_item(a)["route_error"] == "unauthorized"
    assert s.get_item(b)["status"] == "routing" and s.get_item(b)["route_error"] is None
    events = s._conn.execute("SELECT type, payload FROM events WHERE item_id = ?", (a,)).fetchall()
    assert [(e["type"], json.loads(e["payload"])) for e in events] == [("error", {"code": "unauthorized"})]
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_drain_skips_item_deleted_while_another_post_is_in_flight(s):
    import asyncio
    a = s.create_item(kind="task", status="routing", title="a", text="a")
    b = s.create_item(kind="task", status="routing", title="b", text="b")
    gate, started, posted = asyncio.Event(), asyncio.Event(), []

    async def post(item):
        posted.append(item["id"])
        started.set()
        await gate.wait()
        return {"id": 9000 + item["id"], "project": None}

    client = SimpleNamespace(post_task=post)
    notify = AsyncMock()
    task = asyncio.create_task(drain_outbox(s, client, WORKDAY, notify))
    await started.wait()
    s.update_item(b, status="dismissed")  # the owner deletes B while A is posting
    gate.set()
    assert await task == 1
    assert posted == [a] and s.get_item(b)["status"] == "dismissed" and s.get_item(a)["status"] == "filed"


@pytest.mark.asyncio
async def test_drain_keeps_owner_status_when_item_changes_during_its_own_post(s):
    import asyncio
    a = s.create_item(kind="task", status="routing", title="a", text="a")
    gate, started = asyncio.Event(), asyncio.Event()

    async def post(item):
        started.set()
        await gate.wait()
        return {"id": 9001, "project": "Acme"}

    notify = AsyncMock()
    task = asyncio.create_task(drain_outbox(s, SimpleNamespace(post_task=post), WORKDAY, notify))
    await started.wait()
    s.update_item(a, status="dismissed")
    gate.set()
    await task
    assert s.get_item(a)["status"] == "dismissed" and s.get_item(a)["filed_ref"] is None
    ev = s._conn.execute("SELECT type, payload FROM events WHERE item_id=?", (a,)).fetchall()
    assert [e["type"] for e in ev] == ["routed"] and json.loads(ev[0]["payload"])["second_id"] == 9001
    assert notify.await_args.args == (a, 9001, "Acme")


@pytest.mark.asyncio
async def test_post_task_sends_deadline():
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(201, json=OK)

    c = client_for(h)
    await c.post_task({**ITEM, "fields": {"deadline": "2026-11-09"}})
    await c.post_task({**ITEM, "fields": {"deadline": ""}})
    await c.post_task({**ITEM, "fields": {}})
    assert bodies[0]["deadline"] == "2026-11-09"
    assert "deadline" not in bodies[1] and "deadline" not in bodies[2]


@pytest.mark.asyncio
async def test_patch_task_sends_only_given_keys():
    seen = {}

    def h(req):
        seen["method"], seen["url"], seen["body"] = req.method, str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"id": 2601, "project": "Acme", "deadline": "2026-11-09"})

    r = await client_for(h).patch_task(2601, deadline="2026-11-09")
    assert r == {"id": 2601, "project": "Acme", "deadline": "2026-11-09"}
    assert seen == {"method": "PATCH", "url": "http://t/v1/tasks/2601", "body": {"deadline": "2026-11-09"}}
    with pytest.raises(ValueError):
        await client_for(h).patch_task(2601)


@pytest.mark.asyncio
async def test_patch_task_maps_errors():
    def status(code, body=None):
        return client_for(lambda req: httpx.Response(code, json=body or {}))

    with pytest.raises(SecondRejected) as e:
        await status(404, {"error": {"code": "not_found", "message": "x"}}).patch_task(1, deadline="2026-11-09")
    assert e.value.code == "not_found"
    with pytest.raises(SecondUnavailable):
        await status(503).patch_task(1, deadline="2026-11-09")
    with pytest.raises(SecondAuthFailed):
        await status(401, {"error": {"code": "unauthorized"}}).patch_task(1, deadline="2026-11-09")


def filed_item(s, ref="2601", patch=None, project="Acme", created="2026-10-30T10:00:00+00:00"):
    fields = {"second_patch": patch} if patch is not None else {}
    return s.create_item(kind="task", status="filed", title="x", filed_ref=ref, project=project,
                         fields=fields, created_at=created)


@pytest.mark.asyncio
async def test_pending_patch_applied_and_cleared(s):
    from clawdia.second_client import apply_pending_patches

    i = filed_item(s, patch={"deadline": "2026-11-09", "project": "Orion"})
    seen = []

    def h(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"id": 2601, "project": "Orion", "deadline": "2026-11-09"})

    assert await apply_pending_patches(s, client_for(h)) == 1
    item = s.get_item(i)
    assert seen == [{"deadline": "2026-11-09", "project": "Orion"}]
    assert "second_patch" not in item["fields"] and item["fields"]["deadline"] == "2026-11-09"
    assert item["project"] == "Orion"
    with s._lock:
        types = [r["type"] for r in s._conn.execute("SELECT type FROM events WHERE item_id = ?", (i,))]
    assert "patched" in types
    assert await apply_pending_patches(s, client_for(h)) == 0 and len(seen) == 1


@pytest.mark.asyncio
async def test_pending_patch_kept_when_second_down(s):
    from clawdia.second_client import apply_pending_patches

    a = filed_item(s, patch={"deadline": "2026-11-09"}, created="2026-10-30T10:00:00+00:00")
    b = filed_item(s, ref="2602", patch={"deadline": "2026-11-10"}, created="2026-10-31T10:00:00+00:00")
    calls = []

    def h(req):
        calls.append(req.url.path)
        return httpx.Response(503)

    assert await apply_pending_patches(s, client_for(h)) == 0
    assert calls == ["/v1/tasks/2601"]  # stopped after the first failure
    assert s.get_item(a)["fields"]["second_patch"] == {"deadline": "2026-11-09"}
    assert s.get_item(a)["route_error"] == "unavailable"
    assert "second_patch" in s.get_item(b)["fields"]

    client = client_for(lambda req: httpx.Response(401, json={"error": {"code": "unauthorized"}}))
    assert await apply_pending_patches(s, client) == 0
    assert s.get_item(a)["route_error"] == "unauthorized"

    ok = client_for(lambda req: httpx.Response(200, json={"id": 2601, "project": "Acme", "deadline": "2026-11-09"}))
    assert await apply_pending_patches(s, ok) == 2
    assert s.get_item(a)["route_error"] is None and s.get_item(b)["route_error"] is None


@pytest.mark.asyncio
async def test_pending_patch_rejected_is_dropped_and_continues(s):
    from clawdia.second_client import apply_pending_patches

    a = filed_item(s, patch={"project": "Nope"}, created="2026-10-30T10:00:00+00:00")
    b = filed_item(s, ref="2602", patch={"deadline": "2026-11-10"}, created="2026-10-31T10:00:00+00:00")

    def h(req):
        if req.url.path.endswith("2601"):
            return httpx.Response(422, json={"error": {"code": "unknown_project", "message": "x"}})
        return httpx.Response(200, json={"id": 2602, "project": "Acme", "deadline": "2026-11-10"})

    assert await apply_pending_patches(s, client_for(h)) == 1
    assert "second_patch" not in s.get_item(a)["fields"] and s.get_item(a)["route_error"] == "unknown_project"
    assert "second_patch" not in s.get_item(b)["fields"]


@pytest.mark.asyncio
async def test_pending_patch_skips_non_numeric_filed_ref(s):
    from clawdia.second_client import apply_pending_patches

    filed_item(s, ref="abc", patch={"deadline": "2026-11-09"})
    assert await apply_pending_patches(s, client_for(lambda req: httpx.Response(500))) == 0


@pytest.mark.asyncio
async def test_pending_patch_keeps_edits_made_while_in_flight(s):
    import asyncio
    from clawdia.second_client import apply_pending_patches

    i = filed_item(s, patch={"deadline": "2026-11-09"})
    gate, started = asyncio.Event(), asyncio.Event()

    async def patch(second_id, **kw):
        started.set()
        await gate.wait()
        return {"id": second_id, "project": "Acme", "deadline": "2026-11-09"}

    task = asyncio.create_task(apply_pending_patches(s, SimpleNamespace(patch_task=patch)))
    await started.wait()
    s.update_item(i, fields={"second_patch": {"deadline": "2026-11-16"}, "reminders_off": True})
    gate.set()
    assert await task == 1
    f = s.get_item(i)["fields"]
    assert f["second_patch"] == {"deadline": "2026-11-16"} and f["reminders_off"] is True
    assert f["deadline"] == "2026-11-09"


@pytest.mark.asyncio
async def test_pending_patch_rejected_keeps_edits_made_while_in_flight(s):
    import asyncio
    from clawdia.second_client import apply_pending_patches

    i = filed_item(s, patch={"project": "Nope"})
    gate, started = asyncio.Event(), asyncio.Event()

    async def patch(second_id, **kw):
        started.set()
        await gate.wait()
        raise SecondRejected("unknown_project")

    task = asyncio.create_task(apply_pending_patches(s, SimpleNamespace(patch_task=patch)))
    await started.wait()
    s.update_item(i, fields={"second_patch": {"project": "Orion"}, "reminders_off": True})
    gate.set()
    await task
    f = s.get_item(i)["fields"]
    assert f == {"second_patch": {"project": "Orion"}, "reminders_off": True}


@pytest.mark.asyncio
async def test_pending_patch_still_sent_for_dismissed_item(s):
    from clawdia.second_client import apply_pending_patches

    i = filed_item(s, patch={"deadline": "2026-11-09"})
    s.update_item(i, status="dismissed")
    ok = client_for(lambda req: httpx.Response(200, json={"id": 2601, "project": "Acme", "deadline": "2026-11-09"}))
    assert await apply_pending_patches(s, ok) == 1
    assert s.get_item(i)["status"] == "dismissed" and "second_patch" not in s.get_item(i)["fields"]


@pytest.mark.asyncio
async def test_pending_patch_outcomes_per_item(s):
    from clawdia.second_client import apply_pending_patches

    a = filed_item(s, patch={"deadline": "2026-11-09"}, created="2026-10-30T10:00:00+00:00")
    b = filed_item(s, ref="2602", patch={"project": "Nope"}, created="2026-10-31T10:00:00+00:00")
    c = filed_item(s, ref="2603", patch={"deadline": "2026-11-10"}, created="2026-11-01T10:00:00+00:00")
    d = filed_item(s, ref="2604", patch={"deadline": "2026-11-11"}, created="2026-11-01T11:00:00+00:00")

    def h(req):
        if req.url.path.endswith("2601"):
            return httpx.Response(200, json={"id": 2601, "project": "Acme", "deadline": "2026-11-09"})
        if req.url.path.endswith("2602"):
            return httpx.Response(405)
        return httpx.Response(503)

    outcomes: dict[int, str] = {}
    assert await apply_pending_patches(s, client_for(h), outcomes) == 1
    assert outcomes == {a: "patched", b: "rejected:http_405", c: "pending", d: "pending"}


@pytest.mark.asyncio
async def test_pull_transcripts_posts_route():
    seen = {}

    def h(req):
        seen["method"], seen["path"], seen["auth"] = req.method, req.url.path, req.headers["authorization"]
        return httpx.Response(200, json={"status": "cooldown"})

    assert await client_for(h).pull_transcripts() == {"status": "cooldown"}
    assert seen == {"method": "POST", "path": "/v1/transcripts/pull", "auth": "Bearer tok"}
    with pytest.raises(SecondAuthFailed):
        await client_for(lambda r: httpx.Response(403, json={"error": {"code": "forbidden"}})).pull_transcripts()
    with pytest.raises(SecondUnavailable):
        await client_for(lambda r: httpx.Response(503, json={"error": {"code": "unavailable"}})).pull_transcripts()


@pytest.mark.asyncio
async def test_refresh_sources_posts_route():
    seen = {}

    def h(req):
        seen["method"], seen["path"], seen["auth"] = req.method, req.url.path, req.headers["authorization"]
        return httpx.Response(200, json={"refreshed": ["calendar"], "skipped": [], "failed": [], "timed_out": False})

    assert (await client_for(h).refresh_sources())["refreshed"] == ["calendar"]
    assert seen == {"method": "POST", "path": "/v1/sources/refresh", "auth": "Bearer tok"}
    with pytest.raises(SecondAuthFailed):
        await client_for(lambda r: httpx.Response(403, json={"error": {"code": "forbidden"}})).refresh_sources()
    with pytest.raises(SecondUnavailable):
        await client_for(lambda r: httpx.Response(503, json={"error": {"code": "unavailable"}})).refresh_sources()


@pytest.mark.asyncio
async def test_refresh_sources_uses_the_long_timeout():
    seen = {}

    def h(req):
        seen["timeout"] = req.extensions["timeout"]
        return httpx.Response(200, json={"refreshed": []})

    await client_for(h).refresh_sources()
    assert seen["timeout"]["read"] == 70.0
    await client_for(h).pull_transcripts()
    assert seen["timeout"]["read"] == 10.0
