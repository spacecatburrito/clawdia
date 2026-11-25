import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from clawdia.storage import Store


def make(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    s.ensure_schema()
    return s


def test_item_roundtrip_and_outbox(tmp_path):
    s = Store(tmp_path / "c.sqlite"); s.ensure_schema()
    i = s.create_item(kind="task", status="routing", title="Ask Bruno", text="напомнить Bruno", tg_message_id=5)
    assert s.get_item(i)["title"] == "Ask Bruno"
    assert [o["id"] for o in s.outbox()] == [i]
    s.update_item(i, status="filed", filed_ref="2601")
    assert s.outbox() == [] and s.get_item(i)["filed_ref"] == "2601"
    s.add_event(i, "filed", {"second_id": 2601})
    assert s.list_items(kind="task", status="filed")[0]["id"] == i


def test_reminders_due_and_snooze(tmp_path):
    s = Store(tmp_path / "c.sqlite"); s.ensure_schema()
    i = s.create_item(kind="event", status="new", title="Party")
    r = s.add_reminder(i, datetime(2026, 11, 7, 19, 0, tzinfo=timezone.utc))
    assert [x["id"] for x in s.due_reminders(datetime(2026, 11, 7, 19, 1, tzinfo=timezone.utc))] == [r]
    s.snooze_reminder(r, datetime(2026, 11, 8, 19, 0, tzinfo=timezone.utc))
    assert s.due_reminders(datetime(2026, 11, 7, 19, 1, tzinfo=timezone.utc)) == []


def test_schema_is_idempotent_and_backup_opens(tmp_path):
    s = Store(tmp_path / "c.sqlite"); s.ensure_schema(); s.ensure_schema()
    s.backup_to(tmp_path / "b.sqlite")
    assert sqlite3.connect(tmp_path / "b.sqlite").execute("select count(*) from items").fetchone()[0] == 0


def test_json_columns_decoded(tmp_path):
    s = make(tmp_path)
    i = s.create_item(
        kind="event", fields={"ask_when": "Sat 19:00"}, forward_from={"name": "Ann"},
        urls=["https://a.example"], media_paths=["a.jpg", "b.jpg"], media_group_id="g1",
    )
    it = s.get_item(i)
    assert it["fields"]["ask_when"] == "Sat 19:00"
    assert it["forward_from"]["name"] == "Ann"
    assert it["urls"] == ["https://a.example"]
    assert len(it["media_paths"]) == 2
    assert it["media_group_id"] == "g1"
    j = s.create_item(kind="idea")
    it2 = s.get_item(j)
    assert it2["fields"] is None and it2["forward_from"] is None
    assert it2["urls"] is None and it2["media_paths"] is None
    s.update_item(j, fields={"k": 1})
    assert s.get_item(j)["fields"] == {"k": 1}


def test_defaults_and_unknown_keys(tmp_path):
    s = make(tmp_path)
    i = s.create_item(kind="idea")
    it = s.get_item(i)
    assert it["status"] == "new"
    assert it["created_at"].endswith("+00:00")
    assert it["updated_at"] is not None
    with pytest.raises(ValueError):
        s.create_item(kind="idea", bogus=1)
    with pytest.raises(ValueError):
        s.update_item(i, bogus=1)
    assert s.get_item(9999) is None


def test_update_sets_updated_at(tmp_path):
    s = make(tmp_path)
    old = "2020-01-01T00:00:00+00:00"
    i = s.create_item(kind="idea", created_at=old)
    assert s.get_item(i)["created_at"] == "2020-01-01T00:00:00.000000+00:00"
    s.update_item(i, title="t")
    assert s.get_item(i)["updated_at"] > old


def test_list_items_filters_and_order(tmp_path):
    s = make(tmp_path)
    a = s.create_item(kind="job", status="new", created_at="2026-10-01T10:00:00+00:00")
    b = s.create_item(kind="event", status="new", created_at="2026-10-02T10:00:00+00:00")
    c = s.create_item(kind="job", status="filed", created_at="2026-10-02T10:00:00+00:00")
    d = s.create_item(kind="idea", status="new", created_at="2026-09-01T10:00:00+00:00")
    assert [x["id"] for x in s.list_items()] == [c, b, a, d]
    assert [x["id"] for x in s.list_items(kind=["job", "event"])] == [c, b, a]
    assert [x["id"] for x in s.list_items(kind=("job", "event"), status={"new"})] == [b, a]
    assert [x["id"] for x in s.list_items(status="filed")] == [c]
    since = datetime(2026, 10, 2, tzinfo=timezone(timedelta(hours=2)))  # 2026-10-01T22:00Z
    assert [x["id"] for x in s.list_items(since=since)] == [c, b]
    assert len(s.list_items(limit=2)) == 2
    assert s.list_items(kind=[]) == []


def test_outbox_oldest_first_task_and_event_kinds_only(tmp_path):
    s = make(tmp_path)
    late = s.create_item(kind="task", status="routing", created_at="2026-10-02T00:00:00+00:00")
    early = s.create_item(kind="event", status="routing", created_at="2026-10-01T00:00:00+00:00")
    s.create_item(kind="job", status="routing")
    s.create_item(kind="idea", status="routing")
    s.create_item(kind="event", status="new")
    assert [o["id"] for o in s.outbox()] == [early, late]


def test_events_and_meta(tmp_path):
    s = make(tmp_path)
    i = s.create_item(kind="idea")
    s.add_event(i, "created", {"a": 1})
    row = s._conn.execute("select type, payload, at from events").fetchone()
    assert row["type"] == "created" and row["payload"] == '{"a": 1}' and row["at"]
    assert s.meta_get("nope") is None
    assert s.meta_get("schema_version") == "6"
    s.meta_set("last_update_id", "42")
    s.meta_set("last_update_id", "43")
    assert s.meta_get("last_update_id") == "43"


def test_due_reminders_timezone_and_flags(tmp_path):
    s = make(tmp_path)
    i = s.create_item(kind="event")
    lisbon_plus1 = timezone(timedelta(hours=1))
    r = s.add_reminder(i, datetime(2026, 11, 7, 20, 0, tzinfo=lisbon_plus1))  # 19:00Z
    assert s.due_reminders(datetime(2026, 11, 7, 18, 59, tzinfo=timezone.utc)) == []
    # 20:30+01:00 is 19:30Z, due
    assert [x["id"] for x in s.due_reminders(datetime(2026, 11, 7, 20, 30, tzinfo=lisbon_plus1))] == [r]
    # exactly at the time is due
    assert [x["id"] for x in s.due_reminders(datetime(2026, 11, 7, 19, 0, tzinfo=timezone.utc))] == [r]
    # snooze keeps sent_at NULL and fires again later
    s.snooze_reminder(r, datetime(2026, 11, 8, 19, 0, tzinfo=timezone.utc))
    assert s.due_reminders(datetime(2026, 11, 8, 18, 0, tzinfo=timezone.utc)) == []
    assert [x["id"] for x in s.due_reminders(datetime(2026, 11, 8, 19, 0, tzinfo=timezone.utc))] == [r]
    s.mark_reminder_sent(r, datetime(2026, 11, 8, 19, 1, tzinfo=timezone.utc))
    assert s.due_reminders(datetime(2026, 11, 9, tzinfo=timezone.utc)) == []


def test_disable_reminders(tmp_path):
    s = make(tmp_path)
    i = s.create_item(kind="event")
    j = s.create_item(kind="event")
    s.add_reminder(i, datetime(2026, 11, 7, tzinfo=timezone.utc))
    k = s.add_reminder(j, datetime(2026, 11, 7, tzinfo=timezone.utc))
    s.disable_reminders(i)
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    assert [x["id"] for x in s.due_reminders(now)] == [k]


def test_naive_datetime_rejected(tmp_path):
    s = make(tmp_path)
    i = s.create_item(kind="event")
    with pytest.raises(ValueError):
        s.add_reminder(i, datetime(2026, 11, 7, 19, 0))
    with pytest.raises(ValueError):
        s.due_reminders(datetime(2026, 11, 7, 19, 0))


def test_get_reminder(tmp_path):
    s = Store(tmp_path / "c.sqlite"); s.ensure_schema()
    i = s.create_item(kind="idea", status="new")
    r = s.add_reminder(i, datetime(2026, 11, 7, 19, 0, tzinfo=timezone.utc))
    got = s.get_reminder(r)
    assert got["id"] == r and got["item_id"] == i and got["sent_at"] is None
    assert s.get_reminder(r + 1) is None
    s.close()


# -- schema v2: turns and per-chat items ------------------------------------

def _cols(s, table):
    return [r[1] for r in s._conn.execute(f"PRAGMA table_info({table})")]


def test_v1_db_migrates_to_v2_once(tmp_path):
    import shutil
    from pathlib import Path

    dst = tmp_path / "v1.sqlite"
    shutil.copy(Path(__file__).parent / "fixtures" / "v1.sqlite", dst)
    raw = sqlite3.connect(dst)
    assert "chat_id" not in [r[1] for r in raw.execute("PRAGMA table_info(items)")]
    assert raw.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
    raw.close()

    s = Store(dst)
    s.ensure_schema()
    assert "chat_id" in _cols(s, "items")
    assert "turns" in [r[0] for r in s._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert s.meta_get("schema_version") == "6"
    assert s.get_item(1)["title"] == "old"
    s.close()

    s = Store(dst)
    s.ensure_schema()
    assert s.meta_get("schema_version") == "6"
    assert _cols(s, "items").count("chat_id") == 1
    s.close()


def _names(s):
    return {r[0] for r in s._conn.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}


def test_fresh_db_is_v2_and_matches_migrated(tmp_path):
    import shutil
    from pathlib import Path

    s = make(tmp_path)
    assert s.meta_get("schema_version") == "6"
    assert "chat_id" in _cols(s, "items")
    assert "idx_items_chat_id" in _names(s)
    dst = tmp_path / "m.sqlite"
    shutil.copy(Path(__file__).parent / "fixtures" / "v1.sqlite", dst)
    m = Store(dst)
    m.ensure_schema()
    assert _names(s) == _names(m)


def test_concurrent_migration_of_v1_db(tmp_path):
    import shutil
    import threading
    from pathlib import Path

    dst = tmp_path / "v1.sqlite"
    shutil.copy(Path(__file__).parent / "fixtures" / "v1.sqlite", dst)
    stores = []
    errors = []
    barrier = threading.Barrier(2)

    def run(st):
        try:
            barrier.wait()
            st = Store(dst)
            stores.append(st)
            st.ensure_schema()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(None,)) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    assert stores[0].meta_get("schema_version") == "6"
    assert _cols(stores[0], "items").count("chat_id") == 1


def test_turns_roundtrip_and_prune_keeps_20(tmp_path):
    s = make(tmp_path)
    t0 = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
    for i in range(25):
        s.add_turn(5, "user", f"m{i}", at=t0 + timedelta(minutes=i))
    s.add_turn(6, "user", "other", at=t0)
    turns = s.recent_turns(5, limit=8)
    assert [t["text"] for t in turns] == [f"m{i}" for i in range(17, 25)]
    assert set(turns[0]) == {"id", "chat_id", "at", "role", "text", "item_id", "processed", "source"}
    assert turns[0]["processed"] == 1
    s.prune_turns(5)
    assert len(s.recent_turns(5, limit=100)) == 20
    assert s.recent_turns(5, limit=100)[0]["text"] == "m5"
    assert len(s.recent_turns(6)) == 1


def test_prune_keeps_unprocessed(tmp_path):
    s = make(tmp_path)
    t0 = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
    old = s.add_turn(5, "user", "pending", processed=False, at=t0)
    for i in range(25):
        s.add_turn(5, "bot", f"b{i}", at=t0 + timedelta(minutes=i + 1))
    s.prune_turns(5, keep=20)
    texts = [t["text"] for t in s.recent_turns(5, limit=100)]
    assert "pending" in texts and len(texts) == 21
    s.mark_turn(old, processed=True, item_id=3)
    t = s.recent_turns(5, limit=100)[0]
    assert t["processed"] == 1 and t["item_id"] == 3


def test_unprocessed_turns_since(tmp_path):
    s = make(tmp_path)
    t0 = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
    s.add_turn(1, "user", "old", processed=False, at=t0 - timedelta(hours=3))
    b = s.add_turn(2, "user", "b", processed=False, at=t0 + timedelta(minutes=2))
    a = s.add_turn(1, "user", "a", processed=False, at=t0)
    s.add_turn(1, "user", "done", at=t0)
    s.add_turn(1, "bot", "botturn", processed=False, at=t0)
    got = s.unprocessed_turns(t0 - timedelta(hours=1))
    assert [t["id"] for t in got] == [a, b]


def test_last_bot_turn_at(tmp_path):
    s = make(tmp_path)
    assert s.last_bot_turn_at(1) is None
    t0 = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
    s.add_turn(1, "bot", "x", at=t0)
    s.add_turn(1, "bot", "y", at=t0 + timedelta(minutes=3))
    s.add_turn(1, "user", "z", at=t0 + timedelta(minutes=9))
    s.add_turn(2, "bot", "w", at=t0 + timedelta(minutes=20))
    got = s.last_bot_turn_at(1)
    assert got == t0 + timedelta(minutes=3) and got.tzinfo is not None


def test_recent_items_per_chat(tmp_path):
    s = make(tmp_path)
    ids = [s.create_item(kind="task", title=f"t{i}", chat_id=7) for i in range(12)]
    s.create_item(kind="task", title="other", chat_id=8)
    s.create_item(kind="task", title="nochat")
    s.update_item(ids[11], status="dismissed")
    s.update_item(ids[10], fields={"a": 1})
    got = s.recent_items(7)
    assert [i["id"] for i in got] == ids[10::-1][:10]
    assert got[0]["fields"] == {"a": 1}
    assert len(s.recent_items(7, limit=3)) == 3
    s.update_item(ids[0], chat_id=9)
    assert [i["id"] for i in s.recent_items(9)] == [ids[0]]


def _open_and_migrate(path):
    from clawdia.storage import Store as S

    st = S(path)
    st.ensure_schema()
    st.close()


def test_two_processes_migrate_v1_db(tmp_path):
    import multiprocessing as mp
    import shutil
    from pathlib import Path

    dst = tmp_path / "v1.sqlite"
    shutil.copy(Path(__file__).parent / "fixtures" / "v1.sqlite", dst)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_open_and_migrate, args=(str(dst),)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert [p.exitcode for p in procs] == [0, 0]
    raw = sqlite3.connect(dst)
    assert raw.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "6"
    assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    raw.close()


def test_migration_2_to_3_keeps_items_and_turns(tmp_path):
    dst = tmp_path / "v2.sqlite"
    s = Store(dst)
    s.ensure_schema()
    i = s.create_item(kind="idea", title="keep me", text="keep me", chat_id=1)
    s.add_turn(1, "user", "hello", item_id=i)
    s._conn.executescript("DROP TABLE agent_runs; DROP TABLE agent_writes; UPDATE meta SET value='2' WHERE key='schema_version';")
    s.close()

    s = Store(dst)
    s.ensure_schema()
    assert s.meta_get("schema_version") == "6"
    assert {"agent_runs", "agent_writes"} <= _names(s)
    assert s.get_item(i)["title"] == "keep me" and s.recent_turns(1)[0]["text"] == "hello"
    assert s.recent_turns(1)[0]["source"] == "typed"  # created on a v3 schema in this test
    s.add_agent_write(1, i, "mark_done", {"item": {"status": "new"}}, {"status": "done"})
    assert s.agent_writes(1)[0]["before"] == {"item": {"status": "new"}}
    s.close()
    s = Store(dst)
    s.ensure_schema()  # idempotent
    assert s.meta_get("schema_version") == "6"
    s.close()


def test_turns_before_the_source_column_are_legacy(tmp_path):
    dst = tmp_path / "v2b.sqlite"
    s = Store(dst)
    s.ensure_schema()
    s.add_turn(1, "user", "forwarded text", item_id=None)
    s._conn.executescript(
        "DROP TABLE agent_runs; DROP TABLE agent_writes; ALTER TABLE turns DROP COLUMN source; "
        "UPDATE meta SET value='2' WHERE key='schema_version';")
    s.close()
    s = Store(dst)
    s.ensure_schema()
    assert [t["source"] for t in s.recent_turns(1)] == ["legacy"]
    s.add_turn(1, "user", "typed now")
    assert [t["source"] for t in s.recent_turns(1)] == ["legacy", "typed"]
    s.close()


def test_migration_3_to_4_keeps_data(tmp_path):
    dst = tmp_path / "v3.sqlite"
    s = Store(dst)
    s.ensure_schema()
    i = s.create_item(kind="idea", title="keep me", text="keep me", chat_id=1)
    s._conn.executescript("DROP TABLE memory; DROP TABLE memory_pending; UPDATE meta SET value='3' WHERE key='schema_version';")
    s.close()

    s = Store(dst)
    s.ensure_schema()
    assert s.meta_get("schema_version") == "6"
    assert {"memory", "memory_pending"} <= _names(s)
    assert s.get_item(i)["title"] == "keep me"
    now = datetime(2026, 11, 18, tzinfo=timezone.utc)
    m = s.memory_add("about", "likes tea", source="cli", turn_id=None, now=now)
    assert s.memory_get(m)["text"] == "likes tea"
    s.close()


def test_migration_3_to_4_marks_old_bot_turns_legacy(tmp_path):
    dst = tmp_path / "v3.sqlite"
    s = Store(dst)
    s.ensure_schema()
    s.add_turn(1, "user", "typed by Elena", source="typed")
    s.add_turn(1, "bot", "old bot reply, may echo captured text", source="typed")
    s.add_turn(1, "bot", "Saved #1", source="capture_reply")
    s.add_turn(1, "user", "forwarded", source="capture")
    s._conn.executescript("DROP TABLE memory; DROP TABLE memory_pending; UPDATE meta SET value='3' WHERE key='schema_version';")
    s.close()

    s = Store(dst)
    s.ensure_schema()
    assert [(t["role"], t["source"]) for t in s.recent_turns(1)] == [
        ("user", "typed"), ("bot", "legacy"), ("bot", "capture_reply"), ("user", "capture")]
    s.add_turn(1, "bot", "new typed bot reply", source="typed")  # after the migration the source is trusted again
    s.close()
    s = Store(dst)
    s.ensure_schema()  # idempotent: the second open changes nothing
    assert s.recent_turns(1)[-1]["source"] == "typed" and s.recent_turns(1)[1]["source"] == "legacy"
    s.close()


def test_migrate_v3_to_v4(tmp_path):
    import shutil
    from pathlib import Path

    dst = tmp_path / "v1.sqlite"
    shutil.copy(Path(__file__).parent / "fixtures" / "v1.sqlite", dst)
    s = Store(dst)
    s.ensure_schema()
    assert {"memory", "memory_pending"} <= _names(s)
    assert s.meta_get("schema_version") == "6"
    assert s.get_item(1)["title"] == "old"
    s.close()
    s = Store(dst)
    s.ensure_schema()  # idempotent
    assert s.meta_get("schema_version") == "6"
    s.close()


def _wait(s, now, chat_id=1, ttl=timedelta(hours=6)):
    return s.wait_open(chat_id, 7, 42, "standup", 100, now, ttl, "q")


def test_migrate_v4_to_v5_adds_call_waits(tmp_path):
    dst = tmp_path / "v4.sqlite"
    s = Store(dst)
    s.ensure_schema()
    i = s.create_item(kind="idea", title="keep me", text="keep me", chat_id=1)
    s._conn.executescript("DROP TABLE call_waits; UPDATE meta SET value='4' WHERE key='schema_version';")
    s.close()

    s = Store(dst)
    s.ensure_schema()
    assert s.meta_get("schema_version") == "6"
    assert "call_waits" in _names(s)
    assert s.get_item(i)["title"] == "keep me"
    now = datetime(2026, 11, 20, 10, tzinfo=timezone.utc)
    w = _wait(s, now)
    got = s.wait_get(w)
    assert got["status"] == "open" and got["title_hint"] == "standup" and got["baseline_id"] == 100
    assert got["expires_at"] == (now + timedelta(hours=6)).isoformat(timespec="microseconds")
    assert [r["id"] for r in s.waits_open()] == [w]
    s.close()
    s = Store(dst)
    s.ensure_schema()  # idempotent
    assert s.meta_get("schema_version") == "6"
    s.close()


def test_wait_claim_is_once(tmp_path):
    s = make(tmp_path)
    now = datetime(2026, 11, 20, 10, tzinfo=timezone.utc)
    w = _wait(s, now)
    assert s.wait_claim(w, 555, now) is True
    assert s.wait_claim(w, 556, now) is False
    got = s.wait_get(w)
    assert got["status"] == "answering" and got["transcript_id"] == 555
    assert got["last_check_at"] == now.isoformat(timespec="microseconds")
    assert s.waits_open() == []
    s.close()


def test_wait_cancel_chat_only_open(tmp_path):
    s = make(tmp_path)
    now = datetime(2026, 11, 20, 10, tzinfo=timezone.utc)
    a, b, c, other = _wait(s, now), _wait(s, now), _wait(s, now), _wait(s, now, chat_id=2)
    s.wait_claim(b, 1, now)
    s.wait_set_status(c, "answered")
    assert s.wait_cancel_chat(1) == 1
    assert s.wait_get(a)["status"] == "cancelled"
    assert s.wait_get(b)["status"] == "answering"
    assert s.wait_get(c)["status"] == "answered"
    assert s.wait_get(other)["status"] == "open"
    s.close()


def test_wait_cancel_single_and_waits_open_filters(tmp_path):
    s = make(tmp_path)
    now = datetime(2026, 11, 20, 10, tzinfo=timezone.utc)
    a, b, other = _wait(s, now), _wait(s, now), _wait(s, now, chat_id=2)
    assert [r["id"] for r in s.waits_open()] == [a, b, other]
    assert [r["id"] for r in s.waits_open(1)] == [a, b]
    assert s.wait_cancel(a) is True
    assert s.wait_cancel(a) is False
    s.wait_claim(b, 1, now)
    assert s.wait_cancel(b) is False  # only open -> cancelled
    assert s.wait_get(999) is None
    s.close()


def test_waits_reset_answering_after_15_min(tmp_path):
    s = make(tmp_path)
    t0 = datetime(2026, 11, 20, 10, tzinfo=timezone.utc)
    old, fresh, still_open = _wait(s, t0), _wait(s, t0), _wait(s, t0)
    s.wait_claim(old, 1, t0)
    s.wait_claim(fresh, 2, t0 + timedelta(minutes=10))
    assert s.waits_reset_answering(t0 + timedelta(minutes=14)) == 0
    assert s.waits_reset_answering(t0 + timedelta(minutes=15)) == 1
    assert s.wait_get(old)["status"] == "open"
    assert s.wait_get(fresh)["status"] == "answering"
    assert s.wait_get(still_open)["status"] == "open"
    assert s.wait_claim(old, 3, t0 + timedelta(minutes=16)) is True  # retry after reset
    s.close()


# -- laptop focus request -------------------------------------------------


def test_focus_request_open_take_clears(tmp_path):
    from datetime import datetime, timezone
    from clawdia.storage import Store
    s = Store(tmp_path / "c.sqlite"); s.ensure_schema()
    now = datetime(2026, 11, 21, 9, 0, tzinfo=timezone.utc)
    assert s.focus_request_take() is None
    s.focus_request_open(now)
    s.focus_request_open(now)  # a second press keeps one request
    assert s.focus_request_take() == {"requested_at": "2026-11-21T09:00:00.000000+00:00"}
    assert s.focus_request_take() is None


def test_focus_report_roundtrip(tmp_path):
    from datetime import datetime, timezone
    from clawdia.storage import Store
    s = Store(tmp_path / "c.sqlite"); s.ensure_schema()
    now = datetime(2026, 11, 21, 9, 0, tzinfo=timezone.utc)
    assert s.focus_report_get() is None
    s.focus_report("failed", "boom", now)
    assert s.focus_report_get() == {"status": "failed", "reason": "boom", "at": "2026-11-21T09:00:00.000000+00:00"}


def _v5_db(path):
    """A database as v5 left it: old memory CHECK, no misses table."""
    s = Store(path)
    s.ensure_schema()
    s._conn.executescript("""
        DROP TABLE misses;
        DROP TABLE memory;
        CREATE TABLE memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section TEXT NOT NULL CHECK (section IN ('about','context')),
            text TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            deleted_at TEXT, source TEXT NOT NULL CHECK (source IN ('chat','button','seed','cli')),
            turn_id INTEGER);
        UPDATE meta SET value='5' WHERE key='schema_version';""")
    return s


def test_migrate_v5_to_v6_keeps_memory_ids(tmp_path):
    dst = tmp_path / "v5.sqlite"
    s = _v5_db(dst)
    now = datetime(2026, 11, 20, 10, tzinfo=timezone.utc)
    a = s.memory_add("about", "first", source="chat", turn_id=None, now=now)
    b = s.memory_add("context", "second", source="chat", turn_id=None, now=now)
    s.memory_delete(b, now)
    c = s.memory_add("about", "third", source="chat", turn_id=None, now=now)
    s.memory_delete(c, now)  # highest id deleted: its number must not be reused
    pid = s.pending_add("about", "pending text", turn_id=5, now=now)
    s.close()

    s = Store(dst)
    s.ensure_schema()
    assert s.meta_get("schema_version") == "6"
    facts = {f["id"]: f for f in s.memory_list(include_deleted=True)}
    assert facts[a]["text"] == "first" and facts[b]["text"] == "second" and facts[b]["deleted_at"]
    assert s.pending_get(pid)["status"] == "open"
    r = s.memory_add("rules", "always answer in Russian", source="button", turn_id=None, now=now)
    assert r > c
    assert "misses" in _names(s)
    s.close()
    s = Store(dst)
    s.ensure_schema()  # idempotent
    assert s.meta_get("schema_version") == "6"
    s.close()


def test_fresh_db_accepts_rules_and_misses(tmp_path):
    s = make(tmp_path)
    now = datetime(2026, 11, 23, 10, tzinfo=timezone.utc)
    assert s.meta_get("schema_version") == "6"
    s.memory_add("rules", "a rule", source="button", turn_id=None, now=now)
    assert s.miss_add(1, None, "calls", "what was said", "no_data", now) > 0
    with pytest.raises(sqlite3.IntegrityError):
        s.memory_add("nonsense", "x", source="chat", turn_id=None, now=now)


def test_miss_add_validates_and_cuts(tmp_path):
    s = make(tmp_path)
    now = datetime(2026, 11, 23, 10, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        s.miss_add(1, None, "t", "r", "bogus", now)
    i = s.miss_add(1, 7, "t" * 100, "r" * 500, "no_tool", now)
    m = s.miss_get(i)
    assert len(m["topic"]) == 60 and len(m["request"]) == 300
    assert m["status"] == "open" and m["turn_id"] == 7 and m["closed_at"] is None
    assert s.miss_get(999) is None


def test_misses_close_topic_case_insensitive(tmp_path):
    s = make(tmp_path)
    now = datetime(2026, 11, 23, 10, tzinfo=timezone.utc)
    a = s.miss_add(1, None, "Итоги звонка", "q1", "no_tool", now)
    b = s.miss_add(1, None, "итоги звонка", "q2", "no_tool", now + timedelta(minutes=1))
    c = s.miss_add(1, None, "other", "q3", "no_data", now + timedelta(minutes=2))
    d = s.miss_add(1, None, "Итоги звонка", "q4", "corrected", now + timedelta(minutes=3))
    assert s.misses_close_topic(d, "backlog", now) == 3
    assert s.misses_close_topic(a, "dismissed", now) == 0  # nothing open is left on that topic
    assert s.miss_get(a)["status"] == "backlog" and s.miss_get(c)["status"] == "open"
    e = s.miss_add(1, None, "итоги звонка", "q5", "no_tool", now + timedelta(minutes=4))
    assert s.misses_close_topic(e, "dismissed", now) == 1
    assert s.miss_get(b)["status"] == "backlog"
    assert [m["id"] for m in s.misses_since(now)] == [e, d, c, b, a]
    assert [m["id"] for m in s.misses_since(now, "open")] == [c]


def test_misses_close_topic_until_leaves_later_misses_open(tmp_path):
    s = make(tmp_path)
    now = datetime(2026, 11, 23, 10, tzinfo=timezone.utc)
    a = s.miss_add(1, None, "Topic", "q1", "no_tool", now)
    b = s.miss_add(1, None, "topic", "q2", "no_tool", now + timedelta(hours=2))
    assert s.misses_close_topic(a, "backlog", now, until=now + timedelta(hours=1)) == 1
    assert s.miss_get(a)["status"] == "backlog" and s.miss_get(b)["status"] == "open"


def test_memory_list_orders_about_context_rules(tmp_path):
    s = make(tmp_path)
    now = datetime(2026, 11, 23, 10, tzinfo=timezone.utc)
    r = s.memory_add("rules", "r", source="cli", turn_id=None, now=now)
    c = s.memory_add("context", "c", source="cli", turn_id=None, now=now)
    a = s.memory_add("about", "a", source="cli", turn_id=None, now=now)
    assert [f["id"] for f in s.memory_list()] == [a, c, r]
