"""The daily focus map for the agent: newest map, sections, cleaning, and task ranking."""

import json
from datetime import date

from clawdia.agent_tools import SECOND_READ_NO_WRITES, call_tool, rank_tasks
from clawdia.focus import is_auto, load_plan, newest_map, today_map, top_items
from tests.test_agent import make, seed, _turn

MAP = """# Daily Focus - 2026-11-18 (Wednesday)

**Sources (Lisbon = UTC):** ingest ok.

## 📋 Brief
The Northwind kickoff is done.

## 🎯 Top-3 (3 open)

1. **Northwind checklist, promised to Hugo by Fri 11-20** | Owner: Elena. [source](<https://docs.google.com/document/d/abc/edit>) [source](</home/user/notes/daily-focus/evidence/x.md>)
2. **Launch week follow-ups** | Owner: Elena. [Hugo](<https://web.telegram.org/k/?account=2#66>)

## 🟡 Queue - open Elena actions, not Top-N
- Coworking check by Fri

## 👀 Their move - watching
- Greta: answer on Silver Heron

## 📅 Calendar
- Today: 18:00-18:30 Mara Quist

## 🕰 Stale - close or revive
- Orchid: paused
"""


def write(folder, day, text=MAP):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{day}.md").write_text(text, encoding="utf-8")


def test_newest_map_ignores_future_and_other_files(tmp_path):
    write(tmp_path, "2026-11-17"); write(tmp_path, "2026-11-18"); write(tmp_path, "2026-11-19")
    (tmp_path / "week-2026-11-16.md").write_text("x")
    day, path = newest_map(tmp_path, date(2026, 11, 18))
    assert day == date(2026, 11, 18) and path.name == "2026-11-18.md"
    assert newest_map(tmp_path / "missing", date(2026, 11, 18)) is None


def test_plan_has_top_in_order_clean_links_no_bold(tmp_path):
    write(tmp_path, "2026-11-18")
    plan = load_plan(tmp_path, date(2026, 11, 18))
    assert plan["map_date"] == "2026-11-18" and plan["is_today"] is True
    top = plan["top"]
    assert top.index("Northwind") < top.index("Launch week")
    assert "**" not in top and "/home/user" not in top
    assert "https://docs.google.com/document/d/abc/edit" in top and "Hugo: https://web.telegram.org" in top
    assert "Coworking" in plan["queue"] and "Mara Quist" in plan["calendar"] and "Greta" in plan["their_move"]
    assert "Orchid" not in json.dumps(plan)  # the stale section is not part of the plan


def test_old_map_is_marked_and_long_top_is_cut(tmp_path):
    write(tmp_path, "2026-11-16", MAP.replace("Launch week follow-ups", "x" * 9000))
    plan = load_plan(tmp_path, date(2026, 11, 18))
    assert plan["is_today"] is False and plan["map_date"] == "2026-11-16"
    assert plan["top_cut"] is True and len(plan["top"]) < 6100
    assert load_plan(tmp_path / "none", date(2026, 11, 18)) == {"error": "no daily focus map on the server yet"}


async def test_today_plan_tool_reads_the_map_and_stops_writes(tmp_path):
    d = make(tmp_path)
    d.settings.focus_dir = tmp_path / "focus"
    write(d.settings.focus_dir, "2026-10-29")  # the test clock is 2026-10-29
    i = seed(d, "Pay invoice")
    t = _turn(d, in_view=[i])
    out = json.loads(await call_tool(t, "today_plan", {}))
    assert out["is_today"] is True and "Northwind" in out["top"]
    assert json.loads(await call_tool(t, "mark_done", {"id": i}))["error"] == SECOND_READ_NO_WRITES


def test_rank_tasks_by_importance_with_old_and_parked_counted():
    today = date(2026, 11, 18)
    tasks = [
        {"id": 2735, "level": "medium", "deadline": "2027-01-01"},           # parked
        {"id": 2678, "level": "high", "deadline": "2026-10-02"},             # overdue 47 days: old
        {"id": 2624, "level": "low", "deadline": None, "is_stale": True},    # stale
        {"id": 2752, "level": "medium", "deadline": "2026-11-23"},
        {"id": 2800, "level": "critical", "deadline": "2026-11-20"},
        {"id": 2801, "level": "high", "deadline": None},
        {"id": 2802, "level": "high", "deadline": "2026-11-19"},
        {"id": 2803, "level": "low", "deadline": "2026-11-10"},               # overdue 8 days: still live
    ]
    out = rank_tasks(tasks, today, shown=3)
    assert [t["id"] for t in out["items"]] == [2800, 2802, 2801]
    assert out["not_shown"] == 2 and out["count"] == 3
    assert (out["old_count"], out["parked_count"]) == (2, 1)
    assert out["old_ids"] == [2678, 2624] and out["parked_ids"] == [2735]


async def test_map_links_are_allowed_in_the_reply(tmp_path):
    from clawdia.agent import check_reply

    d = make(tmp_path)
    d.settings.focus_dir = tmp_path / "focus"
    write(d.settings.focus_dir, "2026-10-29")
    t = _turn(d)
    await call_tool(t, "today_plan", {})
    reply = ("1. Northwind one-pager: https://docs.google.com/document/d/abc/edit\n"
             "2. Launch week: https://web.telegram.org/k/?account=2#66")
    assert check_reply(reply, d.store, t.urls)[1] == []


AUTO_MAP = """Built automatically on the laptop at 08:12.

# Daily Focus - 2026-11-21

## 🔴 Decisions for you
- Energy set to normal (Top-4); say if you want a lighter day.
- Calendar could not be read.

## 🎯 Top-4 (4 open)

1. **First item** | Owner: Elena. [source](</home/user/x.md>)
2. **Second item** [Hugo](<https://web.telegram.org/k/?account=2#66>)
3. **Third item**
4. **Fourth item**
5. **Fifth item**

## 🟡 Queue
- later
"""


def test_decisions_section_is_parsed(tmp_path):
    write(tmp_path, "2026-11-21", AUTO_MAP)
    plan = load_plan(tmp_path, date(2026, 11, 21))
    assert "Energy set to normal" in plan["decisions"] and "Fifth" in plan["top"]


def test_today_map_only_for_today(tmp_path):
    write(tmp_path, "2026-11-20")
    assert today_map(tmp_path, date(2026, 11, 21)) is None
    write(tmp_path, "2026-11-21")
    assert today_map(tmp_path, date(2026, 11, 21)).name == "2026-11-21.md"
    assert today_map(tmp_path / "missing", date(2026, 11, 21)) is None


def test_top_items_are_cleaned_and_limited():
    assert top_items(AUTO_MAP, 4) == ["First item | Owner: Elena.", "Second item Hugo: https://web.telegram.org/k/?account=2#66",
                                      "Third item", "Fourth item"]
    assert top_items(AUTO_MAP, 2) == top_items(AUTO_MAP, 4)[:2]
    assert top_items("# no top here", 4) == []


def test_is_auto():
    assert is_auto(AUTO_MAP) and not is_auto(MAP) and not is_auto("")
