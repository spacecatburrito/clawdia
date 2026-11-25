"""Agent eval (spec 6.14, Task 12b): realistic conversations run through `run_agent` with the real model.

    python -m clawdia.agent_eval --cases evals/agent_cases.json --out evals/results/agent-eval-DATE.md
        [--estimate] [--only id,id] [--yes-paid]

Each case builds an in-memory Store, a fake SecondClient and a fake tagger (no model call), runs the
agent loop and checks which tools the model called, the number of writes and the reply text. The
real run costs money, so it refuses to start without `--yes-paid`; `--estimate` is computed locally
from case sizes and never calls the API. The key comes from CLAWDIA_ANTHROPIC_API_KEY and is never
printed.

Case shape (extra keys are optional):
  id, state {items, turns, jobs, memory}, message, reply_to {message_id, text} | null,
  must_call [{tool, args_subset}], must_not_call [tool], reply_matches [regex], reply_must_not_match [regex],
  max_writes / min_writes (int, optional: writes recorded for the turn), expected_calls (int, optional: model calls, for --estimate).
  state.second (optional): what the agent's Second reads see, {tasks, chats, messages, transcripts} (FakeSecondReads).
  state.focus (optional): the daily focus map on the server, {date: YYYY-MM-DD, text: markdown}.
Owner memory (all optional): state.memory is a list of {section: about|context, text}, seeded before the run as live
facts with source "seed" (M1, M2, ... in order). After the run, memory_count is the exact number of live memory rows,
pending_count the exact number of open memory proposals, pending_max an upper bound on the open proposals (use it
when the model may or may not propose; do not combine it with a larger pending_count). A requested count that the
runner did not measure is a failure.
mem_forward_injection and mem_triage_injection test the code guard (a dirty turn only raises a proposal, never a
stored row), not the model's refusal: the model may well call remember, and the case passes while memory_count stays 0.
Strings in a case may hold the placeholder {EM}, replaced by the em dash character on load, so the
file itself never contains it. must_call counts only calls that ran without an error result
(a refused call does not count). In `args_subset` a string value matches by case-insensitive substring,
any other value by equality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from clawdia.agent import AGENT_RULES, cost_usd, run_agent
from clawdia.agent_tools import TOOLS
from clawdia.capture import Deps
from clawdia.config import Settings
from clawdia.context import ContextCache
from clawdia.schedule import WorkCalendar
from clawdia.second_client import SecondRejected
from clawdia.storage import Store
from clawdia.tagger import TagResult

EM = chr(0x2014)
CHAT = 1
NOW = datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc)  # a Monday, 10:00 in Lisbon
MESSAGE_ID = 9000
KEY_ENV = "CLAWDIA_ANTHROPIC_API_KEY"
REQUIRED_KEYS = ("id", "state", "message", "reply_to", "must_call", "must_not_call", "reply_matches",
                 "reply_must_not_match")

# Local estimate assumptions (rough, stated in the output): ~4 characters per token, each tool
# result adds RESULT_TOKENS to the next call, each call writes ANSWER_TOKENS of output including
# a medium-effort thinking allowance.
CHARS_PER_TOKEN = 4
RESULT_TOKENS = 400
ANSWER_TOKENS = 450


# -- case file ------------------------------------------------------------------------------------


def _subst(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("{EM}", EM)
    if isinstance(value, list):
        return [_subst(v) for v in value]
    if isinstance(value, dict):
        return {k: _subst(v) for k, v in value.items()}
    return value


def _count(case: dict, key: str) -> None:
    value = case.get(key)
    if key in case and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"case {case['id']}: {key} must be a non-negative integer")


def _validate_memory(case: dict) -> None:
    facts = case["state"].get("memory", [])
    if not isinstance(facts, list):
        raise ValueError(f"case {case['id']}: state.memory must be a list")
    for fact in facts:
        if (not isinstance(fact, dict) or fact.get("section") not in ("about", "context")
                or not isinstance(fact.get("text"), str) or not fact["text"].strip()):
            raise ValueError(f"case {case['id']}: bad state.memory entry {fact!r}")
    for key in ("memory_count", "pending_count", "pending_max"):
        _count(case, key)
    if "pending_count" in case and "pending_max" in case and case["pending_count"] > case["pending_max"]:
        raise ValueError(f"case {case['id']}: pending_count exceeds pending_max")


def validate_case(case: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in case]
    if missing:
        raise ValueError(f"case {case.get('id')!r}: missing {', '.join(missing)}")
    if not isinstance(case["id"], str) or not case["id"]:
        raise ValueError("case id must be a non-empty string")
    state = case["state"]
    for key in ("items", "turns", "jobs"):
        if not isinstance(state.get(key), list):
            raise ValueError(f"case {case['id']}: state.{key} must be a list")
    _validate_memory(case)
    ids = [item.get("id") for item in state["items"]]
    if any(not isinstance(i, int) or i < 1 for i in ids) or ids != sorted(set(ids)):
        raise ValueError(f"case {case['id']}: item ids must be positive, unique and ascending")
    if not case["message"].strip():
        raise ValueError(f"case {case['id']}: empty message")
    for m in case["must_call"]:
        if not isinstance(m.get("tool"), str) or not isinstance(m.get("args_subset", {}), dict):
            raise ValueError(f"case {case['id']}: bad must_call entry {m!r}")
    for pattern in case["reply_matches"] + case["reply_must_not_match"]:
        re.compile(pattern)

    second = state.get("second")
    if second is not None and (not isinstance(second, dict)
                               or set(second) - {"tasks", "chats", "messages", "transcripts"}):
        raise ValueError(f"case {case['id']}: state.second must hold only tasks, chats, messages, transcripts")


def load_cases(path: str | Path) -> list[dict]:
    cases = [_subst(c) for c in json.loads(Path(path).read_text(encoding="utf-8"))]
    for c in cases:
        validate_case(c)
    return cases


# -- fakes ----------------------------------------------------------------------------------------


class FakeSecond:
    """Stands in for SecondClient: serves the case's triage jobs and records the decisions."""

    def __init__(self, jobs: list[dict]) -> None:
        self.jobs = jobs
        self.decisions: list[tuple] = []
        self.pulls = 0
        self._next = 9000

    async def list_jobs(self, since, limit=50):
        return list(self.jobs)[:limit]

    async def get_job(self, intake_id):
        for job in self.jobs:
            if job.get("id") == intake_id:
                return job
        raise SecondRejected("not_found", "no such job", 404)

    async def patch_job(self, intake_id, decision, note):
        self.decisions.append((intake_id, decision, note))
        return {"id": intake_id, "decision": decision}

    async def post_task(self, item):
        self._next += 1
        return {"id": self._next, "project": item.get("project") or "Acme"}

    async def post_job(self, item, images, lang):
        self._next += 1
        return {"id": self._next, "status": "queued"}

    async def get_context(self, etag):
        return None  # like a 304: the cached snapshot stays

    async def pull_transcripts(self):
        self.pulls += 1
        return {"status": "ok"}  # nothing new arrives: the case's transcripts are what Second already holds


class FakeSecondReads:
    """Stands in for SecondReader (second-api /v1/read): answers the six read tools from the case's
    `state.second`, shaped like the real server (third-party text under `untrusted`)."""

    def __init__(self, data: dict) -> None:
        self.data = data
        self.calls: list[tuple[str, dict]] = []
        self.freshness = {"latest_message_at": (NOW - timedelta(minutes=12)).isoformat(),
                          "latest_message_age_minutes": 12.0}

    @staticmethod
    def _has(text: Any, needle: str) -> bool:
        return not needle or needle.casefold() in str(text or "").casefold()

    async def call(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        d = self.data
        if name == "list_tasks":
            rows = [t for t in d.get("tasks", []) if self._has(t.get("project"), args.get("project", ""))
                    and self._has(f"{t.get('title')} {t.get('body')}", args.get("search", ""))
                    and (args.get("status", "open") == "all" or t.get("status", "open") == args.get("status", "open"))]
            out = {"items": [{"id": t["id"], "status": t.get("status", "open"), "projects": [t.get("project")],
                              "deadline": t.get("deadline"), "level": t.get("level", "medium"),
                              "is_stale": t.get("is_stale", False), "source_chat_id": t.get("chat_id"),
                              "source_message_tg_id": t.get("tg_message_id"),
                              "untrusted": {"title": t.get("title"), "body": t.get("body")}} for t in rows]}
        elif name == "get_task":
            task = next((t for t in d.get("tasks", []) if t["id"] == args.get("task_id")), None)
            if task is None:
                raise SecondRejected("not_found", f"Task {args.get('task_id')} not found.", 404)
            out = {"item": {"id": task["id"], "status": task.get("status", "open"), "projects": [task.get("project")],
                            "untrusted": {"title": task.get("title"), "body": task.get("body")}}}
        elif name == "list_chats":
            out = {"items": [{"id": c["id"], "type": c.get("type", "group"), "tg_chat_id": c.get("tg_chat_id"),
                              "untrusted": {"title": c["title"]}}
                             for c in d.get("chats", []) if self._has(c["title"], args.get("search", ""))]}
        elif name == "search_messages":
            titles = {c["id"]: c["title"] for c in d.get("chats", [])}
            chats = {c["id"]: c for c in d.get("chats", [])}
            rows = [m for m in d.get("messages", []) if args.get("chat_id") in (None, m.get("chat_id"))
                    and self._has(m.get("text"), args.get("search", ""))]
            out = {"truncated": False, "next_before_id": None, "items": [
                {"id": m["id"], "chat_id": m.get("chat_id"), "sent_at": m.get("sent_at"),
                 "tg_chat_id": chats.get(m.get("chat_id"), {}).get("tg_chat_id"),
                 "chat_type": chats.get(m.get("chat_id"), {}).get("type", "group"),
                 "tg_message_id": m.get("tg_message_id"),
                 "is_from_me": m.get("is_from_me", False),
                 "untrusted": {"chat_title": titles.get(m.get("chat_id")), "sender_name": m.get("sender"),
                               "text": m.get("text")}} for m in rows]}
        elif name == "list_transcripts":
            out = {"items": [{"id": t["id"], "uploaded_at": t.get("uploaded_at"), "length_chars": len(t.get("text", "")),
                              "untrusted": {"filename": t.get("filename")}} for t in d.get("transcripts", [])]}
        elif name == "read_transcript":
            tr = next((t for t in d.get("transcripts", []) if t["id"] == args.get("transcript_id")), None)
            if tr is None:
                raise SecondRejected("not_found", f"Transcript {args.get('transcript_id')} not found.", 404)
            out = {"id": tr["id"], "next_offset": None, "untrusted": {"filename": tr.get("filename"), "text": tr.get("text")}}
        else:
            raise SecondRejected("unknown_tool", f"no read tool {name}", 404)
        out["count"] = len(out.get("items", [])) if "items" in out else None
        return {**out, "freshness": self.freshness}

    async def chats(self) -> dict:
        return {c["id"]: (c.get("tg_chat_id"), c.get("type", "group")) for c in self.data.get("chats", [])}

    async def close(self) -> None:
        return None


async def fake_tag(item, snapshot, settings, **_kw):
    """Deterministic stand-in for the tagger, so the eval pays for the agent model only."""
    text = (item.get("text") or "").strip()
    low = text.casefold()
    if "http" in low and re.search(r"product (lead|manager)|head of product|\bcpo\b|careers|vacanc|вакан", low):
        return TagResult("job", " ".join(text.split())[:60], None, {}, 0.9, None, "fake-tagger")
    kind = "buy" if ("купить" in low or low.startswith("buy ")) else ("idea" if ("идея" in low or "idea" in low) else "unsorted")
    return TagResult(kind, " ".join(text.split())[:60] or "Note", None, {}, 0.9, None, "fake-tagger")


def build_deps(case: dict, state_dir: Path, api_key: str = "unused") -> tuple[Deps, FakeSecond]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    store = Store(state_dir / "c.sqlite")
    store.ensure_schema()
    settings = Settings(bot_token="t", owner_id=CHAT, second_api_token="t", anthropic_api_key=api_key,
                        openai_api_key="k", state_dir=state_dir, brain="agent", agent_daily_usd=1000.0)
    cache = ContextCache(state_dir / "context.json")
    context = case["state"].get("context")
    if context:
        cache.save(context, "eval")
    jobs = [{"captured_at": NOW.isoformat(), "error": None, "decision_synced": False, "notion_url": None,
             "decision": None, **j} for j in case["state"]["jobs"]]
    second = FakeSecond(jobs)
    deps = Deps(store=store, settings=settings, context_cache=cache, second=second,
                calendar=WorkCalendar("mon-fri", "Europe/Lisbon"), bot=SimpleNamespace(),
                tag_fn=fake_tag, transcribe_fn=_none, save_media_fn=_none, now_fn=lambda: NOW, group_wait_s=0)
    deps.bot.send_message = _none
    focus = case["state"].get("focus")
    settings.focus_dir = state_dir / "focus"
    settings.tg_web_account = "2"  # the cases expect a Telegram Web link with an account slot
    if focus:
        settings.focus_dir.mkdir(parents=True, exist_ok=True)
        (settings.focus_dir / f"{focus['date']}.md").write_text(focus["text"], encoding="utf-8")
    second_data = case["state"].get("second")
    if second_data is not None:
        reads = FakeSecondReads(second_data)
        deps.second_reader_fn = lambda: reads
    for item in case["state"]["items"]:
        fields = {k: v for k, v in item.items() if k not in ("id", "reminders")}
        reminders = [datetime.fromisoformat(r) for r in item.get("reminders", [])]
        if reminders:  # a real reminder: rows plus the item's `reminders` field, like add_reminders
            fields["fields"] = {**(fields.get("fields") or {}), "reminders": [r.isoformat() for r in reminders]}
        fields.setdefault("chat_id", CHAT)
        fields.setdefault("text", fields.get("title"))
        fields.setdefault("created_at", NOW - timedelta(hours=1))
        new_id = store.create_item(**fields)
        if new_id != item["id"]:  # the case names its item numbers; ascending order keeps them free
            with store._lock, store._conn:
                store._conn.execute("UPDATE items SET id = ? WHERE id = ?", (item["id"], new_id))
        for at in reminders:
            store.add_reminder(item["id"], at)
    if case["state"]["items"]:
        with store._lock, store._conn:  # the next saved item gets a number above every seeded one
            store._conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'items'",
                                (case["state"]["items"][-1]["id"],))
    for n, turn in enumerate(case["state"]["turns"]):
        role = turn.get("role", "user")
        store.add_turn(CHAT, role, turn["text"], item_id=turn.get("item_id"), processed=True,
                       at=NOW - timedelta(minutes=30 - n), source=turn.get("source", "typed"))
    for fact in case["state"].get("memory", []):
        store.memory_add(fact["section"], fact["text"], source="seed", turn_id=None, now=NOW - timedelta(days=1))
    return deps, second


def open_pending_count(store: Store) -> int:
    with store._lock:
        return int(store._conn.execute("SELECT COUNT(*) FROM memory_pending WHERE status = 'open'").fetchone()[0])


async def _none(*a, **kw):
    return None


def build_message(case: dict) -> Any:
    replied = None
    if case["reply_to"]:
        replied = SimpleNamespace(message_id=case["reply_to"]["message_id"], text=case["reply_to"].get("text"),
                                  caption=None)
    return SimpleNamespace(text=case["message"], caption=None, entities=[], caption_entities=[], forward_origin=None,
                           media_group_id=None, message_id=MESSAGE_ID, date=NOW, chat=SimpleNamespace(id=CHAT),
                           from_user=SimpleNamespace(id=CHAT), reply_to_message=replied)


# -- recording client and checker -----------------------------------------------------------------


class Recorder:
    """Wraps the client: keeps every tool call the model made and the cost of every response."""

    def __init__(self, client: Any, model: str) -> None:
        self._client, self._model = client, model
        self.messages = self
        self.tool_calls: list[dict] = []  # each: name, input, id, ok (None until its result is seen)
        self.cost = 0.0
        self.calls = 0

    def _settle(self, messages: list) -> None:
        """Mark the earlier tool calls with the outcome the model was given: ok unless is_error."""
        last = messages[-1] if messages else None
        if not last or last.get("role") != "user" or not isinstance(last.get("content"), list):
            return
        for block in last["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                for call in self.tool_calls:
                    if call["id"] == block.get("tool_use_id"):
                        call["ok"] = not block.get("is_error", False)

    async def create(self, **kw):
        self._settle(kw.get("messages") or [])
        response = await self._client.messages.create(**kw)
        self.calls += 1
        self.cost += cost_usd(self._model, response.usage)
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                self.tool_calls.append({"name": block.name, "input": dict(block.input or {}),
                                        "id": getattr(block, "id", None), "ok": None})
        return response


def _arg_matches(actual: Any, wanted: Any) -> bool:
    if isinstance(wanted, str):
        return isinstance(actual, str) and wanted.casefold() in actual.casefold()
    return actual == wanted


def check_case(case: dict, tool_calls: list[dict], reply: str | None, writes: int, *,
               memory_rows: int | None = None, pending_open: int | None = None) -> list[str]:
    """Failure messages for one run; an empty list is a pass. memory_rows / pending_open are the live memory rows
    and open proposals after the run; a case that asks for a count needs the measurement."""
    fails: list[str] = []
    if reply is None:
        fails.append("no agent reply")
    called = [c["name"] for c in tool_calls]  # attempts, including refused ones
    ran = [c for c in tool_calls if c.get("ok", True) is True]  # calls that ran without an error result
    for need in case["must_call"]:
        subset = need.get("args_subset") or {}
        if not any(c["name"] == need["tool"] and all(_arg_matches(c["input"].get(k), v) for k, v in subset.items())
                   for c in ran):
            fails.append(f"missing call {need['tool']} {json.dumps(subset, ensure_ascii=False)}")
    for tool in case["must_not_call"]:
        if tool in called:
            fails.append(f"forbidden call {tool}")
    if "max_writes" in case and writes > case["max_writes"]:
        fails.append(f"{writes} writes, at most {case['max_writes']}")
    if "min_writes" in case and writes < case["min_writes"]:
        fails.append(f"{writes} writes, at least {case['min_writes']}")
    for key, actual, label in (("memory_count", memory_rows, "memory rows"),
                               ("pending_count", pending_open, "open pending proposals")):
        if key in case:
            if actual is None:
                fails.append(f"{key} asked but {label} not measured")
            elif actual != case[key]:
                fails.append(f"{actual} {label}, expected {case[key]}")
    if "pending_max" in case:
        if pending_open is None:
            fails.append("pending_max asked but open pending proposals not measured")
        elif pending_open > case["pending_max"]:
            fails.append(f"{pending_open} open pending proposals, at most {case['pending_max']}")
    if reply is not None:
        for pattern in case["reply_matches"]:
            if not re.search(pattern, reply, re.IGNORECASE | re.MULTILINE):
                fails.append(f"reply does not match {pattern!r}")
        for pattern in case["reply_must_not_match"]:
            if re.search(pattern, reply, re.IGNORECASE | re.MULTILINE):
                fails.append(f"reply matches forbidden {pattern!r}")
        if EM in reply:
            fails.append("reply has a long dash")
    return fails


@dataclass
class Result:
    id: str
    passed: bool
    failures: list[str]
    tools: list[str]
    cost: float
    seconds: float
    calls: int
    reply: str | None = None
    writes: int = 0
    memory_rows: int | None = None
    pending_open: int | None = None


async def run_case(case: dict, client: Any, work_dir: Path) -> Result:
    deps, _second = build_deps(case, Path(work_dir) / case["id"])
    try:
        recorder = Recorder(client, deps.settings.agent_model)
        deps.ai_client = recorder
        turn_id = deps.store.add_turn(CHAT, "user", case["message"], processed=False, at=NOW)
        started = time.monotonic()
        outcome = await run_agent(build_message(case), deps, turn_id=turn_id, now=NOW)
        seconds = time.monotonic() - started
        writes = len(deps.store.agent_writes(turn_id))
        reply = outcome.text if outcome is not None else None
        recorder._settle([])  # nothing: calls whose result the model never saw stay unsettled
        calls = [c if c["ok"] is not None else {**c, "ok": False} for c in recorder.tool_calls]
        memory_rows, pending_open = len(deps.store.memory_list()), open_pending_count(deps.store)
        fails = check_case(case, calls, reply, writes, memory_rows=memory_rows, pending_open=pending_open)
        shown = [c["name"] + ("" if c["ok"] else "(refused)") for c in calls]
        return Result(case["id"], not fails, fails, shown, recorder.cost, seconds,
                      recorder.calls, reply, writes, memory_rows, pending_open)
    finally:
        deps.store.close()


# -- estimate -------------------------------------------------------------------------------------


def _tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def estimate(cases: list[dict], model: str | None = None) -> dict:
    """Local cost estimate: no network. Per case, calls = the case's expected_calls, else 1 + required tool calls (at least 2,
    at most the agent's 6); every call re-reads system + tools + history + message + earlier results,
    priced at the uncached input rate (an upper bound; prompt caching makes real runs cheaper)."""
    model = model or Settings.model_fields["agent_model"].default
    base = _tokens(AGENT_RULES) + _tokens(json.dumps(TOOLS))
    rows = []
    for c in cases:
        state = c["state"]
        context = _tokens(json.dumps(state["items"], ensure_ascii=False)) // 4 + sum(_tokens(t["text"]) for t in state["turns"])
        first = base + context + _tokens(c["message"]) + 60  # 60: the trusted header
        calls = min(6, c.get("expected_calls") or max(2, 1 + len(c["must_call"])))
        tin = sum(first + RESULT_TOKENS * k for k in range(calls))
        tout = ANSWER_TOKENS * calls
        usage = SimpleNamespace(input_tokens=tin, output_tokens=tout)
        rows.append({"id": c["id"], "calls": calls, "input_tokens": tin, "output_tokens": tout,
                     "cost": cost_usd(model, usage)})
    return {"cases": rows, "total": sum(r["cost"] for r in rows), "model": model}


def format_estimate(est: dict) -> str:
    lines = [f"Estimated cost ({est['model']}, local, no API call; ~{CHARS_PER_TOKEN} chars/token, uncached input):"]
    for r in est["cases"]:
        lines.append(f"{r['id']:<36} calls={r['calls']} in={r['input_tokens']:>6} out={r['output_tokens']:>5} ${r['cost']:.4f}")
    lines.append(f"TOTAL ${est['total']:.2f} for {len(est['cases'])} cases")
    return "\n".join(lines)


# -- report and CLI -------------------------------------------------------------------------------


def format_report(results: list[Result], model: str, when: str) -> str:
    passed = sum(1 for r in results if r.passed)
    lines = [f"# Clawdia agent eval {when}", "", f"Model {model}. Pass {passed}/{len(results)}.", "",
             "| case | pass | tools called | cost | seconds |", "|---|---|---|---|---|"]
    for r in results:
        tools = ", ".join(r.tools) or "-"
        lines.append(f"| {r.id} | {'yes' if r.passed else 'NO'} | {tools} | ${r.cost:.4f} | {r.seconds:.1f} |")
    lines.append(f"| TOTAL | {passed}/{len(results)} | | ${sum(r.cost for r in results):.4f} | "
                 f"{sum(r.seconds for r in results):.1f} |")
    failed = [r for r in results if not r.passed]
    if failed:
        lines += ["", "## Failures", ""]
        for r in failed:
            lines.append(f"### {r.id}")
            lines += [f"- {f}" for f in r.failures]
            lines.append(f"- reply: {(r.reply or '').strip()[:400]!r}")
    return "\n".join(lines).replace(EM, " - ") + "\n"


def make_client(api_key: str) -> Any:
    import anthropic  # lazy: tests and --estimate never import the SDK client
    return anthropic.AsyncAnthropic(api_key=api_key)


async def run_all(cases: list[dict], client: Any) -> list[Result]:
    results = []
    with tempfile.TemporaryDirectory(prefix="clawdia-eval-") as tmp:
        for c in cases:
            results.append(await run_case(c, client, Path(tmp)))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clawdia.agent_eval", description=__doc__.split("\n")[0])
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--estimate", action="store_true", help="print the local cost estimate, no API call")
    p.add_argument("--only", help="comma-separated case ids")
    p.add_argument("--yes-paid", action="store_true", help="required for a real (paid) run")
    args = p.parse_args(argv)

    cases = load_cases(args.cases)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = set(wanted) - {c["id"] for c in cases}
        if unknown:
            print(f"unknown case id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c["id"] in wanted]
    if args.estimate:
        print(format_estimate(estimate(cases)))
        return 0
    if not args.yes_paid:
        print("This run calls the paid model API. Pass --yes-paid to run it, or --estimate to see the cost.",
              file=sys.stderr)
        return 2
    key = os.environ.get(KEY_ENV, "")
    if not key:
        print(f"{KEY_ENV} is not set", file=sys.stderr)
        return 2
    results = asyncio.run(run_all(cases, make_client(key)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out.write_text(format_report(results, Settings.model_fields["agent_model"].default, when), encoding="utf-8")
    passed = sum(1 for r in results if r.passed)
    print(f"{passed}/{len(results)} passed, cost ${sum(r.cost for r in results):.4f}, report {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
