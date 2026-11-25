import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawdia import jobs
from clawdia.bot import on_undo
from clawdia.capture import process
from clawdia.jobs import MAX_IMAGES, job_images, localize, poll_jobs, render_triage, submit_job
from clawdia.second_client import SecondUnavailable, job_idempotency_key
from clawdia.tagger import TagResult
from tests.test_capture import deps, msg

NOW = datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc)
EM = chr(0x2014)

RESULT = {
    "company": "Larkspur", "role": "Product Manager, Payments Platform", "rating": 4, "verdict": "apply",
    "about": "Regulated crypto trading and yield for partners.",
    "why_fit": [{"point": "CEX and regulated KYC", "evidence": "SECRET QUOTE A"},
                {"point": "Stablecoin rails", "evidence": "SECRET QUOTE B"}],
    "risks": ["Remote - Americas, ask about Spain"],
    "cv_suggestion": {"id": "c1", "name": "Product Lead Payments & Crypto", "drive_url": "https://drive.example/secret",
                      "reason": "r", "runners_up": [{"id": "c2", "name": "Head of Product", "reason": "x"}]},
    "official_url": "https://larkspur.example/jobs/1", "injection_suspected": False, "skip_reason": None,
}


def job_deps(tmp_path, **kw):
    d = deps(tmp_path, TagResult("job", "Role", None, {}, 0.9, None, "m"), now=NOW)
    d.second.post_job = AsyncMock(return_value={"id": 41, "status": "queued"})
    d.second.get_job = AsyncMock(return_value={"id": 41, "status": "queued", "error": None, "result": None,
                                              "notion_url": None})
    d.second.patch_job = AsyncMock(return_value={})
    d.bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=900)))
    return d


def routing_job(d, **fields):
    base = {"lang": "ru", "job_intake_id": 41, "job_submitted_at": NOW.isoformat()}
    return d.store.create_item(kind="job", status="routing", title="Role", text="вакансия", chat_id=1,
                               fields={**base, **fields})


def done(result=None, notion="https://notion.so/p1"):
    return {"id": 41, "status": "done", "error": None, "result": result or RESULT, "notion_url": notion}


@pytest.mark.asyncio
async def test_job_capture_posts_and_reply_says_sent_for_review(tmp_path):
    d = job_deps(tmp_path)
    c = await process(msg("https://example.com/job Продакт"), d)
    item = d.store.get_item(c.item_id)
    assert item["status"] == "routing" and item["fields"]["job_intake_id"] == 41 and item["fields"]["job_submitted_at"]
    assert d.second.post_job.await_count == 1 and d.second.post_task.await_count == 0
    assert "-> Sent for review, result in a few minutes" in c.reply_text
    from clawdia.capture import build_reply
    assert "-> Отдала на разбор, пришлю результат через пару минут" in build_reply(item, d.calendar, NOW, "ru")


@pytest.mark.asyncio
async def test_job_capture_when_second_down_says_so_and_retries(tmp_path):
    d = job_deps(tmp_path)
    d.second.post_job = AsyncMock(side_effect=SecondUnavailable("down"))
    c = await process(msg("https://example.com/job"), d)
    item = d.store.get_item(c.item_id)
    assert item["status"] == "routing" and item["route_error"] == "unavailable" and "job_intake_id" not in item["fields"]
    assert "-> Second is down, I will send it when it is back" in c.reply_text
    from clawdia.capture import build_reply
    assert "-> Second недоступен, отправлю, как только ответит" in build_reply(item, d.calendar, NOW, "ru")
    d.second.post_job = AsyncMock(return_value={"id": 5, "status": "queued"})
    await poll_jobs(d, NOW)
    again = d.store.get_item(c.item_id)
    assert again["fields"]["job_intake_id"] == 5 and again["route_error"] is None


def test_job_images_caps_count_and_size(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"{i}.jpg"; p.write_bytes(b"x" * 10); paths.append(str(p))
    big = tmp_path / "big.png"; big.write_bytes(b"x" * 5_000_001)
    doc = tmp_path / "cv.pdf"; doc.write_bytes(b"%PDF")
    sent, skipped = job_images({"media_paths": [str(big), *paths, str(doc)]})
    assert len(sent) == MAX_IMAGES and all(m == "image/jpeg" for m, _ in sent)
    assert skipped == ["big.png", "4.jpg", "5.jpg"]


@pytest.mark.asyncio
async def test_skipped_images_are_named_in_the_reply(tmp_path):
    d = job_deps(tmp_path)
    big = tmp_path / "big.png"; big.write_bytes(b"x" * 5_000_001)
    d.save_media_fn = AsyncMock(return_value=(big, "image/png"))
    c = await process(msg("job", photo=[SimpleNamespace(file_id="a")]), d)
    assert "Not sent for review: big.png" in c.reply_text
    from clawdia.capture import build_reply
    assert "Не отправила на разбор: big.png (больше 4 или больше 5 МБ)" in build_reply(
        d.store.get_item(c.item_id), d.calendar, NOW, "ru")


@pytest.mark.asyncio
async def test_poll_done_sends_triage_and_files_item(tmp_path):
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(return_value=done())
    i = routing_job(d)
    assert await poll_jobs(d, NOW) == 1
    chat, = d.bot.send_message.await_args.args
    text = d.bot.send_message.await_args.kwargs["text"]
    assert chat == 1 and text.startswith("Larkspur - Product Manager") and "https://notion.so/p1" in text
    item = d.store.get_item(i)
    assert item["status"] == "filed" and item["filed_ref"] == "https://notion.so/p1"
    assert item["fields"]["triage_msg_id"] == 900 and item["fields"]["triage_sent_at"]
    turn = d.store.recent_turns(1, 5)[-1]
    assert turn["role"] == "bot" and turn["item_id"] == i and turn["text"] == text
    assert turn["source"] == "triage"
    await poll_jobs(d, NOW)  # filed items are not polled again
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_poll_done_falls_back_to_owner_and_duplicate_url(tmp_path):
    d = job_deps(tmp_path)
    res = {**RESULT, "duplicate_of": {"page_id": "p", "url": "https://notion.so/dup"}}
    d.second.get_job = AsyncMock(return_value=done(res, notion=None))
    i = d.store.create_item(kind="job", status="routing", title="R", chat_id=None,
                            fields={"lang": "en", "job_intake_id": 41, "job_submitted_at": NOW.isoformat()})
    await poll_jobs(d, NOW)
    assert d.bot.send_message.await_args.args == (1,)  # owner_id
    assert d.store.get_item(i)["filed_ref"] == "https://notion.so/dup"


@pytest.mark.asyncio
async def test_poll_skip_dismisses_with_reason(tmp_path):
    d = job_deps(tmp_path)
    res = {**RESULT, "verdict": "skip", "skip_reason": "CFD broker"}
    d.second.get_job = AsyncMock(return_value=done(res, notion=None))
    i = routing_job(d)
    await poll_jobs(d, NOW)
    item = d.store.get_item(i)
    assert item["status"] == "dismissed" and item["notes"] == "CFD broker"
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_poll_failed_keeps_item_in_inbox(tmp_path):
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(return_value={"id": 41, "status": "failed", "error": "boom", "result": None})
    i = routing_job(d)
    await poll_jobs(d, NOW)
    item = d.store.get_item(i)
    assert item["status"] == "new" and item["route_error"] == "triage_failed"
    assert d.bot.send_message.await_count == 1
    await poll_jobs(d, NOW)
    assert d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_poll_cancelled_dismisses_silently(tmp_path):
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(return_value={"id": 41, "status": "cancelled", "error": None, "result": None})
    i = routing_job(d)
    await poll_jobs(d, NOW)
    assert d.store.get_item(i)["status"] == "dismissed" and d.bot.send_message.await_count == 0


@pytest.mark.asyncio
async def test_delay_message_once_after_15_min(tmp_path):
    d = job_deps(tmp_path)
    i = routing_job(d)
    await poll_jobs(d, NOW + timedelta(minutes=14))
    assert d.bot.send_message.await_count == 0
    await poll_jobs(d, NOW + timedelta(minutes=16))
    await poll_jobs(d, NOW + timedelta(minutes=20))
    assert d.bot.send_message.await_count == 1
    assert "задерживается" in d.bot.send_message.await_args.kwargs["text"]
    assert d.store.get_item(i)["fields"]["delay_noticed"] is True and d.store.get_item(i)["status"] == "routing"


@pytest.mark.asyncio
async def test_spend_cap_message_once_per_day(tmp_path):
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(return_value={"id": 41, "status": "queued", "error": "spend_cap", "result": None})
    a, b = routing_job(d), routing_job(d)
    later = NOW + timedelta(minutes=30)  # past the delay threshold: R18 still sends no delay message
    await poll_jobs(d, later)
    await poll_jobs(d, later + timedelta(minutes=5))
    assert d.bot.send_message.await_count == 1
    text = d.bot.send_message.await_args.kwargs["text"]
    assert "лимит" in text and f"#{a}" in text and f"#{b}" in text  # every vacancy waiting under the cap is named
    assert not d.store.get_item(a)["fields"].get("delay_noticed")
    await poll_jobs(d, later + timedelta(days=1))
    assert d.bot.send_message.await_count == 2


def test_render_triage_ru_matches_template():
    text = render_triage(RESULT, "ru", 7)
    assert text.splitlines() == [
        "Larkspur - Product Manager, Payments Platform · 4/5 · подаваться",
        "Про что: Regulated crypto trading and yield for partners.",
        "Почему ты: CEX and regulated KYC, Stablecoin rails",
        "Риски: Remote - Americas, ask about Spain",
        'CV: "Product Lead Payments & Crypto" (запасные: Head of Product)',
        "вакансия: https://larkspur.example/jobs/1",
    ]
    assert "SECRET QUOTE" not in text and "drive.example" not in text and EM not in text
    with_notion = render_triage({**RESULT, "notion_url": "https://notion.so/p1"}, "ru", 7)
    assert with_notion.splitlines()[-1] == "Notion: https://notion.so/p1 · вакансия: https://larkspur.example/jobs/1"


def test_render_triage_en():
    text = render_triage({**RESULT, "notion_url": "https://notion.so/p1", "verdict": "maybe"}, "en", 7)
    lines = text.splitlines()
    assert lines[0].endswith("· 4/5 · maybe") and lines[1].startswith("About: ") and lines[2].startswith("Why you: ")
    assert lines[3].startswith("Risks: ") and lines[4].startswith('CV: "Product Lead') and "(alternatives:" in lines[4]
    assert lines[5] == "Notion: https://notion.so/p1 · posting: https://larkspur.example/jobs/1"


def test_render_triage_flags_injection():
    ru = render_triage({**RESULT, "injection_suspected": True}, "ru", 7)
    assert "Внимание: в тексте вакансии были инструкции, я их не выполняла" in ru
    assert "Внимание" not in render_triage(RESULT, "ru", 7)
    assert len(render_triage({**RESULT, "about": "x" * 9000}, "en", 7)) <= 4000


def fake_ai(payload):
    block = SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(content=[block]))))


@pytest.mark.asyncio
async def test_localize_translates_only_the_three_fields(tmp_path):
    d = job_deps(tmp_path)
    res = {**RESULT, "about": "Trading for 3 partners.", "risks": ["Remote in 2 regions"]}
    d.ai_client = fake_ai({"about": "Торговля для 3 партнёров.", "why_fit": ["CEX и KYC", "Стейблкоин-рейлы"],
                           "risks": ["Удалёнка в 2 регионах"]})
    out = await localize(res, "ru", d)
    assert out["about"] == "Торговля для 3 партнёров." and out["why_fit"][0] == {"point": "CEX и KYC", "evidence": "SECRET QUOTE A"}
    assert out["risks"] == ["Удалёнка в 2 регионах"] and out["company"] == "Larkspur"
    assert d.ai_client.messages.create.await_args.kwargs["model"] == d.settings.tag_model
    assert await localize(res, "en", d) is res


@pytest.mark.asyncio
async def test_localize_rejects_new_digits(tmp_path):
    d = job_deps(tmp_path)
    res = {**RESULT, "about": "Trading for 3 partners."}
    d.ai_client = fake_ai({"about": "Торговля для 30 партнёров.", "why_fit": ["a", "b"], "risks": ["c"]})
    assert await localize(res, "ru", d) == res
    d.ai_client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("x"))))
    assert await localize(res, "ru", d) == res


@pytest.mark.asyncio
async def test_poll_done_ru_uses_localized_text(tmp_path):
    d = job_deps(tmp_path)
    d.ai_client = fake_ai({"about": "Крипто-торговля.", "why_fit": ["пункт один", "пункт два"], "risks": ["риск"]})
    d.second.get_job = AsyncMock(return_value=done())
    routing_job(d)
    await poll_jobs(d, NOW)
    assert "Про что: Крипто-торговля." in d.bot.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_undo_on_job_sends_skip(tmp_path):
    d = job_deps(tmp_path)
    i = routing_job(d)
    q = SimpleNamespace(data=f"undo:{i}", answer=AsyncMock(), message=SimpleNamespace(edit_text=AsyncMock()),
                        from_user=SimpleNamespace(id=1))
    await on_undo(q, d)
    d.second.patch_job.assert_awaited_once_with(41, "skip", "Undone in Clawdia")
    assert d.store.get_item(i)["status"] == "dismissed"


@pytest.mark.asyncio
async def test_undo_skip_failure_is_retried_by_the_poll_loop(tmp_path):
    d = job_deps(tmp_path)
    d.second.patch_job = AsyncMock(side_effect=SecondUnavailable("down"))
    i = routing_job(d)
    q = SimpleNamespace(data=f"undo:{i}", answer=AsyncMock(), message=SimpleNamespace(edit_text=AsyncMock()),
                        from_user=SimpleNamespace(id=1))
    await on_undo(q, d)
    assert d.store.get_item(i)["fields"]["job_decision_pending"] == "skip"
    d.second.patch_job = AsyncMock(return_value={})
    await poll_jobs(d, NOW)
    d.second.patch_job.assert_awaited_once_with(41, "skip", "Undone in Clawdia")
    assert "job_decision_pending" not in d.store.get_item(i)["fields"]


@pytest.mark.asyncio
async def test_same_duplicate_is_not_submitted(tmp_path):
    d = job_deps(tmp_path)
    d.tag_fn = AsyncMock(return_value=TagResult("job", "PM at ORIOLE", None, {
        "company": "ORIOLE", "role": "Product Manager", "url": "https://lnkd.in/aBcDeF12"}, 0.9, None, "m"))
    first = await process(msg("https://lnkd.in/aBcDeF12"), d)
    d.now_fn = lambda: NOW + timedelta(hours=1)
    second = await process(msg("https://lnkd.in/aBcDeF12", mid=2), d)
    assert second.item_id != first.item_id and d.store.get_item(second.item_id)["status"] == "dismissed"
    assert d.second.post_job.await_count == 1


def test_idempotency_key_shape():
    item = {"id": 12, "created_at": "2026-11-09T10:00:00+00:00"}
    assert job_idempotency_key(item) == "clawdia-job-12-1794218400000000"


def test_no_prinyala_in_sources():
    root = Path(jobs.__file__).parent
    hits = [p.name for p in root.rglob("*.py") if "приняла" in p.read_text(encoding="utf-8").lower()]
    assert hits == []


def test_no_em_dash_in_jobs_source():
    assert EM not in Path(jobs.__file__).read_text(encoding="utf-8")


def test_inbox_shows_jobs_in_triage_and_needing_a_look(tmp_path):
    from clawdia.render import render_inbox
    d = job_deps(tmp_path)
    routing_job(d)
    d.store.create_item(kind="job", status="new", title="x", route_error="triage_failed")
    out = render_inbox(d.store, None, NOW)
    assert "jobs in triage: 1" in out and "jobs needing a look: 1" in out and "waiting for session: 1 jobs" in out


# -- fix round 1 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_does_not_claim_sent_without_an_intake_id(tmp_path):
    from clawdia.capture import build_reply
    from clawdia.second_client import SecondAuthFailed
    d = job_deps(tmp_path)
    d.second.post_job = AsyncMock(side_effect=SecondAuthFailed("http_401"))
    c = await process(msg("https://example.com/job"), d)
    item = d.store.get_item(c.item_id)
    assert item["route_error"] == "unauthorized" and "Sent for review" not in c.reply_text
    assert "-> Second rejects my key, the vacancy is waiting" in c.reply_text
    assert "-> Second не принимает мой ключ, вакансия ждёт в разборе" in build_reply(item, d.calendar, NOW, "ru")
    other = d.store.create_item(kind="job", status="routing", title="x", route_error="http_500", fields={})
    assert "Second is down" in build_reply(d.store.get_item(other), d.calendar, NOW, "en")


def test_filed_job_reply_shows_the_notion_link(tmp_path):
    from clawdia.capture import build_reply
    d = job_deps(tmp_path)
    i = d.store.create_item(kind="job", status="filed", title="x", filed_ref="https://notion.so/p1")
    text = build_reply(d.store.get_item(i), d.calendar, NOW, "en")
    assert "-> Notion: https://notion.so/p1" in text and "Second #" not in text


@pytest.mark.asyncio
async def test_poll_401_keeps_routing_and_sends_nothing(tmp_path):
    from clawdia.second_client import SecondAuthFailed
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(side_effect=SecondAuthFailed("http_401"))
    i = routing_job(d)
    await poll_jobs(d, NOW)
    assert d.store.get_item(i)["status"] == "routing" and d.bot.send_message.await_count == 0


@pytest.mark.asyncio
async def test_client_maps_401_on_get_and_patch_to_auth_failed():
    import httpx
    from clawdia.second_client import SecondAuthFailed, SecondClient
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"code": "unauthorized", "message": "x"}})), base_url="http://t")
    c = SecondClient("http://t", "tok", http)
    with pytest.raises(SecondAuthFailed):
        await c.get_job(1)
    with pytest.raises(SecondAuthFailed):
        await c.patch_job(1, "skip", None)


@pytest.mark.asyncio
async def test_poll_404_sends_one_failed_message(tmp_path):
    from clawdia.second_client import SecondRejected
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(side_effect=SecondRejected("not_found", "", 404))
    i = routing_job(d)
    await poll_jobs(d, NOW)
    await poll_jobs(d, NOW)
    item = d.store.get_item(i)
    assert item["status"] == "new" and item["route_error"] == "triage_failed" and d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_poll_rejected_resubmit_sends_one_failed_message(tmp_path):
    from clawdia.second_client import SecondRejected
    d = job_deps(tmp_path)
    d.second.post_job = AsyncMock(side_effect=SecondRejected("idempotency_key_conflict", "", 409))
    i = d.store.create_item(kind="job", status="routing", title="x", text="t", chat_id=1, fields={})
    await poll_jobs(d, NOW)
    await poll_jobs(d, NOW)
    item = d.store.get_item(i)
    assert item["status"] == "new" and item["route_error"] == "triage_failed" and d.bot.send_message.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["text", {**RESULT, "duplicate_of": "oops"}, {**RESULT, "why_fit": "x"}])
async def test_poll_malformed_result_fails_once(tmp_path, bad):
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(return_value=done(bad))
    i = routing_job(d)
    await poll_jobs(d, NOW)
    await poll_jobs(d, NOW)
    item = d.store.get_item(i)
    assert item["status"] == "new" and item["route_error"] == "triage_failed" and d.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_working_row_with_stale_spend_cap_sends_no_cap_message(tmp_path):
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(return_value={"id": 41, "status": "working", "error": "spend_cap", "result": None})
    routing_job(d)
    await poll_jobs(d, NOW)
    assert d.bot.send_message.await_count == 0


@pytest.mark.asyncio
async def test_delay_message_goes_out_while_second_is_down(tmp_path):
    d = job_deps(tmp_path)
    d.second.get_job = AsyncMock(side_effect=SecondUnavailable("down"))
    routing_job(d)
    routing_job(d)
    for minutes in (30, 31):
        await poll_jobs(d, NOW + timedelta(minutes=minutes))
    assert d.bot.send_message.await_count == 2  # one per vacancy, never repeated
    assert "задерживается" in d.bot.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_messages_follow_the_chat_language_not_the_posting(tmp_path):
    d = job_deps(tmp_path)
    d.store.add_turn(1, "user", "что по вакансиям?")
    d.second.get_job = AsyncMock(return_value=done())
    routing_job(d, lang="en")
    await poll_jobs(d, NOW)
    assert "Про что:" in d.bot.send_message.await_args.kwargs["text"]
    e = job_deps(tmp_path / "en")
    e.store.add_turn(1, "user", "what about the jobs?")
    e.second.get_job = AsyncMock(return_value=done())
    routing_job(e, lang="ru")
    await poll_jobs(e, NOW)
    assert "About:" in e.bot.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_failed_send_does_not_repeat_the_localize_call(tmp_path):
    d = job_deps(tmp_path)
    d.ai_client = fake_ai({"about": "Крипто.", "why_fit": ["раз", "два"], "risks": ["риск"]})
    d.second.get_job = AsyncMock(return_value=done())
    d.bot.send_message = AsyncMock(side_effect=[RuntimeError("tg"), SimpleNamespace(message_id=5)])
    i = routing_job(d)
    await poll_jobs(d, NOW)
    assert d.store.get_item(i)["status"] == "routing"
    await poll_jobs(d, NOW)
    assert d.ai_client.messages.create.await_count == 1 and d.store.get_item(i)["status"] == "filed"
    assert "Про что: Крипто." in d.bot.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_concurrent_pending_skip_survives_triage_fields(tmp_path):
    d = job_deps(tmp_path)
    i = routing_job(d)
    real = d.second.get_job

    async def get(_id):
        fields = dict(d.store.get_item(i)["fields"]); fields["job_decision_pending"] = "skip"
        d.store.update_item(i, fields=fields)
        return done()
    d.second.get_job = get
    await poll_jobs(d, NOW)
    assert d.store.get_item(i)["fields"].get("job_decision_pending") == "skip"


@pytest.mark.asyncio
async def test_pending_skip_kept_on_auth_failure(tmp_path):
    from clawdia.second_client import SecondAuthFailed
    d = job_deps(tmp_path)
    d.second.patch_job = AsyncMock(side_effect=SecondAuthFailed("http_401"))
    i = d.store.create_item(kind="job", status="dismissed", title="x",
                            fields={"job_intake_id": 41, "job_decision_pending": "skip"})
    await poll_jobs(d, NOW)
    assert d.store.get_item(i)["fields"]["job_decision_pending"] == "skip"


@pytest.mark.asyncio
async def test_undo_then_requeue_reopens_the_vacancy_and_poll_reports_it(tmp_path):
    """Skip on a queued intake -> server cancelled -> item dismissed; the later decision re-queues the
    server row (R5): the item must go back to routing so the result reaches Elena."""
    from clawdia.jobs import reopen_if_requeued
    d = job_deps(tmp_path)
    i = routing_job(d, delay_noticed=True, triage_sent_at="x", triage_text="old")
    d.second.get_job = AsyncMock(return_value={"id": 41, "status": "cancelled", "error": None, "result": None})
    await poll_jobs(d, NOW)
    assert d.store.get_item(i)["status"] == "dismissed"
    assert reopen_if_requeued(d, i, "skip", {"status": "queued"}) is False  # a skip never reopens
    assert reopen_if_requeued(d, i, "apply", {"status": "queued"}) is True
    item = d.store.get_item(i)
    assert item["status"] == "routing" and item["fields"]["job_intake_id"] == 41
    assert not {"delay_noticed", "triage_sent_at", "triage_text"} & set(item["fields"])
    d.second.get_job = AsyncMock(return_value=done())
    assert await poll_jobs(d, NOW) == 1
    assert d.bot.send_message.await_count == 1 and d.store.get_item(i)["status"] == "filed"


@pytest.mark.asyncio
async def test_clear_decision_requeue_reopens_dismissed_vacancy(tmp_path):
    from clawdia.bot import _clear_decision
    d = job_deps(tmp_path)
    i = d.store.create_item(kind="job", status="dismissed", title="x", chat_id=1, fields={"job_intake_id": 41})
    d.second.patch_job = AsyncMock(return_value={"id": 41, "status": "queued"})
    await _clear_decision(d, i, 41)
    assert d.store.get_item(i)["status"] == "routing"


@pytest.mark.asyncio
async def test_pending_clear_retry_reopens_dismissed_vacancy(tmp_path):
    d = job_deps(tmp_path)
    i = d.store.create_item(kind="job", status="dismissed", title="x", chat_id=1,
                            fields={"job_intake_id": 41, "job_decision_pending": "clear"})
    d.second.patch_job = AsyncMock(return_value={"id": 41, "status": "working"})
    d.second.get_job = AsyncMock(return_value=done())
    await poll_jobs(d, NOW)
    assert d.store.get_item(i)["status"] == "filed" and d.bot.send_message.await_count == 1


def test_triage_contract_fixture_renders_runners_up_without_evidence_or_drive_url():
    import json
    from pathlib import Path
    from clawdia import agent_tools
    from clawdia.jobs import render_triage
    result = json.loads((Path(__file__).parent / "fixtures" / "triage_result.json").read_text(encoding="utf-8"))
    for lang in ("ru", "en"):
        text = render_triage(result, lang, 5)
        assert "Generalist PM" in text and "Product Lead Payments" in text
        assert "docs.google.com" not in text and "Led a card payments product" not in text
    view = agent_tools._triage(result)
    assert view["cv"] == "Product Lead Payments" and view["cv_runners_up"] == ["Generalist PM"]
    dumped = json.dumps(view)
    assert "docs.google.com" not in dumped and "Led a card payments product" not in dumped


@pytest.mark.asyncio
async def test_ops_health_maps_401_to_auth_failed():
    import httpx
    from clawdia.second_client import SecondAuthFailed, SecondClient
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(403, json={"error": {"code": "forbidden", "message": "x"}})), base_url="http://t")
    with pytest.raises(SecondAuthFailed):
        await SecondClient("http://t", "tok", http).get_ops_health()
