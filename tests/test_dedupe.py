"""A vacancy captured twice is one card. Regression: Clawdia #10 and #11 (2026-11-05 22:54 and
2026-11-06 14:43) were the same LinkedIn post with the same short link and became two items."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from clawdia.capture import build_reply, process
from clawdia.converse import handle_text
from clawdia.dedupe import company_key, posting_keys, role_key, url_key
from clawdia.tagger import TagResult
from clawdia.understand import Understanding
from tests.test_capture import deps, msg
from tests.test_converse import make as make_converse
from tests.test_converse import msg as converse_msg

T0 = datetime(2026, 11, 5, 21, 54, tzinfo=timezone.utc)
SHORT = "https://lnkd.in/aBcDeF12"


def job(url=SHORT, company="ORIOLE", role="Product Manager", title="Product Manager at ORIOLE (London or Europe)"):
    fields = {"company": company, "role": role, "location": "London or elsewhere in Europe"}
    if url:
        fields["url"] = url
    return TagResult("job", title, None, fields, 0.95, None, "claude-haiku-5-5")


# -- keys -----------------------------------------------------------------


def test_short_link_is_its_own_id_and_keeps_case():
    assert url_key(SHORT) == url_key("lnkd.in/aBcDeF12") == url_key(SHORT + ".")
    assert url_key(SHORT) != url_key("https://lnkd.in/abcdef12")
    assert url_key(SHORT) != url_key("https://lnkd.in/other123")


def test_linkedin_job_forms_share_one_key():
    plain = url_key("https://www.linkedin.com/jobs/view/4000000002/")
    assert plain == "linkedin:job:4000000002"
    assert url_key("https://linkedin.com/jobs/view/product-manager-at-oriole-4000000002?trackingId=abc&refId=x") == plain
    assert url_key("https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4000000002") == plain


def test_ats_requisitions_match_across_hosts():
    assert url_key("https://boards.greenhouse.io/copperline/jobs/5000000001") == \
        url_key("https://job-boards.eu.greenhouse.io/copperline/jobs/5000000001?utm_source=x") == "gh:5000000001"
    assert url_key("https://copperline.example/careers/open?gh_jid=5000000001") == "gh:5000000001"
    ashby = "https://jobs.ashbyhq.com/quill/0a1b2c3d-1111-4222-8333-444455556666"
    assert url_key(ashby) == url_key(ashby + "/application") == url_key("https://quill.example/careers?ashby_jid=0a1b2c3d-1111-4222-8333-444455556666")


def test_notion_page_key_ignores_slug():
    a = url_key("https://app.notion.com/p/oriole/Senior-Product-Manager-a1b2c3d4e5f60718293a4b5c6d7e8f90")
    assert a == "notion:a1b2c3d4e5f60718293a4b5c6d7e8f90"
    assert url_key("https://www.notion.so/a1b2c3d4e5f60718293a4b5c6d7e8f90") == a


@pytest.mark.parametrize("url", [
    "https://quill.example", "https://quill.example/", "https://quill.example/careers", "https://www.linkedin.com/in/jane-doe-example",
    "https://www.linkedin.com/company/oriole/jobs", "https://t.me/somechannel", "not a link", "", None,
])
def test_links_that_do_not_name_one_posting_have_no_key(url):
    assert url_key(url) is None


def test_two_telegram_posts_of_one_channel_differ():
    assert url_key("https://t.me/jobs_ch/101") != url_key("https://t.me/jobs_ch/102")


def test_company_and_role_normalisation():
    assert company_key("ORIOLE, Inc.") == company_key("oriole") == "oriole"
    assert role_key("Senior Product Manager (London or Europe)") == role_key("Product Manager") == "product manager"
    assert role_key("Lead Product Manager, DeFi at Copperline") == role_key("Product Manager, DeFi")
    assert role_key("Head of Product") != role_key("Product Manager")


def test_posting_keys_read_urls_and_fields():
    item = {"urls": ["https://x.com/web3jobs"], "forward_from": {"link": SHORT},
            "fields": {"url": "https://jobs.ashbyhq.com/quill/0a1b2c3d-1111-4222-8333-444455556666"}}
    assert posting_keys(item) == {url_key(SHORT), "uuid:0a1b2c3d-1111-4222-8333-444455556666"}


# -- the regression ---------------------------------------------------------


@pytest.mark.asyncio
async def test_same_short_link_sixteen_hours_later_is_merged(tmp_path):
    d = deps(tmp_path, job(), now=T0)
    first = await process(msg("", mid=1), d)
    assert first.reply_text.startswith(f"Saved #{first.item_id} · job ·") and first.keyboard
    d.now_fn = lambda: T0 + timedelta(hours=16)
    second = await process(msg("", mid=2), d)
    # created_at is the store's wall clock, so the date in the reply is not the simulated one
    head = f"Already have #{first.item_id} · job · Product Manager at ORIOLE (London or Europe) ("
    assert second.reply_text.startswith(head) and second.reply_text.endswith(", in triage)")
    assert second.keyboard == []
    new = d.store.get_item(second.item_id)
    assert new["status"] == "dismissed" and new["fields"]["duplicate_of"] == first.item_id
    assert d.store.get_item(first.item_id)["status"] == "routing"
    assert [i["id"] for i in d.store.list_items(kind="job", status="routing")] == [first.item_id]
    assert "duplicate" in d.store.event_types(second.item_id) and "resent" in d.store.event_types(first.item_id)


@pytest.mark.asyncio
async def test_russian_reply_for_a_merged_duplicate(tmp_path):
    d = deps(tmp_path, job(), now=T0)
    first = await process(msg("", mid=1), d)
    second = await process(msg("", mid=2), d)
    item = d.store.get_item(second.item_id)
    text = build_reply(item, d.calendar, T0, "ru")
    assert text.startswith(f"Уже есть #{first.item_id} · вакансия · Product Manager at ORIOLE") and "на разборе" in text


@pytest.mark.asyncio
async def test_two_captures_in_flight_leave_one_card(tmp_path):
    async def slow_tag(*_a, **_k):
        await asyncio.sleep(0)  # both items exist before either is tagged
        return job()

    d = deps(tmp_path, job(), now=T0)
    d.tag_fn = slow_tag
    a, b = await asyncio.gather(process(msg("", mid=1), d), process(msg("", mid=2), d))
    statuses = sorted(d.store.get_item(c.item_id)["status"] for c in (a, b))
    assert statuses == ["dismissed", "routing"]
    assert sorted(bool(c.keyboard) for c in (a, b)) == [False, True]


@pytest.mark.asyncio
async def test_resent_screenshot_is_attached_to_the_original(tmp_path):
    media = AsyncMock(side_effect=[(tmp_path / "a.jpg", "image/jpeg"), (tmp_path / "b.jpg", "image/jpeg")])
    d = deps(tmp_path, job(), now=T0, save_media=media)
    first = await process(msg("", photo=[SimpleNamespace(file_id="a")], mid=1), d)
    await process(msg("", photo=[SimpleNamespace(file_id="b")], mid=2), d)
    assert d.store.get_item(first.item_id)["media_paths"] == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]


@pytest.mark.asyncio
async def test_same_company_and_role_with_another_link_is_flagged_not_merged(tmp_path):
    d = deps(tmp_path, job(), now=T0)
    first = await process(msg("", mid=1), d)
    d.tag_fn = AsyncMock(return_value=job(url="https://oriole.example/careers/pm-2", role="Senior Product Manager"))
    second = await process(msg("", mid=2), d)
    assert d.store.get_item(second.item_id)["status"] == "routing"
    assert second.reply_text.startswith(f"Saved #{second.item_id} · job ·")
    assert f"Possible duplicate of #{first.item_id} · Product Manager at ORIOLE" in second.reply_text
    assert second.keyboard == [[("Undo", f"undo:{second.item_id}")]]


@pytest.mark.asyncio
async def test_same_company_another_role_is_a_new_vacancy(tmp_path):
    d = deps(tmp_path, job(), now=T0)
    await process(msg("", mid=1), d)
    d.tag_fn = AsyncMock(return_value=job(url="https://oriole.example/careers/designer", role="Product Designer"))
    second = await process(msg("", mid=2), d)
    assert second.reply_text.startswith("Saved #") and "uplicate" not in second.reply_text


@pytest.mark.asyncio
async def test_undone_original_is_not_a_duplicate_source(tmp_path):
    d = deps(tmp_path, job(), now=T0)
    first = await process(msg("", mid=1), d)
    d.store.update_item(first.item_id, status="dismissed")
    d.store.add_event(first.item_id, "undone", {})
    second = await process(msg("", mid=2), d)
    assert second.reply_text.startswith(f"Saved #{second.item_id} · job ·")
    assert d.store.get_item(second.item_id)["status"] == "routing"


@pytest.mark.asyncio
async def test_original_triaged_out_still_catches_a_resend(tmp_path):
    d = deps(tmp_path, job(), now=T0)
    first = await process(msg("", mid=1), d)
    d.store.update_item(first.item_id, status="dismissed")
    d.store.add_event(first.item_id, "dropped", {"via": "cli"})
    second = await process(msg("", mid=2), d)
    assert second.reply_text.startswith(f"Already have #{first.item_id}") and "dropped" in second.reply_text


@pytest.mark.asyncio
async def test_original_older_than_thirty_days_is_ignored(tmp_path):
    d = deps(tmp_path, job(), now=T0)
    old = d.store.create_item(kind="job", status="new", title="old", created_at=T0 - timedelta(days=31),
                              fields={"company": "ORIOLE", "role": "Product Manager", "url": SHORT})
    second = await process(msg("", mid=2), d)
    assert second.reply_text.startswith(f"Saved #{second.item_id}") and old != second.item_id


@pytest.mark.asyncio
async def test_other_kinds_are_never_merged(tmp_path):
    idea = TagResult("idea", "Read this", None, {"url": SHORT}, 0.9, None, "m")
    d = deps(tmp_path, idea, now=T0)
    await process(msg("", mid=1), d)
    second = await process(msg("", mid=2), d)
    assert second.reply_text.startswith("Saved #2 · idea")


@pytest.mark.asyncio
async def test_a_failing_check_keeps_the_capture(tmp_path):
    d = deps(tmp_path, job(), now=T0)
    with patch("clawdia.capture.resolve", side_effect=RuntimeError("boom")):
        c = await process(msg("", mid=1), d)
    assert c.reply_text.startswith("Saved #1 · job") and d.store.get_item(1)["status"] == "routing"
    assert "error" in d.store.event_types(1)


@pytest.mark.asyncio
async def test_conversation_path_replies_without_the_model_and_without_undo(tmp_path):
    d = make_converse(tmp_path, Understanding(intent="capture", language="ru", capture=job()),
                      Understanding(intent="capture", language="ru", capture=job()),
                      tag=job(), now=T0)
    first = await handle_text(converse_msg("вакансия ORIOLE " + SHORT, mid=1), d)
    calls_before = d.reply_fn.await_count
    second = await handle_text(converse_msg("вакансия ORIOLE " + SHORT, mid=2), d)
    assert second.text.startswith(f"Уже есть #{first.item_id} · вакансия ·")
    assert second.keyboard == [] and d.reply_fn.await_count == calls_before


@pytest.mark.asyncio
async def test_shared_footer_link_does_not_merge_different_vacancies(tmp_path):
    footer = "Find more jobs: https://x.com/web3jobs https://forms.gle/abc https://bit.ly/xyz https://company.com/en"
    d = deps(tmp_path, job(url=None, company="Quill", role="PM", title="PM at Quill"))
    first = await process(msg(f"PM at Quill {footer}", mid=1), d)
    d.tag_fn.return_value = job(url=None, company="Copperline", role="Head of Growth", title="Head of Growth at Copperline")
    second = await process(msg(f"Head of Growth at Copperline {footer}", mid=2), d)
    assert [d.store.get_item(i)["status"] for i in (first.item_id, second.item_id)] == ["routing", "routing"]
    assert first.item_id != second.item_id


@pytest.mark.asyncio
async def test_same_link_but_different_company_is_not_merged(tmp_path):
    d = deps(tmp_path, job(company="ORIOLE"))
    a = await process(msg("a", mid=1), d)
    d.tag_fn.return_value = job(company="Copperline", role="Product Manager")
    b = await process(msg("b", mid=2), d)
    assert d.store.get_item(b.item_id)["status"] == "routing" and a.item_id != b.item_id
