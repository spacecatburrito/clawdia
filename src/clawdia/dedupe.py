"""Duplicate detection for captured vacancies.

A vacancy forwarded twice (the same LinkedIn post, a second screenshot of the same job) must
not become two cards. Pure functions plus one resolver; no network, so a capture never waits
on it and a failure here can never lose the item.

Two strengths of match, newest-first window of `WINDOW_DAYS`:

- "same": the two items share a posting key (the same LinkedIn job id or post id, the same
  Greenhouse / Lever / Ashby requisition, the same lnkd.in short link or Notion page), taken from
  the tagger's url fields and the forward link, and the company and role do not contradict. The new item is merged
  into the earlier one: dismissed with a pointer, its screenshots attached to the original.
- "similar": same company and same role after dropping level words, but no shared link. The new
  item is kept (two different openings can share a title) and only flagged in the reply.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from clawdia.storage import Store

WINDOW_DAYS = 30

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HEX32 = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
_LINKEDIN_JOB_PATH = re.compile(r"/jobs/view/(?:[^/]*-)?(\d{6,})")
_LINKEDIN_POST = re.compile(r"(?:share|activity|ugcpost)[-:](\d{15,})")
_SCHEME_LESS = re.compile(r"^[\w-]+(?:\.[\w-]+)+/", re.IGNORECASE)
_TRACKING = {"trk", "trackingid", "refid", "ref", "src", "source", "fbclid", "gclid", "lipi", "midtoken",
             "midsig", "eid", "otptoken", "trkemail", "rcm", "position", "pagenum", "refererid"}
# Last path segments that list many openings instead of naming one.
_LISTING_SEGMENTS = {"careers", "career", "jobs", "job", "vacancies", "openings", "open-positions",
                     "join-us", "about", "team", "company", "work-with-us", "hiring"}
_COMPANY_SUFFIX = {"inc", "ltd", "limited", "llc", "gmbh", "ag", "bv", "sa", "plc", "corp", "co", "the",
                   "ou", "oü", "ab", "as", "oy", "srl", "sl", "pte"}
_LEVEL_WORDS = {"senior", "sr", "junior", "jr", "lead", "staff", "principal", "associate", "i", "ii", "iii", "iv"}
_BRACKETS = re.compile(r"[\(\[][^)\]]*[\)\]]")
_AT_TAIL = re.compile(r"\s+at\s+.*$", re.IGNORECASE)


def url_key(url: Any) -> str | None:
    """A stable id of the posting a link names, or None when the link does not name one
    (a company homepage, a listing page, a person's profile). Short links are their own id."""
    if not isinstance(url, str):
        return None
    raw = url.strip().rstrip(".,;:!?)")
    if "://" not in raw:
        if not _SCHEME_LESS.match(raw):
            return None
        raw = "https://" + raw
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        query = [(k.lower(), v) for k, v in parse_qsl(parts.query)]
    except ValueError:
        return None
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    path = parts.path
    low = path.lower()
    segments = [s for s in low.split("/") if s]
    q = dict(query)

    if host.endswith("linkedin.com"):
        if segments[:1] in (["in"], ["company"], ["school"], ["pub"]):
            return None
        m = _LINKEDIN_JOB_PATH.search(low)
        if m:
            return f"linkedin:job:{m.group(1)}"
        if re.fullmatch(r"\d{6,}", q.get("currentjobid", "")):
            return f"linkedin:job:{q['currentjobid']}"
        m = _LINKEDIN_POST.search(low)
        if m:
            return f"linkedin:post:{m.group(1)}"
    if host in ("t.me", "telegram.me") and len(segments) < 2:
        return None
    if re.fullmatch(r"\d+", q.get("gh_jid", "")):
        return f"gh:{q['gh_jid']}"
    if host.endswith("greenhouse.io"):
        m = re.search(r"/jobs/(\d+)", low)
        if m:
            return f"gh:{m.group(1)}"
    for value in (*q.values(), low):
        m = _UUID.search(value)
        if m:
            return f"uuid:{m.group(0)}"
    if "notion" in host:
        m = _HEX32.search(low)
        if m:
            return f"notion:{m.group(0)}"
    if not segments or segments[-1] in _LISTING_SEGMENTS:
        return None
    if host not in ("lnkd.in", "t.me", "telegram.me"):  # any other page (a shared footer link, a company page) names no posting
        return None
    kept = sorted(f"{k}={v}" for k, v in query if k not in _TRACKING and not k.startswith("utm_"))
    # the path keeps its case: short-link codes such as lnkd.in/aBcDeF12 are case-sensitive
    return f"url:{host}{path.rstrip('/')}" + (f"?{'&'.join(kept)}" if kept else "")


def company_key(name: Any) -> str:
    words = [w for w in re.findall(r"\w+", str(name or "").lower()) if w not in _COMPANY_SUFFIX]
    return "".join(words)


def role_key(role: Any) -> str:
    text = _AT_TAIL.sub("", _BRACKETS.sub(" ", str(role or "")))
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in _LEVEL_WORDS]
    return " ".join(words)


def posting_keys(item: dict[str, Any]) -> set[str]:
    fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
    # Only links the tagger named as the posting and the forward's own link: a link that merely
    # appears in the text (a job board footer) must not make two vacancies "the same".
    origin = item.get("forward_from")
    urls = [fields.get(k) for k in ("url", "link", "apply_url", "posting_url")]
    urls.append(origin.get("link") if isinstance(origin, dict) else None)
    return {k for k in (url_key(u) for u in urls) if k}


def _counts(store: Store, cand: dict[str, Any]) -> bool:
    """Whether an earlier item can be the original of a new one. A withdrawn capture (Undo, deleted
    in conversation) and an earlier duplicate do not count; a vacancy Elena triaged out does."""
    if cand["status"] in ("new", "routing", "filed", "done"):
        return True
    return cand["status"] == "dismissed" and "dropped" in store.event_types(cand["id"])


def find_duplicate(store: Store, item: dict[str, Any], now: datetime) -> tuple[str, dict[str, Any]] | None:
    keys = posting_keys(item)
    fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
    co, ro = company_key(fields.get("company")), role_key(fields.get("role"))
    if not keys and not (co and ro):
        return None
    similar = None
    candidates = store.list_items(kind="job", since=now - timedelta(days=WINDOW_DAYS), limit=500)
    for cand in reversed(candidates):  # oldest first: the original wins
        if cand["id"] == item["id"] or not _counts(store, cand):
            continue
        cf = cand.get("fields") if isinstance(cand.get("fields"), dict) else {}
        cco, cro = company_key(cf.get("company")), role_key(cf.get("role"))
        contradicts = bool((co and cco and co != cco) or (ro and cro and ro != cro))
        if keys & posting_keys(cand) and not contradicts:
            return "same", cand
        if similar is None and co and ro and company_key(cf.get("company")) == co and role_key(cf.get("role")) == ro:
            similar = cand
    return ("similar", similar) if similar else None


def _pointer(prefix: str, cand: dict[str, Any]) -> dict[str, Any]:
    return {prefix: cand["id"], f"{prefix}_title": cand.get("title") or "",
            f"{prefix}_status": cand["status"], f"{prefix}_at": cand["created_at"]}


def _attach_media(store: Store, source: dict[str, Any], target_id: int) -> None:
    """Add the new item's screenshots to the original (a re-send is often a newer screenshot)."""
    target = store.get_item(target_id)
    have = list(target.get("media_paths") or [])
    new = [p for p in (source.get("media_paths") or []) if p not in have]
    if not new:
        return
    extra = {} if target.get("media_path") else {"media_path": new[0], "media_mime": source.get("media_mime")}
    store.update_item(target_id, media_paths=have + new, **extra)


def resolve(store: Store, item_id: int, now: datetime) -> str | None:
    """Check a just-tagged job against earlier ones and record the outcome on the item.

    Returns "same" (the item is now a dismissed duplicate), "similar" (kept, flagged) or None.
    Nothing awaits between the lookup and the writes, so of two concurrent captures of one
    vacancy the one tagged second always sees the first.
    """
    item = store.get_item(item_id)
    if item is None or item["kind"] != "job":
        return None
    found = find_duplicate(store, item, now)
    if found is None:
        return None
    strength, cand = found
    fields = dict(item.get("fields") or {})
    if strength == "same":
        fields.update(_pointer("duplicate_of", cand))
        store.update_item(item_id, status="dismissed", fields=fields, route_error=None)
        store.disable_reminders(item_id)
        store.add_event(item_id, "duplicate", {"of": cand["id"]})
        store.add_event(cand["id"], "resent", {"by": item_id})
        _attach_media(store, item, cand["id"])
    else:
        fields.update(_pointer("possible_duplicate_of", cand))
        store.update_item(item_id, fields=fields)
        store.add_event(item_id, "possible_duplicate", {"of": cand["id"]})
    return strength
