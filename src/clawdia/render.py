"""Pure text for the bot commands: renderers and the store-level done/drop handlers."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from clawdia.kinds import KINDS, WORK_KINDS
from clawdia.storage import Store

PAGE_SIZE = 20
MAX_CHARS = 4000  # Telegram caps a message at 4096
SHOW_SNIPPET = 500
UNKNOWN_SHELF = f"Unknown shelf. Use one of: {' '.join(KINDS)}"

SHELF_ALIASES: dict[str, str] = {
    **{kind: kind for kind in KINDS},
    "tasks": "task", "jobs": "job", "events": "event", "recipes": "recipe", "films": "film",
    "places": "place", "ideas": "idea",
    "фильмы": "film", "места": "place", "рецепты": "recipe", "идеи": "idea", "купить": "buy",
}


def cap(text: str, limit: int = MAX_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _title(item: dict) -> str:
    return " ".join((item.get("title") or "").split()) or "(no title)"


MAX_ID = 2**63 - 1  # SQLite INTEGER; anything larger cannot exist and must not reach the driver


def parse_id(arg: str | None) -> int | None:
    token = (arg or "").strip().lstrip("#")
    return int(token) if token.isascii() and token.isdigit() else None


def find_item(store: Store, item_id: int) -> dict | None:
    return store.get_item(item_id) if item_id <= MAX_ID else None


def second_task_ref(item: dict) -> str | None:
    """`filed_ref` when it is a Second task id (digits); a session filing (e.g. a Notion URL) gives None."""
    ref = str(item.get("filed_ref") or "")
    return ref if ref.isascii() and ref.isdigit() else None


def second_note(item: dict) -> str:
    """Extra line for an item that already lives in Second: closing it here does not close it there."""
    ref = second_task_ref(item)
    return f"\nSecond #{ref} stays open, close it there" if ref else ""


def ago(seconds: int, lang: str = "en") -> str:
    seconds = max(0, int(seconds))
    ru = lang == "ru"
    if seconds < 60:
        return f"{seconds} с назад" if ru else f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} мин назад" if ru else f"{seconds // 60} min ago"
    if seconds < 48 * 3600:
        return f"{seconds // 3600} ч назад" if ru else f"{seconds // 3600}h ago"
    return f"{seconds // 86400} д назад" if ru else f"{seconds // 86400}d ago"


def plural_ru(n: int, one: str, few: str, many: str) -> str:
    """Russian plural form: 1 задача, 2 задачи, 5 задач, 11 задач, 21 задача."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def tasks_word(n: int, lang: str) -> str:
    """"1 task" / "2 tasks" / "1 задача" / "2 задачи" / "5 задач"."""
    if lang == "ru":
        return f"{n} {plural_ru(n, 'задача', 'задачи', 'задач')}"
    return f"{n} task" + ("" if n == 1 else "s")


GREETING_RU = """Привет, я Клавдия. Присылай мне что угодно: текст, голосовое, скриншот, ссылку или пересланное сообщение. Я сохраню и разложу по полкам: задачи, вакансии, события, рецепты, фильмы, места, покупки, идеи.

Говори со мной обычными словами, голосом или текстом. Например:
"покажи фильмы"
"это задача на пятницу" - поправлю то, что только что сохранила
"напомни завтра в 10" или "не напоминай"
"готово" или "удали это"

Кнопка Undo под моим ответом забирает сохранённое обратно. Рабочие задачи я сама передаю в Second."""

GREETING_EN = """Hi, I'm Clawdia. Send me anything: text, a voice note, a screenshot, a link or a forward. I save it and put it on a shelf: tasks, jobs, events, recipes, films, places, things to buy, ideas.

Talk to me in plain words, by voice or text. For example:
"show my films"
"make it a task for Friday" - I fix the item I just saved
"remind me tomorrow at 10" or "no reminders"
"done" or "delete it"

The Undo button under my reply takes a capture back. I pass work tasks to Second myself."""


def greeting(recent_user_texts: list[str]) -> str:
    """The /start and /help text in the language of Elena's latest message; Russian when there is none."""
    for text in reversed(recent_user_texts):
        if text and text.strip() and not text.strip().startswith("/"):
            from clawdia.understand import guess_language  # local: understand imports render indirectly
            return GREETING_EN if guess_language(text) == "en" else GREETING_RU
    return GREETING_RU


HELP_TEXT = """Send me anything and I shelve it. Talk to me normally:
"покажи фильмы" - a shelf, like /list film
"это задача" - fix the item I just saved (kind, project, deadline)
"напомни в пятницу" - a reminder on that item
"не напоминай" - its reminders off
"сохрани это" - save your previous message if I took it for a question
Undo under a reply takes that capture back.
Forwards, photos, documents and voice are always saved.
/list <shelf> [all] [page] - open items on a shelf
/show <id> - everything about one item
/done <id> - mark done
/drop <id> - drop it
/inbox - counts, queue and context age
/sync - refresh context and drain the Second queue
/flush - send queued work to Second now
/memory - what I remember about you
Shelves: task job event recipe film place buy idea unsorted"""


def status_line(store: Store, now: datetime, tz: str = "UTC") -> str:
    """One line: items captured today (local), outbox depth and the last Telegram poll."""
    midnight = now.astimezone(ZoneInfo(tz)).replace(hour=0, minute=0, second=0, microsecond=0)
    today = len(store.list_items(since=midnight, limit=1_000_000))
    queued = len(store.outbox())
    raw = store.meta_get("last_poll_at")
    try:
        poll = ago(int((now - datetime.fromisoformat(raw)).total_seconds())) if raw else "never"
    except (ValueError, TypeError):
        poll = "unknown"
    return f"Captured today: {today}, Second queue: {queued}, last poll: {poll}"


# -- /list --------------------------------------------------------------


def parse_list_args(arg: str | None) -> tuple[str, bool, int] | None:
    """`<shelf> [all] [page]` -> (kind, include_all, page); None when the shelf is unknown."""
    tokens = (arg or "").lower().split()
    kind = SHELF_ALIASES.get(tokens[0]) if tokens else None
    if kind is None:
        return None
    include_all, page = False, 1
    for token in tokens[1:]:
        if token == "all":
            include_all = True
        elif token.isascii() and token.isdigit():
            page = max(1, int(token))
    return kind, include_all, page


def render_list(items: list[dict], shelf: str, page: int, show_status: bool = False) -> str:
    total = len(items)
    pages = max(1, -(-total // PAGE_SIZE))
    page = min(max(1, page), pages)
    head = f"{shelf} · {total} {'total' if show_status else 'open'}" if total else f"{shelf} · 0 open"
    lines = [head]
    for item in items[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        status = f" ({item['status']})" if show_status and item.get("status") != "new" else ""
        lines.append(f"#{item['id']} · {_title(item)}{status}")
    if pages > 1:
        lines.append(f"page {page}/{pages}")
    return cap("\n".join(lines))


# -- /show --------------------------------------------------------------


def _local(stamp: str | None, tz: str) -> str | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp).astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return stamp


def render_show(item: dict, tz: str = "UTC") -> str:
    lines = [f"#{item['id']} · {item['kind']} · {item['status']}", _title(item)]
    if item.get("project"):
        lines.append(f"Project: {item['project']}")
    created = _local(item.get("created_at"), tz)
    if created:
        lines.append(f"Created: {created} ({tz})")
    origin = item.get("forward_from") or {}
    parts = [origin.get(k) for k in ("name", "chat_title", "link")]
    parts = [p for i, p in enumerate(parts) if p and not (i == 1 and p == origin.get("name"))]
    if parts:
        lines.append("From: " + " · ".join(parts))
    for key, value in (item.get("fields") or {}).items():
        lines.append(f"{key}: {value}")
    for label, key in (("Text", "text"), ("Caption", "caption"), ("Transcript", "transcript"), ("OCR", "ocr_text")):
        if item.get(key):
            lines.append(f"{label}: {item[key][:SHOW_SNIPPET]}")
    for url in item.get("urls") or []:
        lines.append(f"URL: {url}")
    if item.get("filed_ref"):
        lines.append(f"Notion: {item['filed_ref']}" if item.get("kind") == "job" else f"Second: #{item['filed_ref']}")
    if item.get("route_error"):
        lines.append(f"Route error: {item['route_error']}")
    media = item.get("media_paths") or []
    if media:
        lines.append(f"Media: {len(media)}")
    return cap("\n".join(lines))


# -- /inbox -------------------------------------------------------------


def render_inbox(store: Store, context_age_s: int | None, now: datetime, tz: str = "UTC") -> str:
    items = store.list_items(status=["new", "routing", "filed"], limit=1_000_000)
    per_kind = {k: sum(1 for i in items if i["kind"] == k) for k in KINDS}
    lines = ["Inbox (new, routing, filed)"]
    lines += [f"{k}: {n}" for k, n in per_kind.items() if n] or ["nothing yet"]
    waiting = {k: sum(1 for i in items if i["kind"] == k and i["status"] == "new") for k in ("job", "event")}
    lines.append(f"waiting for session: {waiting['job']} jobs, {waiting['event']} events")
    in_review = sum(1 for i in items if i["kind"] == "job" and i["status"] == "routing")
    if in_review:
        lines.append(f"jobs in triage: {in_review}")
    routing = [i for i in items if i["status"] == "routing" and i["kind"] in WORK_KINDS and i["kind"] != "job"]
    line = f"routing: {len(routing)}"
    if routing:
        oldest = min(datetime.fromisoformat(i["created_at"]) for i in routing)
        line += f" (oldest {max(0, int((now - oldest).total_seconds() // 60))} min)"
    lines.append(line)
    midnight = now.astimezone(ZoneInfo(tz)).replace(hour=0, minute=0, second=0, microsecond=0)
    filed_today = sum(1 for i in items if i["status"] == "filed" and i.get("filed_at")
                      and datetime.fromisoformat(i["filed_at"]) >= midnight)
    lines.append(f"filed today: {filed_today}")
    failed = sum(1 for i in items if i["kind"] == "job" and i["status"] == "new" and i.get("route_error"))
    if failed:
        lines.append(f"jobs needing a look: {failed}")
    lines.append("context: none" if context_age_s is None else f"context: {ago(context_age_s)}")
    return cap("\n".join(lines))


# -- /done and /drop ----------------------------------------------------


def _close(store: Store, arg: str, command: str, status: str, verb: str, event: str) -> str:
    item_id = parse_id(arg)
    if item_id is None:
        return f"Usage: /{command} <id>"
    item = find_item(store, item_id)
    if item is None:
        return f"No item #{item_id}"
    title = _title(item)
    if item["status"] in ("done", "dismissed"):
        current = "done" if item["status"] == "done" else "dropped"
        return f"Already {current} #{item_id} · {title}{second_note(item)}"
    store.update_item(item_id, status=status, route_error=None)
    store.add_event(item_id, event, {})
    store.disable_reminders(item_id)
    return f"{verb} #{item_id} · {title}{second_note(item)}"


def handle_done(store: Store, arg: str) -> str:
    return _close(store, arg, "done", "done", "Done", "done")


def handle_drop(store: Store, arg: str) -> str:
    return _close(store, arg, "drop", "dismissed", "Dropped", "dropped")
