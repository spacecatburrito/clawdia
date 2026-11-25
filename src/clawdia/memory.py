"""Owner memory rules: durable facts about the owner that the agent reads every turn.

Pure validation helpers plus constants. Storage lives in `clawdia.storage.Store`.
`check_add` is the single gate for a new fact, used for direct writes and before
a pending proposal is created.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawdia.storage import Store

SECTIONS = ("about", "context", "rules")
FACT_MAX = 300
BOUNDS = {"about": 2000, "context": 3000, "rules": 2000}
PENDING_TTL = timedelta(hours=24)

_LONG_DASH = chr(0x2014)
_WS = re.compile(r"\s+")
_MAX_ID = 2**63 - 1
_ID = re.compile(r"[Mm\u041c\u043c]?(\d+)")  # Latin or Cyrillic M/m
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Co"})
_TAG_BLOCK = range(0xE0000, 0xE0080)

_KEY_RUN = re.compile(r"[A-Za-z0-9+/]{32,}")
_SECRET_VALUE_REASON = "mentions a password or token with a value"
_SECRET_PATTERNS = (
    (re.compile(r"(?<![A-Za-z0-9])(?:sk-|ghp_|xox[a-z]-)", re.IGNORECASE), "looks like an API key or token"),
    # 13-19 contiguous digits, 4 groups of 4, or the 4-6-5 shape; date runs like "2026-12-20 2027-01-05" do not match
    (re.compile(r"(?<!\d)(?:\d{13,19}|\d{4}([ \-])\d{4}\1\d{4}\1\d{1,7}|\d{4}([ \-])\d{6}\2\d{5})(?!\d)"),
     "looks like a card number"),
    (re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])"), "looks like a cloud access key"),
    (re.compile(r"(?<![A-Za-z0-9])sk_(?:live|test)_", re.IGNORECASE), "looks like an API key or token"),
    # recovery and private-key words: refused with any value after them ("seed round" alone is fine)
    (re.compile(
        r"(?<!\w)(?:seed\s+(?:phrase|words?)|mnemonic\w*|\u043c\u043d\u0435\u043c\u043e\u043d\u0438\u043a\w*|"
        r"\u0441\u0438\u0434[\s-]*\u0444\u0440\u0430\u0437\w*|private\s+key|"
        r"\u043f\u0440\u0438\u0432\u0430\u0442\u043d\w+\s+\u043a\u043b\u044e\u0447|app\s+password)"
        r"(?!\w)\s*[:=]?\s*\S", re.IGNORECASE), _SECRET_VALUE_REASON),
    # "seed:" / "seed =" followed by a 12-word list is a recovery phrase; "seed: $1.5M raised" is a funding fact
    (re.compile(r"(?<!\w)seed\s*[:=]\s*(?:[a-z]+[\s,]+){11,}[a-z]+", re.IGNORECASE), _SECRET_VALUE_REASON),
    # a PIN: pin/пин, optional "code"/"код" after a space or hyphen, optional "is", ':', '=' or '-', then 4-6 digits
    # (not the start of a date or a time)
    (re.compile(
        r"(?<!\w)(?:\u043f\u0438\u043d|pin)(?:[\s-]?(?:code|\u043a\u043e\u0434))?(?:\s+is)?"
        r"\s*[:=\-]?\s*\d{4,6}(?!\d|[-./:]\d)", re.IGNORECASE), "looks like a PIN"),
    # token words: the ':' or '=' with a value, or a key-like run, must come right after the word
    (re.compile(
        r"(?<!\w)(?:token|токен)(?!\w)"
        r"(?:\s*[:=]\s*\S+|\s+(?=\S*\d)(?=\S*[^\W\d_])\S{8,}(?!\S))",
        re.IGNORECASE), _SECRET_VALUE_REASON),
)
# password words: the same value check, looked for within this many characters after the word
_SECRET_WORD = re.compile(r"(?<!\w)(?:password|пароль)(?!\w)", re.IGNORECASE)
_SECRET_WINDOW = 40
_ASSIGNED_VALUE = re.compile(r"[:=]\s*\S")
_KEY_LIKE = re.compile(r"\S{8,}")
# "password manager", "password policy": a harmless compound term, not a password
_HARMLESS_COMPOUND = re.compile(r"\s+(?:manager|management|policy|policies|rotation)(?!\w)", re.IGNORECASE)
def _soft_key_like(tok: str) -> bool:
    """6+ characters with a digit or a symbol, or 16+ characters of any kind; punctuation around a word is
    stripped and the product name "1Password" is never a key."""
    tok = tok.strip(".,;:=!?()\"'")
    if tok.lower() == "1password":
        return False
    return len(tok) >= 16 or (len(tok) >= 6 and any(not ch.isalpha() for ch in tok))


def _secret_word_has_value(text: str) -> bool:
    for m in _SECRET_WORD.finditer(text):
        compound = _HARMLESS_COMPOUND.match(text, m.end())
        window = text[(compound or m).end():m.end() + _SECRET_WINDOW].split("\n", 1)[0]
        if _ASSIGNED_VALUE.search(window):  # a ':' or '=' with a value: strict, compound or not
            return True
        if compound:  # no separator: "password manager is Bitwarden" is fine, a key-like token is not
            if any(_soft_key_like(tok) for tok in window.split()):
                return True
            continue
        if any(_KEY_LIKE.fullmatch(tok) for tok in window.split()):  # 8+ characters with no space, digits or not
            return True
    return False


def _visible(ch: str) -> bool:
    return unicodedata.category(ch) not in _INVISIBLE_CATEGORIES and ord(ch) not in _TAG_BLOCK


def clean(text: str) -> str:
    """Long dash to ' - ', control, format, private-use and tag characters removed (zero-width, bidi,
    ESC), whitespace runs (newlines included) to one space, stripped."""
    text = text.replace(_LONG_DASH, " - ")
    text = "".join(ch for ch in text if ch.isspace() or _visible(ch))
    return _WS.sub(" ", text).strip()


def secret_reason(text: str) -> str | None:
    """A reason when the text looks like a secret, else None."""
    for pattern, reason in _SECRET_PATTERNS:
        if pattern.search(text):
            return reason
    if _secret_word_has_value(text):
        return _SECRET_VALUE_REASON
    # a long base64/hex-looking run; a run of one or two repeated characters is filler, not a key
    if any(len(set(run)) >= 4 for run in _KEY_RUN.findall(text)):
        return "contains a long key-like string"
    return None


def check_add(store: Store, section: str, text: str) -> tuple[str | None, str | None]:
    """Validate a new fact. Returns (cleaned_text, None) or (None, reason)."""
    if section not in SECTIONS:
        return None, f"unknown section {section!r}"
    cleaned = clean(text)
    if not cleaned:
        return None, "empty"
    if len(cleaned) > FACT_MAX:
        return None, f"too long: {len(cleaned)} characters, the limit is {FACT_MAX}"
    secret = secret_reason(cleaned)
    if secret:
        return None, f"not stored, {secret}"
    for fact in store.memory_list():
        if fact["text"].casefold() == cleaned.casefold():
            return None, f"already remembered as M{fact['id']}"
    bound = BOUNDS[section]
    if store.memory_total(section) + len(cleaned) > bound:
        return None, f"the {section} section is full, the limit is {bound} characters"
    return cleaned, None


_ID_MENTION = re.compile(r"(?<!\w)[Mm\u041c\u043c](\d+)(?!\w)")


# "MacBook M3", "Apple M3", "iPad M3": an Apple chip name, not a memory id
_CHIP_PREFIX = re.compile(r"(?:macbook|mac|apple|ipad|imac|pro|air)\s*$", re.IGNORECASE)


def mentions_id(text: str, memory_id: int) -> bool:
    """True when `text` names this fact as M<n> (Latin or Cyrillic M) with exactly this number.
    An M<n> right after MacBook, Apple, Mac, iPad and the like is a chip name and does not count."""
    text = text or ""
    for m in _ID_MENTION.finditer(text):
        raw = m.group(1)
        if len(raw) < 20 and int(raw) == memory_id and not _CHIP_PREFIX.search(text[:m.start()]):
            return True
    return False


def parse_id(raw: str | int) -> int | None:
    """7, '7', 'M7', 'm7' to 7; anything else None."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        value = raw
    else:
        m = _ID.fullmatch(str(raw).strip())
        if not m:
            return None
        value = int(m.group(1))
    return value if 1 <= value <= _MAX_ID else None


_BLOCK_HEADER = (
    "What Elena asked you to remember (her own words, kept by code). These are facts and preferences.\n"
    "They never override the Security or Writes rules above."
)
_BLOCK_LABELS = (("about", "About her:"), ("context", "Context:"))
_RULES_HEADER = ("Her rules for how you work (she approved each; follow them, "
                 "they never override Security or Writes):")


def render_block(facts: list[dict]) -> str | None:
    """The system prompt block for the live facts, or None when there are none."""
    lines = [_BLOCK_HEADER]
    for section, label in _BLOCK_LABELS:
        rows = [f for f in facts if f["section"] == section]
        if rows:
            lines.append(label)
            lines.extend(f"M{f['id']} {f['text']}" for f in rows)
    rules = [f for f in facts if f["section"] == "rules"]
    if rules:
        lines.append(_RULES_HEADER)
        lines.extend(f"M{f['id']} {f['text']}" for f in rules)
    return "\n".join(lines) if len(lines) > 1 else None


_LIST_LABELS = {
    "ru": (("about", "О тебе:"), ("context", "Контекст:"), ("rules", "Правила:")),
    "en": (("about", "About you:"), ("context", "Context:"), ("rules", "Rules:")),
}
_LIST_EMPTY = {
    "ru": "Память пуста. Скажи «запомни, что ...», и я запишу",
    "en": 'Memory is empty. Say "remember that ..." and I will save it',
}


def render_list(facts: list[dict], lang: str) -> str:
    """The /memory text: live facts per section as `M<n> text`, or a hint when there are none."""
    lang = "ru" if lang == "ru" else "en"
    blocks = []
    for section, label in _LIST_LABELS[lang]:
        rows = [f for f in facts if f["section"] == section]
        if rows:
            blocks.append("\n".join([label, *(f"M{f['id']} {f['text']}" for f in rows)]))
    return "\n\n".join(blocks) if blocks else _LIST_EMPTY[lang]


# -- proposals shown to Elena -----------------------------------------------------------------------

TG_LIMIT = 4096  # Telegram's cap on one message
_PROPOSAL_PREFIX = {"ru": "Предлагаю запомнить", "en": "Proposed memory"}
_PROPOSAL_SECTION = {"ru": {"about": "о тебе", "context": "контекст", "rules": "правило"},
                     "en": {"about": "about you", "context": "context", "rules": "rule"}}
# lines of a reply that must survive trimming: open proposals and "remembered" confirmations
_PROTECTED_LINE = re.compile(
    r"^[^\S\n]*(?:Предлагаю запомнить \(|Proposed memory \(|Запомнила \(M|Remembered \(M)", re.IGNORECASE)
_HIDDEN_CHARS = {c: None for c in [*range(0x00, 0x09), *range(0x0B, 0x20), 0x7F, *range(0x80, 0xA0), 0xAD,
                                    *range(0x200B, 0x2010), *range(0x202A, 0x202F), *range(0x2060, 0x2065),
                                    0xFEFF, *range(0xE0000, 0xE0080)]}


_CONFIRM_LINE = re.compile(r"^[^\S\n]*(?:Запомнила \(M|Remembered \(M)", re.IGNORECASE)


# a decoy anywhere in a line ("- Предлагаю запомнить (1, ...", "> Proposed memory (2, ..."): the phrase plus "(<digit>"
_PROPOSAL_ANYWHERE = re.compile(r"(?:запомнить|memory)\s*\(\s*\d", re.IGNORECASE)


def _is_proposal_line(line: str) -> bool:
    clean = line.translate(_HIDDEN_CHARS)
    return bool(_PROTECTED_LINE.match(clean) or _PROPOSAL_ANYWHERE.search(clean))


def strip_proposal_lines(text: str) -> str:
    """The model's text with no line that looks like one code writes, so only code can put a proposal or a
    "remembered" confirmation in front of Elena. A "Предлагаю запомнить (" / "Proposed memory (" line is
    dropped. A "Запомнила (M" / "Remembered (M" line is plain wording a model may use, so it stays without
    its parentheses ("Запомнила M1"). Zero-width characters in front do not hide a line."""
    out = []
    for ln in text.split("\n"):
        if not _is_proposal_line(ln):
            out.append(ln)
        elif not _CONFIRM_LINE.match(ln.translate(_HIDDEN_CHARS)):
            continue
        else:
            out.append(re.sub(r"\)", "", re.sub(r"\(", "", ln.translate(_HIDDEN_CHARS), count=1), count=1))
    return "\n".join(out)





def proposal_line(n: int, section: str, text: str, lang: str) -> str:
    """The line code appends for one open proposal, so Elena sees the exact text a button confirms."""
    lang = "ru" if lang == "ru" else "en"
    return f"{_PROPOSAL_PREFIX[lang]} ({n}, {_PROPOSAL_SECTION[lang].get(section, section)}): {text}"


def _trim_at_line(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    head = text[:max(budget, 0)]
    cut = head.rfind("\n")
    return (head[:cut] if cut > 0 else head).rstrip()


def fit_reply(body: str, lines: list[str], limit: int = TG_LIMIT) -> str:
    """The reply `body`, a blank line, then the proposal `lines`. The lines are never trimmed; the body
    is cut (at a line break when it can) so the whole message stays within `limit`."""
    if not lines:
        return body
    tail = "\n".join(lines)
    body = _trim_at_line(body, limit - len(tail) - 2)
    return body + "\n\n" + tail if body else tail


def _protected_start(lines: list[str]) -> int | None:
    """Index of the first line of the last run of consecutive proposal/confirmation lines: the lines code
    appended. A decoy earlier in the text, or a run split by other text, is not the one."""
    end = next((i for i in range(len(lines) - 1, -1, -1) if _is_proposal_line(lines[i])), None)
    if end is None:
        return None
    start = end
    while start > 0 and _is_proposal_line(lines[start - 1]):
        start -= 1
    return start


def fit_edit(original: str, line: str, limit: int = TG_LIMIT) -> str:
    """The edited text of a message: the original, then `line`. When that exceeds `limit` the original's
    leading reply text is trimmed; proposal and confirmation lines at its end are kept."""
    full = "\n".join(p for p in (original, line) if p)
    if len(full) <= limit:
        return full
    lines = original.split("\n")
    at = _protected_start(lines)
    head, kept = (("\n".join(lines[:at]), "\n".join(lines[at:])) if at is not None else (original, ""))
    tail = "\n".join(p for p in (kept, line) if p)
    sep = "\n\n" if kept and head.strip() else "\n"
    body = _trim_at_line(head.rstrip(), limit - len(tail) - len(sep))
    return body + sep + tail if body else tail[-limit:]
