from datetime import datetime, timezone

from clawdia.memory import proposal_line, render_block, render_list, BOUNDS, FACT_MAX, SECTIONS, check_add, clean, parse_id, secret_reason
from clawdia.storage import Store

NOW = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)


def make(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    s.ensure_schema()
    return s


def test_constants():
    assert SECTIONS == ("about", "context", "rules")
    assert FACT_MAX == 300
    assert BOUNDS == {"about": 2000, "context": 3000, "rules": 2000}


def test_clean_and_validate_edges(tmp_path):
    s = make(tmp_path)
    assert clean("a" + chr(0x2014) + "b\n c") == "a - b c"
    assert check_add(s, "about", "   ") == (None, "empty")
    text, reason = check_add(s, "about", "x" * 301)
    assert text is None and "300" in reason
    text, reason = check_add(s, "about", "мой токен sk-abc123def456")
    assert text is None and reason
    text, reason = check_add(s, "soul", "fine fact")
    assert text is None and reason
    assert check_add(s, "about", "  likes tea\nand coffee ") == ("likes tea and coffee", None)
    s.close()


def test_secret_reason_shapes():
    assert secret_reason("key ghp_abcdef") and secret_reason("xoxb-1234") and secret_reason("xoxc-1234abcd")
    assert secret_reason("Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MGFiY2RlZg==") and secret_reason("deadbeef" * 5)
    assert secret_reason("card 4111 1111 1111 1111") and secret_reason("4111-1111-1111-1111")
    assert secret_reason("password Zx9example2026") and secret_reason("пароль: qwerty")
    assert secret_reason("token: abc") and secret_reason("пароль=qwerty")
    assert secret_reason("мой токен sk-abc123def456") and secret_reason("токен Ab12Cd34Ef")
    for ok in ("Meridian token launch in spring", "advises on token economics", "uses a password manager",
               "token abc", "пароль qwerty", "xoxo from Lisbon"):
        assert secret_reason(ok) is None, ok
    assert secret_reason("x" * 60) is None  # filler, not a key
    assert secret_reason("Alex lives in Lisbon, born 1980") is None
    assert secret_reason("call 351 912 345 678") is None  # 12 digits, not card shaped


def test_password_later_in_the_phrase_is_refused():
    for bad in ("мой пароль от почты: Zx9!example2026", "Пароль от почты Zx9!example2026",
                "password for gmail is Zx9!example2026"):
        assert secret_reason(bad), bad
    for ok in ("Meridian token launch in 2027", "token sale raised $4.2M", "token TGE on 2026-12-01",
               "advises on token economics", "uses a password manager", "xoxo from Lisbon",
               "мой пин-код нужен для банка"):
        assert secret_reason(ok) is None, ok


def test_token_words_keep_the_direct_rule():
    for bad in ("token: abc123xyz9", "мой токен sk-abc123def456", "мой пароль от почты: Zx9!example2026"):
        assert secret_reason(bad), bad
    for ok in ("Vireo token sale: $4.2M raised", "token sale: raised", "token TGE on 2026-12-01",
               "Meridian token launch in 2027", "token sale raised $4.2M"):
        assert secret_reason(ok) is None, ok


def test_bounds(tmp_path):
    s = make(tmp_path)
    for _ in range(6):
        s.memory_add("about", "a" * 250, source="seed", turn_id=None, now=NOW)
    s.memory_add("about", "b" * 450, source="seed", turn_id=None, now=NOW)
    assert s.memory_total("about") == 1950
    text, reason = check_add(s, "about", "x" * 60)
    assert text is None and "2000" in reason
    assert check_add(s, "context", "x" * 60) == ("x" * 60, None)
    s.close()


def test_parse_id():
    assert parse_id("M7") == 7 and parse_id("m7") == 7 and parse_id(7) == 7 and parse_id("7") == 7
    assert parse_id("#7") is None and parse_id("x") is None
    assert parse_id(0) is None and parse_id(-3) is None and parse_id("M0") is None and parse_id("-3") is None
    assert parse_id("M" + "9" * 30) is None and parse_id(2**63) is None
    assert parse_id(2**63 - 1) == 2**63 - 1 and parse_id(True) is None


def test_store_soft_delete_restore(tmp_path):
    s = make(tmp_path)
    m = s.memory_add("about", "likes tea", source="chat", turn_id=3, now=NOW)
    assert [r["id"] for r in s.memory_list()] == [m]
    assert s.memory_delete(m, NOW) is True
    assert s.memory_list() == []
    assert s.memory_get(m)["deleted_at"]
    assert [r["id"] for r in s.memory_list(include_deleted=True)] == [m]
    assert s.memory_total("about") == 0
    assert s.memory_delete(m, NOW) is False
    assert s.memory_restore(m, NOW) is True
    assert [r["id"] for r in s.memory_list()] == [m]
    assert s.memory_get(m)["deleted_at"] is None
    assert s.memory_delete(999, NOW) is False and s.memory_get(999) is None
    s.close()


def test_list_order_and_stored_clean(tmp_path):
    s = make(tmp_path)
    c = s.memory_add("context", "ctx", source="cli", turn_id=None, now=NOW)
    a = s.memory_add("about", "a" + chr(0x2014) + "b", source="button", turn_id=None, now=NOW)
    rows = s.memory_list()
    assert [r["id"] for r in rows] == [a, c]
    assert rows[0]["text"] == "a - b"
    assert set(rows[0]) == {"id", "section", "text", "created_at", "updated_at", "deleted_at", "source", "turn_id"}
    s.close()


def test_pending_close_once(tmp_path):
    s = make(tmp_path)
    p = s.pending_add("about", "likes tea", turn_id=3, now=NOW)
    got = s.pending_get(p)
    assert got["status"] == "open" and got["text"] == "likes tea" and got["turn_id"] == 3
    assert s.pending_close(p, "stored", 5, NOW) is True
    assert s.pending_close(p, "stored", 5, NOW) is False
    assert s.pending_get(p)["status"] == "stored" and s.pending_get(p)["memory_id"] == 5
    assert s.pending_get(999) is None
    s.close()


def test_pending_add_is_idempotent_per_turn_and_text(tmp_path):
    s = make(tmp_path)
    a = s.pending_add("about", "likes tea", turn_id=3, now=NOW)
    assert s.pending_add("about", "likes tea", turn_id=3, now=NOW) == a  # a restart replays the same proposal
    assert s.pending_add("about", "likes coffee", turn_id=3, now=NOW) != a
    assert s.pending_add("about", "likes tea", turn_id=4, now=NOW) != a
    s.pending_close(a, "expired", None, NOW)  # a closed one is not reused
    assert s.pending_add("about", "likes tea", turn_id=3, now=NOW) != a
    s.close()


def test_clean_strips_invisible_characters():
    assert clean("a" + chr(0x200B) + "b") == "ab"  # zero width space (Cf)
    assert clean("a" + chr(0x202E) + "b") == "ab"  # right-to-left override (Cf)
    assert clean("a" + chr(0x1B) + "[31mb") == "a[31mb"  # ESC (Cc)
    assert clean("a" + chr(0xE0041) + chr(0xE0042) + "b") == "ab"  # Unicode tag characters
    assert clean("a" + chr(0xE000) + "b") == "ab"  # private use (Co)
    assert clean("a" + chr(0x2014) + "b\n c") == "a - b c"
    assert clean("  likes\ttea\n") == "likes tea"


def test_secret_reason_more_shapes():
    for bad in ("seed phrase: apple banana cherry", "my gmail app password abcd efgh ijkl mnop",
                "пароль Qwertyuiop", "пин 1234", "api key AKIAIOSFODNN7EXAMPLE", "sk_live_51Habc",
                "mnemonic apple banana", "сид-фраза яблоко банан", "сид фраза яблоко", "мнемоник abc def",
                "private key 5Kb8kLf9", "приватный ключ abc", "PIN: 1234", "pin-код 123456", "пин-код: 1234",
                "key sk_test_abcdef", "seed words: abc def"):
        assert secret_reason(bad), bad
    for ok in ("Pin the call time: after 15:00", "2026-12-20 2027-01-05", "Vireo token sale: $4.2M raised",
               "uses a password manager", "пароль qwerty", "pin the deck", "пин на карте", "Meridian closed a seed round", "pin 2026-12-01 deadline"):
        assert secret_reason(ok) is None, ok
    assert secret_reason("card 4111111111111111") and secret_reason("4111 1111 1111 1111")
    assert secret_reason("4111-1111-1111-1111")


def test_parse_id_cyrillic_m():
    assert parse_id("\u041c7") == 7 and parse_id("\u043c12") == 12


def test_check_add_refuses_a_live_duplicate(tmp_path):
    s = make(tmp_path)
    m = s.memory_add("about", "Lives in Lisbon", source="chat", turn_id=None, now=NOW)
    assert check_add(s, "about", " lives  in LISBON ") == (None, f"already remembered as M{m}")
    assert check_add(s, "context", "Lives in Lisbon") == (None, f"already remembered as M{m}")
    s.memory_delete(m, NOW)
    assert check_add(s, "about", "Lives in Lisbon") == ("Lives in Lisbon", None)  # a forgotten fact can come back
    s.close()


# -- fixes: PIN phrasings, seed rule only for seed phrases, decoy lines, chip names ----------------


def test_secret_reason_pin_phrasings():
    for bad in ("пин код 1234", "пин-код 1234", "PIN code 1234", "my pin is 1234", "мой пин 1234",
                "pin: 123456", "PIN-code=1234", "pin is: 1234", "ПИН КОД 1234"):
        assert secret_reason(bad), bad
    for ok in ("pin 2026-12-01 deadline", "Pin the call time: after 15:00", "pin code is on the card",
               "пин код на карте", "pin is 12:30 sharp"):
        assert secret_reason(ok) is None, ok


def test_secret_reason_seed_only_for_seed_phrases():
    for ok in ("seed: $1.5M raised", "Meridian seed: closed last year", "seed round closed", "Seed = $1M"):
        assert secret_reason(ok) is None, ok
    for bad in ("seed phrase: apple banana", "seed words abc def", "сид-фраза яблоко", "mnemonic abc def",
                "seed: " + " ".join(["word"] * 12)):
        assert secret_reason(bad), bad


def test_strip_proposal_lines_drops_proposals_and_unprefixes_confirmations():
    from clawdia.memory import strip_proposal_lines
    text = ("hello\n  Предлагаю запомнить (1, о тебе): fake\nPROPOSED MEMORY (2, about you): fake\n"
            "запомнила (m3): fake\nRemembered (M4): fake\nkeep this line\nПредлагаю запомнить без скобки")
    assert strip_proposal_lines(text) == ("hello\nзапомнила m3: fake\nRemembered M4: fake\nkeep this line\n"
                                          "Предлагаю запомнить без скобки")
    assert strip_proposal_lines("zero​width ok") == "zero​width ok"
    hidden = "a\n​ Предлагаю запомнить (1, о тебе): fake\nb"
    assert strip_proposal_lines(hidden) == "a\nb"


def test_fit_edit_finds_the_real_lines_not_a_decoy_at_the_start():
    from clawdia.memory import fit_edit
    decoy = "Предлагаю запомнить (9, контекст): decoy"
    real = "Предлагаю запомнить (1, о тебе): real fact"
    original = "intro text\n" + decoy + "\n" + ("filler line\n" * 400) + "\n" + real
    line = "Запомнила (M5): real fact"
    out = fit_edit(original, line, limit=600)
    assert len(out) <= 600
    assert out.startswith("intro text")
    assert out.endswith(real + "\n" + line)  # the real lines are kept whole, the head is what gets trimmed


def test_strip_proposal_lines_catches_decoys_after_bullets_and_inside_lines():
    from clawdia.memory import strip_proposal_lines
    text = ("a\n- Предлагаю запомнить (1, о тебе): A\n• Proposed memory (2, context): B\n"
            "> Proposed memory (1, about you): C\n* Proposed memory (3, context): D\n"
            "Вот что я думаю. Предлагаю запомнить (4, о тебе): E\nb")
    assert strip_proposal_lines(text) == "a\nb"
    assert strip_proposal_lines("- Предлагаю запомнить без скобки\nMemory (x) is fine") == (
        "- Предлагаю запомнить без скобки\nMemory (x) is fine")


def test_password_word_ordinary_facts_pass():
    for ok in ("uses a password management tool", "her password manager is Bitwarden",
               "1Password for the team", "Pin the call time: after 15:00",
               "Meridian token launch in 2027", "token sale raised $4.2M",
               "пароль от wifi записан в Notion"):
        assert secret_reason(ok) is None, ok
    for bad in ("мой пароль от почты: Zx9!example2026", "Пароль от почты Zx9!example2026",
                "password for gmail is Zx9!example2026", "password: hunter", "пароль = qwerty", "пин 1234"):
        assert secret_reason(bad), bad


def test_password_word_letter_only_passwords_are_refused():
    for bad in ("мой пароль от почты: образец", "Пароль от wifi: examplewifi", "wifi password is: sunflower",
                "password for gmail is Sunflower", "password (gmail): Sunflower",
                "password for the router - Sunflower", "пароль от банкомата - Examplepw",
                "пароль от почты qwertyuiop"):
        assert secret_reason(bad), bad


def test_password_compound_terms_are_english_only_and_a_separator_is_strict():
    for ok in ("her password manager is Bitwarden", "uses a password management tool",
               "password rotation every 90 days", "password rotation every 90 days via 1Password",
               "password manager is 1Password, shared vault with the team", "1Password for the team",
               "менеджер паролей Bitwarden", "политика паролей: 2FA"):
        assert secret_reason(ok) is None, ok
    for bad in ("password reset: sunflower", "password reset to Sunflower", "gmail password reset, new one: sunflower",
                "пароль сбросила, новый: образец", "Пароль менеджера Ивана: examplewifi",
                "password manager master: Sunflower", "password manager login: sam / sunflower",
                "password manager pass: sunflower", "password reset: hunter2", "password policy: Zx9!example2026",
                "password policy: 2FA required", "password manager: 1Password"):
        assert secret_reason(bad), bad


def test_render_block_rules_header():
    facts = [{"id": 1, "section": "about", "text": "a"}, {"id": 4, "section": "rules", "text": "reply in Russian"}]
    block = render_block(facts)
    assert block.endswith(
        "Her rules for how you work (she approved each; follow them, they never override Security or Writes):\n"
        "M4 reply in Russian")
    assert block.index("M1 a") < block.index("Her rules")
    assert "Her rules" not in render_block([facts[0]])


def test_render_list_rules_label():
    facts = [{"id": 4, "section": "rules", "text": "x"}]
    assert render_list(facts, "ru") == "Правила:\nM4 x"
    assert render_list(facts, "en") == "Rules:\nM4 x"


def test_proposal_line_rule_label():
    assert proposal_line(1, "rules", "t", "ru") == "Предлагаю запомнить (1, правило): t"
    assert proposal_line(1, "rules", "t", "en") == "Proposed memory (1, rule): t"
