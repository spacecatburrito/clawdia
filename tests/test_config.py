from pathlib import Path

import pytest
from pydantic import ValidationError

from clawdia.config import Settings

REQUIRED = dict(
    bot_token="test-token",
    owner_id=1,
    second_api_token="test-second",
    anthropic_api_key="test-anthropic",
    openai_api_key="test-openai",
)


def test_defaults():
    s = Settings(**REQUIRED)
    assert s.state_dir == Path("/var/lib/clawdia")
    assert s.second_api_url == "http://127.0.0.1:8091"
    assert s.tag_provider == "anthropic"
    assert s.work_days == "mon-fri"
    assert s.timezone == "Europe/Lisbon"
    assert (s.heartbeat_minutes, s.stuck_minutes) == (15, 60)
    assert (s.max_audio_seconds, s.max_document_mb, s.model_timeout_s) == (600, 20, 20)
    assert (s.followup_window_min, s.ask_below) == (15, 0.5)


def test_conversation_settings_from_env(monkeypatch):
    monkeypatch.setenv("CLAWDIA_FOLLOWUP_WINDOW_MIN", "30")
    monkeypatch.setenv("CLAWDIA_ASK_BELOW", "0.3")
    s = Settings(**REQUIRED)
    assert (s.followup_window_min, s.ask_below) == (30, 0.3)


def test_env_prefix(monkeypatch):
    for k, v in REQUIRED.items():
        monkeypatch.setenv(f"CLAWDIA_{k.upper()}", str(v))
    monkeypatch.setenv("CLAWDIA_STATE_DIR", "/tmp/x")
    s = Settings()
    assert s.owner_id == 1 and s.state_dir == Path("/tmp/x")


def test_missing_required_fails_loudly(monkeypatch):
    monkeypatch.delenv("CLAWDIA_BOT_TOKEN", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_extra_owner_ids_default_is_primary_only():
    s = Settings(**REQUIRED)
    assert s.extra_owner_ids == "" and s.allowed_owner_ids == frozenset({1})


def test_extra_owner_ids_parse_from_env(monkeypatch):
    for k, v in REQUIRED.items():
        monkeypatch.setenv(f"CLAWDIA_{k.upper()}", str(v))
    monkeypatch.setenv("CLAWDIA_EXTRA_OWNER_IDS", "12, 34")
    assert Settings(_env_file=None).allowed_owner_ids == frozenset({1, 12, 34})
    monkeypatch.setenv("CLAWDIA_EXTRA_OWNER_IDS", "")
    assert Settings(_env_file=None).allowed_owner_ids == frozenset({1})


def test_extra_owner_ids_invalid_raises(monkeypatch):
    for bad in ("12,abc", "1.5"):
        with pytest.raises(ValueError, match="CLAWDIA_EXTRA_OWNER_IDS"):
            Settings(**REQUIRED, extra_owner_ids=bad)
    for k, v in REQUIRED.items():
        monkeypatch.setenv(f"CLAWDIA_{k.upper()}", str(v))
    monkeypatch.setenv("CLAWDIA_EXTRA_OWNER_IDS", "12,abc")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_pair_window_default_and_env(monkeypatch):
    from clawdia.config import Settings
    base = dict(bot_token="t", owner_id=1, second_api_token="t", anthropic_api_key="k", openai_api_key="k")
    monkeypatch.delenv("CLAWDIA_PAIR_WINDOW_S", raising=False)
    assert Settings(**base).pair_window_s == 4.0
    monkeypatch.setenv("CLAWDIA_PAIR_WINDOW_S", "2.5")
    assert Settings(**base).pair_window_s == 2.5
