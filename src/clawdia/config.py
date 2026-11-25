"""Clawdia settings, read from CLAWDIA_* environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Required fields have no defaults so a misconfigured service fails loudly."""

    model_config = SettingsConfigDict(env_prefix="CLAWDIA_", extra="ignore", hide_input_in_errors=True)

    bot_token: str
    owner_id: int  # primary owner: reminders, drain notifications and stuck warnings go here only
    extra_owner_ids: str = ""  # comma-separated Telegram user ids that may also capture and use commands
    state_dir: Path = Path("/var/lib/clawdia")
    second_api_url: str = "http://127.0.0.1:8091"
    second_api_token: str
    anthropic_api_key: str
    openai_api_key: str
    tag_model: str = "claude-haiku-5-5"
    tag_provider: Literal["anthropic", "openai"] = "anthropic"
    openai_tag_model: str = "gpt-5.4-mini"
    work_days: str = "mon-fri"
    timezone: str = "Europe/Lisbon"
    heartbeat_minutes: int = 15
    stuck_minutes: int = 60
    max_audio_seconds: int = 600
    max_document_mb: int = 20
    model_timeout_s: int = 20
    followup_window_min: int = 15  # corrections and follow-ups only this long after the bot's last reply
    pair_window_s: float = 4.0  # a typed comment and a forward this close together become one item (0 = off)
    ask_below: float = 0.5  # a capture asks one question only below this confidence
    brain: Literal["agent", "classic"] = "agent"  # typed text: tool-using agent, or the phase 2 engine
    agent_model: str = "claude-sonnet-5-5"
    agent_daily_usd: float = 2.0  # Europe/Lisbon day
    agent_max_calls: int = 6  # model calls per message
    agent_timeout_s: int = 45  # per message
    focus_dir: Path = Path("/srv/clawdia-focus")  # daily focus maps pushed from the owner's computer (read-only here)
    tg_web_account: str = ""  # Telegram Web account slot for private-chat links ("2" gives web.telegram.org/k/?account=2#id); empty = none

    @field_validator("extra_owner_ids")
    @classmethod
    def _check_extra_owner_ids(cls, value: str) -> str:
        _parse_ids(value)
        return value

    @property
    def allowed_owner_ids(self) -> frozenset[int]:
        return frozenset({self.owner_id, *_parse_ids(self.extra_owner_ids)})


def _parse_ids(raw: str) -> list[int]:
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            raise ValueError("CLAWDIA_EXTRA_OWNER_IDS must be comma-separated integers") from None
    return ids
