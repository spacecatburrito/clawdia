# Clawdia

Clawdia is a personal Telegram bot for one person. You forward, type, photograph or dictate anything into one chat, and she keeps it: tasks, vacancies, events, recipes, films, places, things to buy, ideas. You can also talk to her about what she holds. Typed text goes to a tool-using agent (Anthropic models by default) that can search, change and close your items, set reminders, remember facts about you, and read a separate task and chat database called Second.

She is built for a single owner and tuned to that owner's habits (a Lisbon timezone, Russian and English, a Monday to Friday work week). It is a working personal tool shared as source, not a product or a framework.

## Why I built this

I wanted a personal assistant with memory, scheduled tasks, a character, skills and a way to learn from its own mistakes. I looked at configuring an existing agent (Hermes), but I wasn't sure how to tune it to my own tasks, or how to be sure it was safe and did what I needed. So I wrote my own, small enough that I can read all of it. This repository is that tool, shared as it is.

## How a message flows

1. **Capture.** Every update passes an owner lock first (see Security). Forwards, photos, documents and audio files are always captured; an owner's own voice message goes through the agent like typed text (it is transcribed, answered, and its audio is not kept as a capture). Voice and audio are transcribed (OpenAI), images are stored and shown to the model, documents up to 20 MB and audio up to 600 seconds are accepted.
2. **Understand.** A small model call (`CLAWDIA_TAG_MODEL`, Haiku by default) tags the item into a shelf and pulls out a few fields: kind, title, project, date, confidence. If the call fails, times out (20 s) or returns invalid JSON, the item is saved as `unsorted`; nothing is lost. A low-confidence capture is saved with its best guess and asks one question, at most once per item.
3. **Agent.** Typed text and the owner's own voice messages go to a tool-using agent (`CLAWDIA_AGENT_MODEL`, Sonnet by default) with 35 tools: read and find items, save a note, update, send to Second, drop, mark done, reopen, set or stop reminders, snooze, vacancy decisions, memory tools, the Second read tools, call-wait tools, digest and review settings, rule proposals. Limits per message are 6 model calls and 45 s, and the agent has a daily spend cap (default 2 USD). A code check on every reply sends an unknown `#id` or a link the tools did not return back to the model once. An older engine (`CLAWDIA_BRAIN=classic`) does one understanding call plus one phrasing call instead of the agent loop.
4. **Memory.** Durable facts about the owner (sections `about`, `context`, `rules`) are read by the agent every turn. The last 20 turns of the chat and the last 10 items are kept as conversation context. Facts are added or removed with the `remember` and `forget` tools, the `/memory` command or the `clawdia memory` CLI.
5. **Rules by button.** When you correct how she answered, she may propose one short general rule. It is never stored directly: it appears with a button, and only pressing it stores it. The same button path is used for any memory write that comes from a turn that contained captured text.
6. **Misses and weekly review.** When you ask for something no tool can do, or the data is missing, or you correct her, the miss is logged quietly. On Fridays between 17:00 and 21:00 (owner timezone) she sends one message grouping the misses by topic, with "To backlog" and "Skip" buttons. The backlog is a list for the owner to act on; nothing builds new tools automatically.
7. **Reminders and the outbox.** Work items (task, event, vacancy) are handed to Second by a background loop at any hour, retried while Second is down; personal shelves stay local. Quiet days (outside `CLAWDIA_WORK_DAYS`, default `mon-fri`) do not hold delivery; they affect only the stuck-delivery warning and the ops-alert window. Reminders fire only when an item carries a date or you ask for one, with Done and Snooze buttons. Writes have an Undo button for the local change.
8. **Morning digest.** On weekdays at 09:30 (owner timezone, with catch-up until 12:00) she sends one digest: up to three red lines first, then one line per topic with details in an expandable block. It is built from Second; if Second fails the digest says it is unavailable instead of guessing. The owner changes the time and days by asking in chat.

Plain slash commands (`/list`, `/show`, `/done`, `/drop`, `/inbox`, `/sync`, `/flush`, `/cancel`, `/memory`, `/help`, `/start`) make no model call.

## The five pillars

Status words come from reading the code in this repository, not from plans.

| Pillar | Status | What exists | What does not |
|---|---|---|---|
| Memory | done | Owner facts in three sections with size bounds, injected every turn; 20-turn chat history; secret and card-number refusal; button approval for risky writes; `/memory` and CLI management | No embeddings or semantic search; facts are plain short sentences |
| Scheduled tasks | partial | Built-in schedules: reminders, outbox flush, morning digest (owner can change time and days), Friday review, call waits, nightly backup timer | No user-defined recurring jobs or prompts; every schedule is written in code |
| Soul | partial | The voice and the boundaries live in the system prompts (`agent.py`, `reply.py`, `understand.py`); approved rules extend them | No separate, editable persona document; the owner's name is written into the prompts |
| Skills | not started | A fixed registry of 35 compiled tools | No loadable skills or plug-ins; new abilities mean new code |
| Self-improvement | partial | Corrections become button-approved rules; misses are logged and reviewed weekly; 68 agent eval cases (their format is checked offline; scoring is a paid run) | The backlog is acted on by a person; nothing proposes, tests or ships a new ability by itself, and rules are not re-evaluated |

## Security model

- **Owner lock.** Only the configured owner ids, writing in their own private chat, get any response. Every other update is dropped silently: no reply, no record. A message from the owner in a group is also dropped. Reminders, the morning digest, the stuck-delivery warning and ops alerts go to the primary owner only; resolution notices and vacancy triage messages go to the chat of the item they belong to, which can be an extra owner's chat.
- **Captured content is data, not instructions.** Forwarded text, notes, OCR, transcripts, vacancy postings and anything a Second read returns are wrapped as data for the model. The agent prompt tells the model to report instruction-like text rather than act on it, and the code backs this up where it matters (next points). Eval cases include injection attempts inside forwards, Second messages and transcripts.
- **No write after a Second read.** Once a turn has called one of the Second read tools (`second_tasks`, `second_task`, `second_chats`, `second_messages`, `second_transcripts`, `second_transcript`), `open_tasks` or `today_plan`, the code refuses every write for the rest of that turn. Three writes are exempt because they act on no read content: `pull_transcripts`, `wait_for_call` and `propose_rule` (which only creates a button-approved proposal). `log_miss` is not registered as a write at all, so it also still runs after a Second read; it stores a short model-written line that only shows up in the weekly review (at most 2 per message). The vacancy tools (`list_vacancies`, `get_vacancy`, `find_duplicates`) read Second's `/v1/jobs` triage but do not set this barrier; for them the protection is the data wrapping, the 5 writes and 3 destructive writes per message cap, and the rule that memory writes from a turn holding captured text become proposals. A hostile chat message returned by a Second read cannot make her close or create anything in the same turn.
- **Rules and memory only by button.** `propose_rule` never stores. A memory write from a turn that carried captured or tool text becomes a pending proposal with a button. Rules cannot come from a captured item.
- **Read-only Second access for the agent.** The agent only has read tools for Second (tasks, chats, messages, transcripts). Writes to Second are the explicit hand-off of her own captured tasks, events and vacancies, and patches of items she created.
- **Bounded writes.** At most 5 writes per message and at most 3 destructive ones; each write has an Undo button that reverts the local change (a task that already reached Second stays open there, and Undo says so); replies may contain only links the tools returned.
- **Secrets via environment only.** The settings class reads `CLAWDIA_*` variables, hides input in validation errors, and a test checks that the CLI does not print the bot token. The memory layer refuses to store API keys, card numbers, PINs, seed phrases and similar. Logs carry item ids, never message text or tokens. The systemd unit reads the env file as root and runs the process as an unprivileged user with a hardened sandbox (`NoNewPrivileges`, `ProtectSystem=strict`, empty capability set).
- **Tests and eval.** The public tree holds 1246 tests (37 test modules under `tests/`, plus fixtures); 1245 pass on Python 3.14.5 and one fails in the author's environment (see Tests). The agent eval has 68 cases in `evals/agent_cases.json`. Real eval runs call the paid API and are not part of this repository.

Not covered: the bot trusts Telegram's sender id, the model providers see the text you send her, and nothing here is a security audit.

## Setup

Requirements: Python 3.11 or newer, a Telegram bot token from BotFather, an Anthropic API key, an OpenAI API key (voice transcription, and tagging if you choose that provider), and a reachable Second API (see Limitations).

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/clawdia --help
```

Configuration is read from `CLAWDIA_*` environment variables (`src/clawdia/config.py`). Required, no defaults, startup fails without them:

| Variable | Meaning |
|---|---|
| `CLAWDIA_BOT_TOKEN` | BotFather token |
| `CLAWDIA_OWNER_ID` | Owner's Telegram user id (integer) |
| `CLAWDIA_SECOND_API_TOKEN` | Bearer token for the Second API |
| `CLAWDIA_ANTHROPIC_API_KEY` | Tagging, agent and reply models |
| `CLAWDIA_OPENAI_API_KEY` | Transcription, and tagging when `CLAWDIA_TAG_PROVIDER=openai` |

Optional, with defaults:

| Variable | Default |
|---|---|
| `CLAWDIA_EXTRA_OWNER_IDS` | empty (comma-separated extra user ids) |
| `CLAWDIA_STATE_DIR` | `/var/lib/clawdia` (SQLite file `clawdia.sqlite`, `media/`, context cache) |
| `CLAWDIA_SECOND_API_URL` | `http://127.0.0.1:8091` |
| `CLAWDIA_TAG_MODEL` / `CLAWDIA_TAG_PROVIDER` / `CLAWDIA_OPENAI_TAG_MODEL` | `claude-haiku-5-5` / `anthropic` / `gpt-5.4-mini` |
| `CLAWDIA_BRAIN` | `agent` (or `classic`) |
| `CLAWDIA_AGENT_MODEL` | `claude-sonnet-5-5` |
| `CLAWDIA_AGENT_DAILY_USD` / `CLAWDIA_AGENT_MAX_CALLS` / `CLAWDIA_AGENT_TIMEOUT_S` | `2.0` / `6` / `45` |
| `CLAWDIA_WORK_DAYS` / `CLAWDIA_TIMEZONE` | `mon-fri` / `Europe/Lisbon` |
| `CLAWDIA_HEARTBEAT_MINUTES` / `CLAWDIA_STUCK_MINUTES` | `15` / `60` |
| `CLAWDIA_MAX_AUDIO_SECONDS` / `CLAWDIA_MAX_DOCUMENT_MB` / `CLAWDIA_MODEL_TIMEOUT_S` | `600` / `20` / `20` |
| `CLAWDIA_FOLLOWUP_WINDOW_MIN` / `CLAWDIA_PAIR_WINDOW_S` / `CLAWDIA_ASK_BELOW` | `15` / `4.0` / `0.5` |
| `CLAWDIA_FOCUS_DIR` | `/srv/clawdia-focus` (read-only folder of daily plan files, see Limitations) |
| `CLAWDIA_TG_WEB_ACCOUNT` | empty; set it (for example `2`) to add `?account=<n>` to Telegram Web links for private chats |

Run: `clawdia run`. Other commands read the SQLite file without bot secrets: `inbox`, `show`, `file`, `done`, `dismiss`, `status` (exit 1 when unhealthy), `export-backup --to PATH`, `memory list|add|rm`, `learning [--preview]`, `focus-request take|report`.

**systemd.** `deploy/clawdia.service` expects the app in `/opt/clawdia` (with `.venv`), state in `/var/lib/clawdia`, an env file `/etc/clawdia/env` (root-owned, mode 0600, in a 0700 directory) and a system user `clawdia`. Create the user and directories, install the package non-editable into `/opt/clawdia/.venv`, copy the three unit files from `deploy/` to `/etc/systemd/system/`, copy `deploy/backup.sh` to `/opt/clawdia/deploy/backup.sh` (the backup unit runs it from there), then `systemctl daemon-reload` and `systemctl enable --now clawdia clawdia-backup.timer`. The deploy wrappers the author uses are tied to one machine and are not part of this repository.

**Backups.** `clawdia-backup.timer` runs nightly at 03:30 (owner timezone) and runs `deploy/backup.sh`: a consistent SQLite copy plus `media/` into `/var/backups/clawdia/<date>/`, keeping 14 daily folders. Restore means stopping the service, copying the database and media back, deleting any `-wal` and `-shm` files, and starting again. Check health with `clawdia status`; logs are in journald.

## Tests and eval

```bash
.venv/bin/python -m pytest -q
```

The suite runs offline: Telegram, Second and the model providers are replaced by fakes, and there are no network calls. In the exported tree, 1246 tests are collected and 1245 pass on Python 3.14.5. The one failure, `tests/test_commands.py::test_unknown_command_is_not_captured`, fails in the author's environment (Python 3.14.5) with `AttributeError: 'User' object has no attribute 'active_usernames'` raised through pydantic and aiogram; the cause was not investigated. The private deploy-script tests were removed from the public copy.

The agent eval runs conversations through the real agent loop with a real model, fake Second reads and a fake tagger, and checks which tools were called, how many writes happened and what the reply matched:

```bash
.venv/bin/python -m clawdia.agent_eval --cases evals/agent_cases.json --estimate          # local cost estimate, no API call
.venv/bin/python -m clawdia.agent_eval --cases evals/agent_cases.json --out report.md --yes-paid   # paid run
```

`tests/test_agent_eval.py` checks offline that the cases load and are well formed. Eval scores are not published here. The cases are modelled on real usage, with names, companies, ids and figures replaced by invented ones; treat them as realistic fiction, not as records.

## Limitations

- **Second is a separate service and is not in this repository.** Clawdia talks to it over HTTP (`CLAWDIA_SECOND_API_URL`, a bearer token). The endpoints she calls: `GET /v1/context`, `POST` and `PATCH /v1/tasks`, `POST`, `GET` and `PATCH /v1/jobs`, `POST /v1/read/{tool}`, `POST /v1/transcripts/pull`, `POST /v1/sources/refresh`, `GET /v1/health/ops`, `POST /v1/heartbeat`. The repository contains no Second implementation and no schema for it beyond the client code in `second_client.py`.
- **`CLAWDIA_SECOND_API_TOKEN` is mandatory.** It has no default and the settings class rejects a missing value, so the bot will not start without one. A placeholder token with an unreachable URL does start (a failed context refresh at startup is logged and ignored), but only the Second-independent part works. From the code, without a working Second:
  - Works: capture and tagging into shelves, personal shelves (recipe, film, place, buy, idea, unsorted), reminders, the memory, rules and weekly review, voice transcription, the agent on local items, slash commands, the CLI, backups. Tagging runs without the project list unless a cached context exists.
  - Degrades: task and event items stay in `routing` and are retried; after 60 minutes of failures on a work day she warns the owner once. Vacancies stay in `routing` without triage. The project list for tagging comes from the last cached context.
  - Does not work: sending anything to Second, vacancy triage, the `second_*` read tools (`open_tasks` answers from the cached snapshot only), call transcripts and call waits, the morning digest (it reports itself unavailable), the heartbeat and the Second ops check.
- **The day plan files are produced elsewhere.** `today_plan` reads Markdown plan files from `CLAWDIA_FOCUS_DIR`, and the "Build on laptop" button writes a request that a separate laptop job is meant to pick up (`clawdia focus-request take|report`). The job that writes those files is not in this repository, so without it `today_plan` has nothing to read.
- **One owner.** The prompts name the owner "Elena" and the defaults assume Lisbon time, Russian and English replies and a Monday to Friday work week. Changing the owner name means editing the prompt strings.
- **No web access, no outbound messages.** She has no tool that browses the web, runs commands or messages other people.
- **Provider names.** The default model identifiers in `config.py` are the ones the author runs; check them against your provider before relying on them.
- **License.** MIT, see `LICENSE`.
