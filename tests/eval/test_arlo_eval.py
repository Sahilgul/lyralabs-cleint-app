"""ARLO Task Eval — compare all 6 LLM models on realistic ARLO tasks.

This is a *task-quality* eval, not a speed benchmark. Each test case
mirrors a real request ARLO would receive in Slack. For every (model x case)
pair we measure:

    routing_ok   — did the model pick the right strategy?
                   "plan"   → submitted submit_plan_for_approval
                   "tool"   → called a read tool directly
                   "direct" → replied with text, no tool calls
    tool_hit     — did it name a tool containing the expected substring?
    args_ok      — did it fill in actual args (not empty {})?
    voice_ok     — no forbidden phrases / bad markdown in the text reply?
    latency_s    — wall-clock seconds (first token to full response)
    cost_usd     — LiteLLM reported cost
    output_tokens

Cases cover:
    • GHL CRM tasks  (contact search, opportunity moves, appointment booking)
    • Slack tasks    (history lookup, search, send message, canvas)
    • Multi-step planning (combining read → write across tools)
    • Direct reply   (smalltalk / simple questions — no tools needed)

Results are written to:
    logs/arlo-eval-<timestamp>/summary.json
    logs/arlo-eval-<timestamp>/<model-slug>/case-<id>.json

Run all models:
    pytest tests/eval/test_arlo_eval.py::test_arlo_eval_all -v -s -m live

Run one model:
    pytest tests/eval/test_arlo_eval.py -k "MiniMax M2.7" -v -s -m live

Run one case across all models:
    pytest tests/eval/test_arlo_eval.py -k "ghl-contact-search" -v -s -m live
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from lyra_core.common.config import get_settings
from lyra_core.common.llm import _call_resolved, estimate_cost
from lyra_core.llm.router import ResolvedModel

# ---------------------------------------------------------------------------
# Log directory — created fresh each run
# ---------------------------------------------------------------------------

_RUN_TS = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
_LOG_ROOT = Path(__file__).parent.parent.parent / "logs" / f"arlo-eval-{_RUN_TS}"


# ---------------------------------------------------------------------------
# Tool schemas
# Representative GHL + Slack tools ARLO would have in a real session.
# GHL tools come from MCP at runtime; we define them inline here so the eval
# is fully self-contained (no DB / MCP server needed).
# ---------------------------------------------------------------------------


def _fn(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_READ_TOOLS: list[dict[str, Any]] = [
    # GoHighLevel CRM
    _fn(
        "ghl.contacts.search",
        "Search GoHighLevel contacts by name, email, phone, or tag. Returns matching contacts.",
        {
            "query": {"type": "string", "description": "Name, email, or phone to search for"},
            "limit": {"type": "integer", "description": "Max results (default 20)"},
        },
        ["query"],
    ),
    _fn(
        "ghl.opportunities.list",
        "List GHL opportunities (deals) in a pipeline, filtered by stage or status.",
        {
            "pipeline_id": {"type": "string", "description": "Pipeline ID to query"},
            "stage_name": {
                "type": "string",
                "description": "Stage name filter, e.g. 'Negotiation'",
            },
            "status": {"type": "string", "description": "open | won | lost | abandoned"},
            "limit": {"type": "integer"},
        },
        [],
    ),
    _fn(
        "ghl.conversations.list",
        "List CRM conversations (SMS, email, webchat) from GoHighLevel. Filter by read status or date.",
        {
            "unread_only": {"type": "boolean", "description": "Return only unread conversations"},
            "hours_back": {"type": "integer", "description": "Conversations from last N hours"},
            "contact_id": {"type": "string", "description": "Filter to a specific contact"},
            "limit": {"type": "integer"},
        },
        [],
    ),
    _fn(
        "ghl.pipelines.list",
        "List all GoHighLevel pipelines and their stage names.",
        {},
        [],
    ),
    _fn(
        "ghl.calendars.available_slots",
        "Get available appointment slots from a GoHighLevel calendar.",
        {
            "calendar_id": {"type": "string"},
            "date": {"type": "string", "description": "ISO date e.g. 2026-05-08"},
            "timezone": {"type": "string", "description": "e.g. America/New_York"},
        },
        ["date"],
    ),
    # Slack
    _fn(
        "slack.conversations.history",
        "Fetch recent messages from a Slack channel or DM by channel name or ID.",
        {
            "channel": {"type": "string", "description": "Channel ID or name like #general"},
            "limit": {"type": "integer", "description": "Number of messages to fetch"},
        },
        ["channel"],
    ),
    _fn(
        "slack.search.messages",
        "Full-text search across Slack messages. Supports Slack query modifiers (from:, in:, before:, after:).",
        {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Max results (default 20)"},
        },
        ["query"],
    ),
    _fn(
        "slack.users.lookup_by_email",
        "Look up a Slack user by email. Returns user ID, display name, and profile.",
        {
            "email": {"type": "string", "description": "User email address"},
        },
        ["email"],
    ),
    _fn(
        "slack.users.list",
        "List all members of the Slack workspace.",
        {"limit": {"type": "integer"}},
        [],
    ),
    _fn(
        "slack.conversations.list",
        "List Slack channels and DMs the bot is a member of.",
        {"limit": {"type": "integer"}},
        [],
    ),
    # Meta
    _fn(
        "discover_tools",
        (
            "Search the tool registry by intent. Call before any task to find relevant "
            "tools and their exact argument schemas. Do NOT guess tool names — always call "
            "discover_tools first if you are unsure what tool is available."
        ),
        {
            "intent": {
                "type": "string",
                "description": "What you want to do, e.g. 'send SMS to contact', 'list pipeline deals'",
            },
            "limit": {"type": "integer", "default": 10},
        },
        ["intent"],
    ),
]

# Write tools: NOT in the function-call list, but described in the system prompt.
# The model must route writes through submit_plan_for_approval.
_WRITE_TOOLS_PROMPT = (
    "  - ghl.contacts.create: Create a new contact in GoHighLevel CRM.\n"
    "    args (required): {firstName:string, lastName:string, email:string, phone:string}\n"
    "  - ghl.contacts.update: Update fields on an existing GHL contact.\n"
    "    args (required): {contact_id:string, fields:object}\n"
    "  - ghl.opportunities.update: Move a deal to a different pipeline stage.\n"
    "    args (required): {opportunity_id:string, stage_id:string}\n"
    "  - ghl.sms.send: Send an SMS message to a GHL contact.\n"
    "    args (required): {contact_id:string, message:string}\n"
    "  - ghl.email.send: Send an email via GoHighLevel.\n"
    "    args (required): {contact_id:string, subject:string, body:string}\n"
    "  - ghl.calendars.events.create: Book an appointment on a GHL calendar.\n"
    "    args (required): {calendar_id:string, contact_id:string, start_time:string, title:string}\n"
    "  - slack.chat.send_message: Post a message to a Slack channel or open DM.\n"
    "    args (required): {channel:string, text:string}\n"
    "  - slack.canvas.create: Create a new Slack canvas document in a channel.\n"
    "    args (required): {channel_id:string, title:string, markdown_content:string}\n"
    "  - slack.conversations.invite: Invite a user to a Slack channel.\n"
    "    args (required): {channel:string, users:string}"
)

_SUBMIT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_plan_for_approval",
        "description": (
            "Propose a multi-step plan that includes write actions. The user sees an "
            "approval card and clicks Approve to run all steps, or Reject to cancel. "
            "Use this for ANY task that creates / updates / sends / books / deletes / pays. "
            "Steps run in order."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "One-sentence description of what will be done",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "tool_name": {"type": "string"},
                            "args": {"type": "object"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["id", "tool_name", "args", "rationale"],
                    },
                },
            },
            "required": ["goal", "steps"],
        },
    },
}

_ALL_TOOLS = [*_READ_TOOLS, _SUBMIT_PLAN_SCHEMA]

# Tools that have at least one REQUIRED field in their schema.
# Used by _score() — calling a tool with empty args is only wrong when
# the schema actually requires something. Tools with no required fields
# (e.g. ghl.pipelines.list, slack.users.list) are fine with {}.
_TOOLS_WITH_REQUIRED: frozenset[str] = frozenset(
    sc["function"]["name"] for sc in _READ_TOOLS if sc["function"]["parameters"].get("required")
)

# ---------------------------------------------------------------------------
# System prompt — eval-specific ARLO prompt
# ---------------------------------------------------------------------------

_WORKSPACE_FACTS = (
    "  - agency: Apex Marketing Group\n"
    "  - connected_systems: GoHighLevel CRM, Slack, Google Workspace\n"
    "  - primary_pipeline: Sales Pipeline (stages: Lead → Contacted → Proposal Sent → Negotiation → Won)\n"
    "  - team_members: Bob (owner), Sarah (sales), Mike (ops)\n"
    "  - timezone: America/New_York"
)

_SYSTEM_PROMPT = "\n".join(
    [
        "You are ARLO — a senior operations coworker for a marketing agency that runs on GoHighLevel.",
        "You live in Slack. The team relies on you to run real work in their CRM, calendars, and comms.",
        "",
        "# Voice — friendly but professional",
        "Sound like a senior ops teammate: good at the job, respects their time. Not chatty, not robotic.",
        "Lead with the result. No 'Sure!', 'I'd be happy to help!', or 'Hope this helps!' preambles.",
        "Use Slack markdown: *bold*, _italic_, `code`. No ## headings, no **bold**, no | table | pipes |.",
        "",
        "# Tool discipline",
        "READ tools (call freely): ghl.contacts.search, ghl.opportunities.list, ghl.conversations.list,",
        "  ghl.pipelines.list, ghl.calendars.available_slots, slack.conversations.history,",
        "  slack.search.messages, slack.users.lookup_by_email, slack.users.list, discover_tools.",
        "",
        "WRITE tools (must go through submit_plan_for_approval):",
        _WRITE_TOOLS_PROMPT,
        "",
        "For ANY task that creates / updates / sends / books / deletes: call submit_plan_for_approval.",
        "Never call a write tool directly — it will be rejected. Each plan step needs full args.",
        "",
        "For smalltalk or simple factual questions (date, thanks, etc.), reply directly without tools.",
        "",
        "# Workspace context",
        _WORKSPACE_FACTS,
        "",
        "Conversation artifact: (none yet)",
        "Learned workflow shortcuts: (none yet)",
        "",
        "Be the coworker they brag about.",
    ]
)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str  # ghl-read | ghl-write | slack-read | slack-write | multi-step | direct
    user_request: str
    expected_routing: str  # "plan" | "tool" | "direct" | "read_or_plan"
    # "read_or_plan" = model must search/read first (ID unknown), then plan the write.
    # Both tool-call routing AND plan routing count as correct for these cases,
    # because in a single-turn eval the model can only do one of the two steps.
    expected_tool_hint: str  # substring the correct tool name must contain; "" = any tool OK
    notes: str = ""  # human-readable rationale


_CASES: list[EvalCase] = [
    # --- GHL read ---
    EvalCase(
        case_id="ghl-contact-search",
        category="ghl-read",
        user_request="Find all contacts named Johnson in our CRM.",
        expected_routing="tool",
        expected_tool_hint="contacts.search",
        notes="Simple read — should call ghl.contacts.search directly.",
    ),
    EvalCase(
        case_id="ghl-opportunity-list",
        category="ghl-read",
        user_request="Show me all deals stuck in the Negotiation stage.",
        expected_routing="tool",
        expected_tool_hint="opportunities.list",
        notes="Read-only pipeline query.",
    ),
    EvalCase(
        case_id="ghl-conversation-list",
        category="ghl-read",
        user_request="Pull all unread conversations from the last 24 hours.",
        expected_routing="tool",
        expected_tool_hint="conversations.list",
        notes="Unread CRM conversations — read tool.",
    ),
    # --- GHL write → plan ---
    EvalCase(
        case_id="ghl-contact-create",
        category="ghl-write",
        user_request="Create a new contact: Sarah Miller, email sarah@acmecorp.com, phone 555-382-1234.",
        expected_routing="plan",
        expected_tool_hint="contacts.create",
        notes="Write — must go through plan with full args.",
    ),
    EvalCase(
        case_id="ghl-opportunity-stage-move",
        category="ghl-write",
        user_request="Move Alex Danner's deal to Proposal Sent stage.",
        expected_routing="read_or_plan",
        expected_tool_hint="contacts.search",
        notes=(
            "ID-lookup required: the request gives a name not an opportunity_id. "
            "Correct single-turn behavior is to search for the contact/opportunity first. "
            "Multi-turn eval would then score whether it plans the update on turn 2."
        ),
    ),
    EvalCase(
        case_id="ghl-appointment-book",
        category="ghl-write",
        user_request="Book an appointment for the Hernandez account on Thursday at 2pm ET.",
        expected_routing="read_or_plan",
        expected_tool_hint="contacts.search",
        notes=(
            "ID-lookup required: needs contact_id and calendar_id before booking. "
            "Correct first step is to search for the Hernandez contact."
        ),
    ),
    EvalCase(
        case_id="ghl-sms-send",
        category="ghl-write",
        user_request="Send a quick SMS to the Hernandez contact: 'Following up on your proposal — let me know if you have questions.'",
        expected_routing="read_or_plan",
        expected_tool_hint="contacts.search",
        notes=(
            "ID-lookup required: needs contact_id before ghl.sms.send. "
            "Correct first step is to search for 'Hernandez'."
        ),
    ),
    # --- Slack read ---
    EvalCase(
        case_id="slack-channel-history",
        category="slack-read",
        user_request="What did Bob say in #general today?",
        expected_routing="tool",
        expected_tool_hint="conversations.history",
        notes="Explicit Slack reference — should call slack.conversations.history.",
    ),
    EvalCase(
        case_id="slack-message-search",
        category="slack-read",
        user_request="Search Slack for any messages about the Henderson project.",
        expected_routing="tool",
        expected_tool_hint="search.messages",
        notes="Slack full-text search.",
    ),
    EvalCase(
        case_id="slack-user-lookup",
        category="slack-read",
        user_request="Find Sarah's Slack user ID — her email is sarah@apexmkt.com.",
        expected_routing="tool",
        expected_tool_hint="users.lookup_by_email",
        notes="Prefer lookup_by_email over users.list when email is given.",
    ),
    # --- Slack write → plan ---
    EvalCase(
        case_id="slack-send-message",
        category="slack-write",
        user_request="Post a message to #sales-team: 'Quick update — Q2 pipeline is at $180k, 6 deals in proposal stage.'",
        expected_routing="plan",
        expected_tool_hint="chat.send_message",
        notes="Sending to a different channel is a write.",
    ),
    EvalCase(
        case_id="slack-canvas-create",
        category="slack-write",
        user_request="Create a Slack canvas in #ops summarizing our Q2 pipeline status.",
        expected_routing="read_or_plan",
        expected_tool_hint="pipelines.list",
        notes=(
            "Data-gather required: canvas content depends on real pipeline data. "
            "Correct first step is to read pipelines/opportunities, then plan the canvas. "
            "A model that immediately plans without reading has no content to fill the canvas."
        ),
    ),
    # --- Multi-step planning ---
    EvalCase(
        case_id="multi-step-find-and-sms",
        category="multi-step",
        user_request="Find all contacts tagged 'stale-lead' and send each of them an SMS: 'Hey, just checking in — still interested in growing your agency?'",
        expected_routing="read_or_plan",
        expected_tool_hint="contacts.search",
        notes=(
            "Read-then-batch-write: must search for 'stale-lead' contacts first to get IDs, "
            "then plan the SMS sends. Correct first step is contacts.search with a tag filter."
        ),
    ),
    # --- Direct reply (no tools) ---
    EvalCase(
        case_id="direct-thanks",
        category="direct",
        user_request="Thanks ARLO, you're the best!",
        expected_routing="direct",
        expected_tool_hint="",
        notes="Smalltalk — should reply directly, no tool calls.",
    ),
    EvalCase(
        case_id="direct-date-question",
        category="direct",
        user_request="Hey ARLO, what day of the week is it today?",
        expected_routing="direct",
        expected_tool_hint="",
        notes="Simple factual question — no tools, direct reply.",
    ),
]

# ---------------------------------------------------------------------------
# Model registry (same as inference test)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelSpec:
    label: str
    provider: str
    model_id: str
    api_base: str
    env_key: str


_ALL_MODELS: list[_ModelSpec] = [
    _ModelSpec(
        label="DeepSeek V4-Pro",
        provider="deepseek",
        model_id="deepseek/deepseek-v4-pro",
        api_base="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
    ),
    _ModelSpec(
        label="DeepSeek V4-Flash",
        provider="deepseek",
        model_id="deepseek/deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
    ),
    _ModelSpec(
        label="MiniMax M2.7",
        provider="minimax",
        model_id="openai/MiniMax-M2.7",
        api_base="https://api.minimax.io/v1",
        env_key="MINIMAX_API_KEY",
    ),
    _ModelSpec(
        label="MiniMax M2.5",
        provider="minimax",
        model_id="openai/MiniMax-M2.5",
        api_base="https://api.minimax.io/v1",
        env_key="MINIMAX_API_KEY",
    ),
    _ModelSpec(
        label="Kimi K2.6",
        provider="moonshot",
        model_id="openai/kimi-k2.6",
        api_base="https://api.moonshot.ai/v1",
        env_key="KIMI_API_KEY",
    ),
    _ModelSpec(
        label="Kimi K2.5",
        provider="moonshot",
        model_id="openai/kimi-k2.5",
        api_base="https://api.moonshot.ai/v1",
        env_key="KIMI_API_KEY",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FORBIDDEN_PHRASES = [
    r"i['']d be (more than )?happy to",
    r"\bsure[!,]",
    r"hope this helps",
    r"task completed successfully",
    r"i have completed your request",
    r"as an ai",
    r"## \w",  # Markdown H2 headings (wrong in Slack)
    r"\*\*\w",  # **bold** (should be *bold* in Slack)
]
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in _FORBIDDEN_PHRASES]


def _api_key_for(provider: str) -> str | None:
    s = get_settings()
    mapping = {
        "deepseek": s.deepseek_api_key,
        "minimax": s.minimax_api_key,
        "moonshot": s.kimi_api_key,
        "openai": s.openai_api_key,
    }
    raw = mapping.get(provider, "")
    return raw or None


def _build_resolved(spec: _ModelSpec, api_key: str) -> ResolvedModel:
    return ResolvedModel(
        tier=f"arlo_eval_{spec.provider}",
        provider_key=spec.provider,
        model_id=spec.model_id,
        api_key=api_key,
        api_base=spec.api_base,
        source="env",
    )


def _slug(label: str) -> str:
    """Convert model label to filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score(
    case: EvalCase,
    tool_calls: list[dict[str, Any]],
    text: str,
) -> dict[str, bool]:
    """Return a dict of boolean scores for one (model, case) pair.

    Scoring rules:
      routing_ok  — matched the expected strategy (plan/tool/direct/read_or_plan).
      tool_hit    — the expected tool hint is found in called tool names OR in
                    plan step tool_names (fixes: plan cases only call
                    submit_plan_for_approval directly, the real tool is inside the plan).
      args_ok     — no required-field tool was called with empty args, AND plan
                    steps all have non-empty args (fixes: tools with no required
                    fields are fine with {}, e.g. ghl.pipelines.list).
      voice_ok    — no forbidden phrases in the text response.
    """
    called_names = [(tc.get("function") or {}).get("name", "") for tc in tool_calls]
    has_plan = "submit_plan_for_approval" in called_names
    has_tool = bool(tool_calls) and not has_plan
    has_direct = not tool_calls

    # --- routing_ok ---
    if case.expected_routing == "plan":
        routing_ok = has_plan
    elif case.expected_routing == "tool":
        routing_ok = has_tool
    elif case.expected_routing == "read_or_plan":
        # Model must search/read first (ID unknown) — both "read tool" and "plan"
        # are valid first steps in single-turn context.
        routing_ok = has_tool or has_plan
    else:  # "direct"
        routing_ok = has_direct

    # --- tool_hit ---
    # Extract tool names from inside plan steps too (fix: submit_plan_for_approval
    # is the only directly called function; the real write tool lives in the plan).
    plan_step_names: list[str] = []
    for tc in tool_calls:
        if (tc.get("function") or {}).get("name") == "submit_plan_for_approval":
            try:
                plan_args = json.loads((tc.get("function") or {}).get("arguments", "{}") or "{}")
                for step in plan_args.get("steps", []):
                    plan_step_names.append(step.get("tool_name", ""))
            except json.JSONDecodeError:
                pass

    if case.expected_tool_hint:
        all_names = called_names + plan_step_names
        tool_hit = any(case.expected_tool_hint in name for name in all_names)
    else:
        # direct-reply cases: no tool expected → hit = no tool called
        tool_hit = has_direct

    # --- args_ok ---
    # Only penalize empty args when the tool's schema actually requires fields.
    # Tools with no required fields (e.g. ghl.pipelines.list) are correct with {}.
    # For submit_plan_for_approval: every plan step must have non-empty args.
    if tool_calls:
        args_ok = True
        for tc in tool_calls:
            name = (tc.get("function") or {}).get("name", "")
            raw = ((tc.get("function") or {}).get("arguments", "") or "").strip()
            is_empty = raw in ("", "{}", "{ }")

            if name == "submit_plan_for_approval":
                try:
                    plan_data = json.loads(raw or "{}")
                    for step in plan_data.get("steps", []):
                        if not step.get("args"):
                            args_ok = False
                            break
                except json.JSONDecodeError:
                    args_ok = False
            elif name in _TOOLS_WITH_REQUIRED and is_empty:
                # Called a tool that needs required fields but provided nothing
                args_ok = False
            # else: no required fields → empty args is valid
    else:
        args_ok = True  # direct reply — nothing to check

    # --- voice_ok ---
    voice_ok = not any(pat.search(text) for pat in _FORBIDDEN_RE)

    return {
        "routing_ok": routing_ok,
        "tool_hit": tool_hit,
        "args_ok": args_ok,
        "voice_ok": voice_ok,
    }


# ---------------------------------------------------------------------------
# Single (model x case) eval
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    model_label: str
    model_id: str
    case_id: str
    category: str
    user_request: str
    expected_routing: str
    expected_tool_hint: str
    # scores
    routing_ok: bool = False
    tool_hit: bool = False
    args_ok: bool = False
    voice_ok: bool = False
    # perf
    latency_s: float = 0.0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # raw
    tools_called: list[str] = field(default_factory=list)
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    response_text: str = ""
    # outcome
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""


async def _eval_one(model: _ModelSpec, case: EvalCase, api_key: str) -> EvalResult:
    result = EvalResult(
        model_label=model.label,
        model_id=model.model_id,
        case_id=case.case_id,
        category=case.category,
        user_request=case.user_request,
        expected_routing=case.expected_routing,
        expected_tool_hint=case.expected_tool_hint,
    )

    resolved = _build_resolved(model, api_key)
    # Kimi requires temperature=1.0; _call_resolved handles this internally
    # via the moonshot provider check in _build_kwargs.
    temperature = 1.0 if model.provider == "moonshot" else 0.3

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": case.user_request},
    ]

    try:
        t0 = time.monotonic()
        response = await _call_resolved(
            resolved,
            messages=messages,
            tools=_ALL_TOOLS,
            max_tokens=1024,
            temperature=temperature,
            timeout_s=120.0,
        )
        result.latency_s = time.monotonic() - t0

        usage = getattr(response, "usage", None)
        result.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        result.cost_usd = estimate_cost(response)

        msg = response.choices[0].message
        result.response_text = (msg.content or "").strip()

        raw_calls = getattr(msg, "tool_calls", None) or []
        tool_calls: list[dict[str, Any]] = []
        for tc in raw_calls:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tool_calls.append(
                {
                    "id": getattr(tc, "id", ""),
                    "function": {
                        "name": fn.name,
                        "arguments": fn.arguments or "{}",
                    },
                }
            )

        result.tools_called = [(tc["function"]["name"]) for tc in tool_calls]

        # Extract plan steps for submit_plan_for_approval calls
        for tc in tool_calls:
            if tc["function"]["name"] == "submit_plan_for_approval":
                try:
                    plan_args = json.loads(tc["function"]["arguments"] or "{}")
                    result.plan_steps = plan_args.get("steps", [])
                except json.JSONDecodeError:
                    pass

        scores = _score(case, tool_calls, result.response_text)
        result.routing_ok = scores["routing_ok"]
        result.tool_hit = scores["tool_hit"]
        result.args_ok = scores["args_ok"]
        result.voice_ok = scores["voice_ok"]

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:300]

    return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _write_case_log(result: EvalResult) -> None:
    model_dir = _LOG_ROOT / _slug(result.model_label)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"case-{result.case_id}.json"

    payload = {
        "model": result.model_label,
        "model_id": result.model_id,
        "case_id": result.case_id,
        "category": result.category,
        "user_request": result.user_request,
        "expected_routing": result.expected_routing,
        "expected_tool_hint": result.expected_tool_hint,
        "scores": {
            "routing_ok": result.routing_ok,
            "tool_hit": result.tool_hit,
            "args_ok": result.args_ok,
            "voice_ok": result.voice_ok,
        },
        "perf": {
            "latency_s": round(result.latency_s, 3),
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
        },
        "output": {
            "tools_called": result.tools_called,
            "plan_steps": result.plan_steps,
            "response_text": result.response_text,
        },
        "status": ("skipped" if result.skipped else "error" if result.error else "ok"),
        "skip_reason": result.skip_reason,
        "error": result.error,
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_summary(all_results: list[EvalResult]) -> None:
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)

    per_model: dict[str, dict[str, Any]] = {}
    for r in all_results:
        if r.model_label not in per_model:
            per_model[r.model_label] = {
                "model_id": r.model_id,
                "cases_run": 0,
                "cases_skipped": 0,
                "cases_error": 0,
                "routing_ok": 0,
                "tool_hit": 0,
                "args_ok": 0,
                "voice_ok": 0,
                "total_latency_s": 0.0,
                "total_cost_usd": 0.0,
                "total_output_tokens": 0,
            }
        m = per_model[r.model_label]
        if r.skipped:
            m["cases_skipped"] += 1
            continue
        if r.error:
            m["cases_error"] += 1
            continue
        m["cases_run"] += 1
        m["routing_ok"] += int(r.routing_ok)
        m["tool_hit"] += int(r.tool_hit)
        m["args_ok"] += int(r.args_ok)
        m["voice_ok"] += int(r.voice_ok)
        m["total_latency_s"] += r.latency_s
        m["total_cost_usd"] += r.cost_usd
        m["total_output_tokens"] += r.output_tokens

    # Compute rates
    summary_models = {}
    for label, m in per_model.items():
        n = m["cases_run"]
        summary_models[label] = {
            "model_id": m["model_id"],
            "cases_run": n,
            "cases_skipped": m["cases_skipped"],
            "cases_error": m["cases_error"],
            "routing_accuracy": round(m["routing_ok"] / n, 3) if n else None,
            "tool_hit_rate": round(m["tool_hit"] / n, 3) if n else None,
            "args_ok_rate": round(m["args_ok"] / n, 3) if n else None,
            "voice_ok_rate": round(m["voice_ok"] / n, 3) if n else None,
            "avg_latency_s": round(m["total_latency_s"] / n, 2) if n else None,
            "avg_cost_usd": round(m["total_cost_usd"] / n, 6) if n else None,
            "total_cost_usd": round(m["total_cost_usd"], 6),
            "total_output_tokens": m["total_output_tokens"],
        }

    summary = {
        "run_timestamp": _RUN_TS,
        "cases_total": len(_CASES),
        "models_total": len(_ALL_MODELS),
        "models": summary_models,
    }
    (_LOG_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Console printing
# ---------------------------------------------------------------------------

_SEP = "=" * 100
_COL = "{:<22} {:<26} {:<12} {:<10} {:<9} {:<9} {:>8} {:>8}"


def _print_header() -> None:
    print(f"\n{_SEP}")
    print("  LYRALABS — ARLO TASK EVAL")
    print(f"  Log directory: {_LOG_ROOT}")
    print(_SEP)
    print(
        _COL.format(
            "Model", "Case", "Category", "Routing", "Tool Hit", "Args OK", "Latency", "Cost $"
        )
    )
    print("-" * 100)


def _print_row(r: EvalResult) -> None:
    if r.skipped:
        status = f"[SKIP] {r.skip_reason}"
        print(f"  {r.model_label:<20} {r.case_id:<26} {status}")
        return
    if r.error:
        print(f"  {r.model_label:<20} {r.case_id:<26} [ERROR] {r.error[:50]}")
        return
    print(
        _COL.format(
            r.model_label,
            r.case_id,
            r.category,
            "✓" if r.routing_ok else "✗",
            "✓" if r.tool_hit else "✗",
            "✓" if r.args_ok else "✗",
            f"{r.latency_s:.1f}s",
            f"${r.cost_usd:.5f}",
        )
    )


def _print_summary_table(all_results: list[EvalResult]) -> None:
    plan_cases = [c for c in _CASES if c.expected_routing == "plan"]
    readplan_cases = [c for c in _CASES if c.expected_routing == "read_or_plan"]
    print(f"\n{_SEP}")
    print("  SUMMARY BY MODEL")
    print("  Routing% = correct strategy across all 15 cases")
    print(f"  Plan cases ({len(plan_cases)}): must submit_plan_for_approval (all info given)")
    print(
        f"  Read-or-plan ({len(readplan_cases)}): must read first (ID unknown) OR plan — both are correct"
    )
    print(_SEP)
    hdr = "{:<22} {:>6} {:>10} {:>10} {:>9} {:>9} {:>10} {:>12}"
    print(
        hdr.format(
            "Model",
            "Cases",
            "Routing %",
            "Tool Hit %",
            "Args %",
            "Voice %",
            "Avg Lat",
            "Total Cost",
        )
    )
    print("-" * 100)

    per: dict[str, list[EvalResult]] = {}
    for r in all_results:
        per.setdefault(r.model_label, []).append(r)

    for label, results in per.items():
        active = [r for r in results if not r.skipped and not r.error]
        skipped = sum(1 for r in results if r.skipped)
        errors = sum(1 for r in results if r.error)
        n = len(active)
        if n == 0:
            label_suffix = f" (skipped={skipped}, err={errors})"
            print(hdr.format(label + label_suffix, 0, "—", "—", "—", "—", "—", "—"))
            continue
        routing_pct = sum(r.routing_ok for r in active) / n * 100
        tool_pct = sum(r.tool_hit for r in active) / n * 100
        args_pct = sum(r.args_ok for r in active) / n * 100
        voice_pct = sum(r.voice_ok for r in active) / n * 100
        avg_lat = sum(r.latency_s for r in active) / n
        total_cost = sum(r.cost_usd for r in active)
        suffix = f" (skip={skipped})" if skipped or errors else ""
        print(
            hdr.format(
                label + suffix,
                n,
                f"{routing_pct:.0f}%",
                f"{tool_pct:.0f}%",
                f"{args_pct:.0f}%",
                f"{voice_pct:.0f}%",
                f"{avg_lat:.1f}s",
                f"${total_cost:.4f}",
            )
        )
    print(_SEP)
    print(f"  Full logs → {_LOG_ROOT}\n")


# ---------------------------------------------------------------------------
# Test entry points
# ---------------------------------------------------------------------------


@pytest.mark.live
async def test_arlo_eval_all() -> None:
    """Run all 6 models x all 15 cases. Saves JSON logs; prints summary table.

    Pass as long as at least one model x case pair completes without error.
    Skips models whose API key is missing.
    """
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)
    all_results: list[EvalResult] = []

    _print_header()

    for model in _ALL_MODELS:
        api_key = _api_key_for(model.provider)
        if api_key is None:
            for case in _CASES:
                r = EvalResult(
                    model_label=model.label,
                    model_id=model.model_id,
                    case_id=case.case_id,
                    category=case.category,
                    user_request=case.user_request,
                    expected_routing=case.expected_routing,
                    expected_tool_hint=case.expected_tool_hint,
                    skipped=True,
                    skip_reason=f"no {model.env_key} in env",
                )
                all_results.append(r)
                _print_row(r)
            continue

        for case in _CASES:
            print(
                f"  → {model.label:<20} | {case.case_id:<28} ...",
                end="",
                flush=True,
            )
            r = await _eval_one(model, case, api_key)
            all_results.append(r)
            _write_case_log(r)

            if r.error:
                print(f" ERROR: {r.error[:60]}")
            else:
                scores = f"R={'✓' if r.routing_ok else '✗'} T={'✓' if r.tool_hit else '✗'} A={'✓' if r.args_ok else '✗'}"
                print(f" {scores}  {r.latency_s:.1f}s  ${r.cost_usd:.5f}")

    _print_summary_table(all_results)
    _write_summary(all_results)

    # Assertion: at least one model must have been active
    active = [r for r in all_results if not r.skipped]
    assert active, (
        "All models were skipped — set at least DEEPSEEK_API_KEY in .env to run the eval."
    )

    # Warn (not fail) if errors — so partial runs still save useful data
    errors = [r for r in active if r.error]
    if errors:
        msgs = "\n".join(f"  {r.model_label} / {r.case_id}: {r.error}" for r in errors[:10])
        pytest.fail(f"{len(errors)} case(s) returned errors:\n{msgs}")


# ---------------------------------------------------------------------------
# Per-model parametrized test — run one model across all cases
# ---------------------------------------------------------------------------

_MODEL_IDS = [m.label for m in _ALL_MODELS]


@pytest.mark.live
@pytest.mark.parametrize("model", _ALL_MODELS, ids=_MODEL_IDS)
async def test_arlo_eval_model(model: _ModelSpec) -> None:
    """Run all 15 cases for a single model. Useful for focused model comparison.

    Example:
        pytest tests/eval/test_arlo_eval.py -k "MiniMax M2.7" -v -s -m live
    """
    api_key = _api_key_for(model.provider)
    if api_key is None:
        pytest.skip(f"no {model.env_key} configured")

    print(f"\n{'─' * 70}")
    print(f"  Model: {model.label} ({model.model_id})")
    print(f"{'─' * 70}")

    results: list[EvalResult] = []
    for case in _CASES:
        print(f"  [{case.category}] {case.case_id} ...", end="", flush=True)
        r = await _eval_one(model, case, api_key)
        results.append(r)
        _write_case_log(r)

        if r.error:
            print(f" ERROR: {r.error[:60]}")
        else:
            scores = f"routing={'✓' if r.routing_ok else '✗'}  tool={'✓' if r.tool_hit else '✗'}  args={'✓' if r.args_ok else '✗'}  voice={'✓' if r.voice_ok else '✗'}"
            print(f"  {scores}  {r.latency_s:.1f}s  ${r.cost_usd:.5f}")
            if r.response_text:
                snippet = r.response_text[:120].replace("\n", " ")
                print(f"    reply: {snippet}")
            if r.tools_called:
                print(f"    tools: {r.tools_called}")
            if r.plan_steps:
                for s in r.plan_steps:
                    print(
                        f"    plan step: {s.get('tool_name')}  args={json.dumps(s.get('args', {}))[:80]}"
                    )

    # Summary
    active = [r for r in results if not r.error]
    n = len(active)
    if n:
        routing_pct = sum(r.routing_ok for r in active) / n * 100
        tool_pct = sum(r.tool_hit for r in active) / n * 100
        args_pct = sum(r.args_ok for r in active) / n * 100
        voice_pct = sum(r.voice_ok for r in active) / n * 100
        avg_lat = sum(r.latency_s for r in active) / n
        total_cost = sum(r.cost_usd for r in active)
        print(f"\n  ── {model.label} summary ──")
        print(
            f"  Routing:   {routing_pct:.0f}%  |  Tool Hit: {tool_pct:.0f}%  |  Args OK: {args_pct:.0f}%  |  Voice OK: {voice_pct:.0f}%"
        )
        print(f"  Avg Lat:   {avg_lat:.1f}s  |  Total Cost: ${total_cost:.4f}")

    _write_summary(results)
    errors = [r for r in results if r.error]
    if errors:
        msgs = "\n".join(f"  {r.case_id}: {r.error}" for r in errors)
        pytest.fail(f"Cases with errors:\n{msgs}")


# ---------------------------------------------------------------------------
# Per-case parametrized test — run one specific case across all 6 models
# ---------------------------------------------------------------------------

_CASE_IDS = [c.case_id for c in _CASES]


@pytest.mark.live
@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
async def test_arlo_eval_case(case: EvalCase) -> None:
    """Run one specific case across all 6 models. Good for debugging a case.

    Example:
        pytest tests/eval/test_arlo_eval.py -k "ghl-contact-search" -v -s -m live
    """
    print(f"\n{'─' * 70}")
    print(f"  Case    : {case.case_id}  [{case.category}]")
    print(f"  Request : {case.user_request}")
    print(f"  Expected: routing={case.expected_routing}  tool_hint={case.expected_tool_hint!r}")
    print(f"{'─' * 70}")

    results: list[EvalResult] = []
    for model in _ALL_MODELS:
        api_key = _api_key_for(model.provider)
        if api_key is None:
            print(f"  {model.label:<22} SKIP (no {model.env_key})")
            continue

        print(f"  {model.label:<22} ...", end="", flush=True)
        r = await _eval_one(model, case, api_key)
        results.append(r)
        _write_case_log(r)

        if r.error:
            print(f" ERROR: {r.error[:60]}")
        else:
            scores = f"R={'✓' if r.routing_ok else '✗'} T={'✓' if r.tool_hit else '✗'} A={'✓' if r.args_ok else '✗'}"
            print(f" {scores}  {r.latency_s:.1f}s  ${r.cost_usd:.5f}  tools={r.tools_called}")

    if not results:
        pytest.skip("No models had API keys configured — nothing to evaluate.")

    _write_summary(results)
    errors = [r for r in results if r.error]
    if errors:
        msgs = "\n".join(f"  {r.model_label}: {r.error}" for r in errors)
        pytest.fail(f"Errors:\n{msgs}")
