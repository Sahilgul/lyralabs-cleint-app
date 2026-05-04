"""ARLO × GHL Live Eval — real MCP tool calls, real GoHighLevel data.

Unlike `test_arlo_eval.py` (which uses hardcoded mock schemas), this eval:

  1. Fetches the ACTUAL tool schemas from the GHL MCP server at runtime.
     The model sees the exact same tool names and arg descriptions it would
     see in a real ARLO session (e.g. `contacts_search-contacts`, not the
     invented `ghl.contacts.search`).

  2. Runs a multi-turn agent loop:
        LLM → tool call → real GHL execution → result back to LLM → reply
     Read tools are actually invoked against your GHL sub-account so the
     model's final response is grounded in real contact/opportunity data.

  3. Write tools are NOT executed — when the model calls
     `submit_plan_for_approval`, we capture the plan and return a stub
     response ("Plan submitted for approval."). This prevents test runs
     from mutating live CRM data.

Prerequisites (add to .env):
    GHL_EVAL_TOKEN=pit-xxxxxxxx...    # Private Integration Token
    GHL_EVAL_LOCATION_ID=abc123...    # Sub-account location ID

  → GHL sub-account → Settings → Business Info → Private Integrations.
  → Give read scopes: contacts.readonly, opportunities.readonly,
    conversations.readonly, calendars.readonly.

Results are written to:
    logs/arlo-ghl-live-<timestamp>/summary.json
    logs/arlo-ghl-live-<timestamp>/<model-slug>/case-<id>.json

Run:
    pytest tests/eval/test_arlo_ghl_live.py -v -s -m live

Run one model:
    pytest tests/eval/test_arlo_ghl_live.py -k "MiniMax M2.7" -v -s -m live
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from lyra_core.common.config import get_settings
from lyra_core.common.llm import _call_resolved, estimate_cost
from lyra_core.llm.router import ResolvedModel

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------

_RUN_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
_LOG_ROOT = Path(__file__).parent.parent.parent / "logs" / f"arlo-ghl-live-{_RUN_TS}"

# GHL MCP endpoint
_GHL_MCP_URL = "https://services.leadconnectorhq.com/mcp/"

# Max turns per eval case (safety cap on the agent loop)
_MAX_TURNS = 6

# ---------------------------------------------------------------------------
# submit_plan_for_approval schema — always injected alongside GHL tools
# ---------------------------------------------------------------------------

_SUBMIT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_plan_for_approval",
        "description": (
            "Propose a multi-step plan that includes write actions. The user sees an "
            "approval card and clicks Approve to run all steps, or Reject to cancel. "
            "Use this for ANY task that creates / updates / sends / books / deletes. "
            "Steps run in order; each step needs full args."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
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

# ---------------------------------------------------------------------------
# System prompt — matches production ARLO agent prompt closely
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = "\n".join([
    "You are ARLO — a senior operations coworker for a marketing agency that runs on GoHighLevel.",
    "You live in Slack. The team relies on you to run real work in their CRM, calendars, and comms.",
    "",
    "# Voice — friendly but professional",
    "Sound like a senior ops teammate: good at the job, respects their time. Not chatty, not robotic.",
    "Lead with the result. No 'Sure!', 'I'd be happy to help!', or 'Hope this helps!' preambles.",
    "Use Slack markdown: *bold*, _italic_, `code`. No ## headings, no **bold**, no | table | pipes |.",
    "Be specific: real names, counts, dollar amounts, IDs. Never 'several' when you can say '12 contacts'.",
    "",
    "# Tool discipline",
    "You have READ tools (call freely) and WRITE tools (require approval).",
    "For ANY task that creates / updates / sends / books / deletes: call submit_plan_for_approval.",
    "For smalltalk or simple factual questions (date, thanks, etc.): reply directly, no tools.",
    "",
    "# Workflow",
    "1. If unsure of the right tool → call discover_tools(intent='...').",
    "2. Read what you need. If empty results, try a broader filter before reporting 'none found'.",
    "3. Writes → bundle all write steps into one submit_plan_for_approval call.",
    "4. Reply. Lead with the answer. One proactive next step if useful.",
    "",
    "Workspace: Apex Marketing Agency. Pipeline stages: Lead → Contacted → Proposal Sent → Negotiation → Won.",
    "Timezone: America/New_York.",
    "",
    "Be the coworker they brag about.",
])

# ---------------------------------------------------------------------------
# Test cases — realistic ARLO requests exercising real GHL MCP tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GhlEvalCase:
    case_id: str
    category: str           # ghl-read | ghl-write | multi-turn
    user_request: str
    expected_routing: str   # "read_tool" | "plan" | "direct"
    notes: str = ""


_CASES: list[GhlEvalCase] = [
    # --- Single-turn reads (model calls one GHL tool, returns answer) ---
    GhlEvalCase(
        case_id="contact-search-by-name",
        category="ghl-read",
        user_request="Find all contacts with the last name Miller in our CRM.",
        expected_routing="read_tool",
        notes="Should call contacts search; real contacts returned.",
    ),
    GhlEvalCase(
        case_id="open-opportunities",
        category="ghl-read",
        user_request="Show me all open deals in our Sales Pipeline.",
        expected_routing="read_tool",
        notes="Opportunity listing — real pipeline data.",
    ),
    GhlEvalCase(
        case_id="recent-conversations",
        category="ghl-read",
        user_request="Pull the 5 most recent conversations from our CRM.",
        expected_routing="read_tool",
        notes="Conversations list — real data.",
    ),
    GhlEvalCase(
        case_id="pipeline-list",
        category="ghl-read",
        user_request="What pipelines do we have set up in GoHighLevel?",
        expected_routing="read_tool",
        notes="Pipeline metadata — read tool, no mutations.",
    ),
    # --- Multi-turn reads (model may need > 1 tool call to answer) ---
    GhlEvalCase(
        case_id="stuck-deals-detail",
        category="multi-turn",
        user_request=(
            "Which deals have been stuck in the same stage for more than 2 weeks? "
            "Give me names, stage, and how long they've been there."
        ),
        expected_routing="read_tool",
        notes="Likely needs pipeline list then opportunity list. Multi-turn.",
    ),
    GhlEvalCase(
        case_id="contact-with-recent-convo",
        category="multi-turn",
        user_request=(
            "Search for the contact named Johnson and show me their most recent conversation."
        ),
        expected_routing="read_tool",
        notes="Contact search + conversation lookup. Multi-turn.",
    ),
    # --- Write tasks → must produce a plan, no execution ---
    GhlEvalCase(
        case_id="create-contact-plan",
        category="ghl-write",
        user_request=(
            "Create a new contact: Alex Testington, email alex.testington@evaltest.com, "
            "phone 555-999-0001."
        ),
        expected_routing="plan",
        notes="Write — must submit plan with contacts_create. Args must include name + email + phone.",
    ),
    GhlEvalCase(
        case_id="update-opportunity-stage-plan",
        category="ghl-write",
        user_request=(
            "Move the Johnson deal to 'Proposal Sent' stage. "
            "Look up the contact first to get the right opportunity."
        ),
        expected_routing="plan",
        notes="Read then write. Model should search first, then plan the stage update.",
    ),
    GhlEvalCase(
        case_id="book-appointment-plan",
        category="ghl-write",
        user_request=(
            "Book a 30-minute discovery call for the Miller contact next Thursday at 2pm ET."
        ),
        expected_routing="plan",
        notes="Booking is a write — should produce a plan with calendars_book-appointment.",
    ),
    GhlEvalCase(
        case_id="send-sms-plan",
        category="ghl-write",
        user_request=(
            "Send this SMS to the Johnson contact: "
            "'Hi! Just following up on your proposal — any questions?'"
        ),
        expected_routing="plan",
        notes="Message send is HIGH risk — should definitely be a plan, not a direct call.",
    ),
    # --- Direct reply (no GHL tools needed) ---
    GhlEvalCase(
        case_id="direct-thanks",
        category="direct",
        user_request="Thanks ARLO, great work today!",
        expected_routing="direct",
        notes="Smalltalk — no tools, ARLO-voice direct reply.",
    ),
]

# ---------------------------------------------------------------------------
# Model registry
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
# GHL MCP — schema fetch and tool execution
# ---------------------------------------------------------------------------


async def _fetch_ghl_schemas(
    token: str,
    location_id: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Connect to the real GHL MCP server and return (lc_tools, openai_schemas).

    `lc_tools` — LangChain tool objects, used for actual ainvoke() execution.
    `openai_schemas` — OpenAI function-call format, passed to the LLM.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    mc = MultiServerMCPClient(
        {
            "ghl": {
                "url": _GHL_MCP_URL,
                "transport": "http",
                "headers": {
                    "Authorization": f"Bearer {token}",
                    "locationId": location_id,
                },
            }
        }
    )
    lc_tools = await mc.get_tools()

    # Convert each LangChain tool to OpenAI function-call schema.
    # LangChain's BaseTool carries args_schema (a Pydantic model) from the
    # MCP server's inputSchema. We build the OpenAI schema from that.
    openai_schemas: list[dict[str, Any]] = []
    for t in lc_tools:
        schema = getattr(t, "args_schema", None)
        if schema is not None:
            try:
                json_schema = schema.model_json_schema()
            except Exception:
                json_schema = {"type": "object", "properties": {}}
        else:
            json_schema = {"type": "object", "properties": {}}

        # Remove $defs / $schema meta-fields that confuse some providers.
        json_schema.pop("$defs", None)
        json_schema.pop("$schema", None)
        json_schema.pop("title", None)

        openai_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or t.name)[:500],
                    "parameters": json_schema,
                },
            }
        )

    # Always append submit_plan_for_approval at the end.
    openai_schemas.append(_SUBMIT_PLAN_SCHEMA)
    return lc_tools, openai_schemas


async def _execute_ghl_tool(
    tool_name: str,
    args: dict[str, Any],
    lc_tools: list[Any],
    token: str,
    location_id: str,
) -> str:
    """Invoke a real GHL MCP tool and return the result as a string.

    Creates a fresh MCP connection per call (matches production McpToolAdapter
    behaviour — safe and isolated, just slightly slower).
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    mc = MultiServerMCPClient(
        {
            "ghl": {
                "url": _GHL_MCP_URL,
                "transport": "http",
                "headers": {
                    "Authorization": f"Bearer {token}",
                    "locationId": location_id,
                },
            }
        }
    )
    fresh_tools = await mc.get_tools()
    tool = next((t for t in fresh_tools if t.name == tool_name), None)
    if tool is None:
        return f"[eval] Tool '{tool_name}' not found on GHL MCP server."
    try:
        result = await tool.ainvoke(args)
        if isinstance(result, str):
            return result[:4000]  # cap to avoid blowing context
        return json.dumps(result, default=str)[:4000]
    except Exception as exc:
        return f"[eval] Tool execution error: {type(exc).__name__}: {exc}"[:500]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_FORBIDDEN_PHRASES = [
    r"i['']d be (more than )?happy to",
    r"\bsure[!,]",
    r"hope this helps",
    r"task completed successfully",
    r"i have completed your request",
    r"as an ai",
    r"## \w",
    r"\*\*\w",
]
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in _FORBIDDEN_PHRASES]


def _score_result(
    case: GhlEvalCase,
    all_tool_calls: list[dict[str, Any]],
    final_text: str,
    tool_results_text: list[str],
) -> dict[str, bool | int]:
    called = [tc["name"] for tc in all_tool_calls]
    has_plan = "submit_plan_for_approval" in called
    has_read_tool = bool(called) and not has_plan
    has_direct = not called

    # routing_ok
    if case.expected_routing == "plan":
        routing_ok = has_plan
    elif case.expected_routing == "read_tool":
        routing_ok = has_read_tool or has_plan  # plan is acceptable too
    else:  # direct
        routing_ok = has_direct

    # args_ok — plan steps must have non-empty args
    args_ok = True
    if has_plan:
        plan_tc = next(
            (tc for tc in all_tool_calls if tc["name"] == "submit_plan_for_approval"), None
        )
        if plan_tc:
            try:
                plan_args = json.loads(plan_tc.get("arguments", "{}"))
                steps = plan_args.get("steps", [])
                args_ok = all(bool(s.get("args")) for s in steps) if steps else False
            except json.JSONDecodeError:
                args_ok = False

    # grounded — does the final text reference content from tool results?
    grounded = False
    if tool_results_text and final_text:
        # Check if any word from tool results (>4 chars, not numbers) appears in reply.
        all_tool_content = " ".join(tool_results_text).lower()
        tool_words = {
            w for w in re.findall(r"[a-zA-Z]{5,}", all_tool_content)
            if w not in {"false", "title", "phone", "email", "first", "contact", "fields", "value"}
        }
        reply_lower = final_text.lower()
        grounded = any(w in reply_lower for w in list(tool_words)[:60])
    elif case.expected_routing == "direct":
        grounded = True  # no tools expected — trivially grounded

    # voice_ok
    voice_ok = not any(pat.search(final_text) for pat in _FORBIDDEN_RE)

    return {
        "routing_ok": routing_ok,
        "args_ok": args_ok,
        "grounded": grounded,
        "voice_ok": voice_ok,
        "turns": len(all_tool_calls),
    }


# ---------------------------------------------------------------------------
# Main eval loop — one (model × case)
# ---------------------------------------------------------------------------


@dataclass
class LiveEvalResult:
    model_label: str
    model_id: str
    case_id: str
    category: str
    user_request: str
    expected_routing: str
    # scores
    routing_ok: bool = False
    args_ok: bool = False
    grounded: bool = False
    voice_ok: bool = False
    # perf
    latency_s: float = 0.0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    # trace
    turns: int = 0
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    final_response: str = ""
    # status
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""


def _api_key_for(provider: str) -> str | None:
    s = get_settings()
    mapping = {
        "deepseek": s.deepseek_api_key,
        "minimax": s.minimax_api_key,
        "moonshot": s.kimi_api_key,
        "openai": s.openai_api_key,
    }
    return mapping.get(provider, "") or None


def _build_resolved(spec: _ModelSpec, api_key: str) -> ResolvedModel:
    return ResolvedModel(
        tier=f"ghl_live_eval_{spec.provider}",
        provider_key=spec.provider,
        model_id=spec.model_id,
        api_key=api_key,
        api_base=spec.api_base,
        source="env",
    )


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _serialize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Convert LiteLLM tool_calls objects to plain dicts."""
    result = []
    for tc in raw_calls or []:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        result.append(
            {
                "id": getattr(tc, "id", ""),
                "name": fn.name,
                "arguments": fn.arguments or "{}",
            }
        )
    return result


async def _eval_one(
    model: _ModelSpec,
    case: GhlEvalCase,
    api_key: str,
    lc_tools: list[Any],
    ghl_schemas: list[dict[str, Any]],
    ghl_token: str,
    ghl_location_id: str,
) -> LiveEvalResult:
    result = LiveEvalResult(
        model_label=model.label,
        model_id=model.model_id,
        case_id=case.case_id,
        category=case.category,
        user_request=case.user_request,
        expected_routing=case.expected_routing,
    )

    resolved = _build_resolved(model, api_key)
    temperature = 1.0 if model.provider == "moonshot" else 0.3

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": case.user_request},
    ]

    t_start = time.monotonic()
    all_tool_calls: list[dict[str, Any]] = []
    tool_results_for_scoring: list[str] = []

    try:
        for _turn in range(_MAX_TURNS):
            response = await _call_resolved(
                resolved,
                messages=messages,
                tools=ghl_schemas,
                max_tokens=1500,
                temperature=temperature,
                timeout_s=120.0,
            )

            result.total_output_tokens += (
                getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
            )
            result.total_cost_usd += estimate_cost(response)

            msg = response.choices[0].message
            text = (msg.content or "").strip()
            raw_calls = _serialize_tool_calls(getattr(msg, "tool_calls", None))

            # Build the assistant message dict for history (matches OpenAI format)
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            if raw_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in raw_calls
                ]
            messages.append(assistant_msg)

            # No tool calls → model gave a direct reply — done
            if not raw_calls:
                result.final_response = text
                break

            # Execute each tool call
            tool_response_msgs: list[dict[str, Any]] = []
            for tc in raw_calls:
                all_tool_calls.append(tc)
                tool_name = tc["name"]

                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                if tool_name == "submit_plan_for_approval":
                    # Capture plan, don't execute — this is the approval gate simulation
                    try:
                        plan_data = json.loads(tc["arguments"] or "{}")
                        result.plan_steps = plan_data.get("steps", [])
                    except json.JSONDecodeError:
                        pass
                    tool_result_str = json.dumps(
                        {"status": "submitted", "message": "Plan submitted for approval."}
                    )
                    # After submitting a plan, the loop is done — model replies with summary
                else:
                    # Real GHL MCP execution
                    print(f"      [GHL] {tool_name}({json.dumps(args, default=str)[:80]}) ...", end="", flush=True)
                    tool_result_str = await _execute_ghl_tool(
                        tool_name, args, lc_tools, ghl_token, ghl_location_id
                    )
                    print(f" → {len(tool_result_str)} chars")
                    tool_results_for_scoring.append(tool_result_str)

                tool_response_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result_str,
                    }
                )

            messages.extend(tool_response_msgs)

            # If a plan was submitted, do one more LLM turn to get the summary reply
            plan_submitted = any(tc["name"] == "submit_plan_for_approval" for tc in raw_calls)
            if plan_submitted:
                # One final turn to get the natural-language plan summary
                response2 = await _call_resolved(
                    resolved,
                    messages=messages,
                    tools=ghl_schemas,
                    max_tokens=800,
                    temperature=temperature,
                    timeout_s=60.0,
                )
                result.total_cost_usd += estimate_cost(response2)
                result.final_response = (response2.choices[0].message.content or "").strip()
                break

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:400]

    result.latency_s = time.monotonic() - t_start
    result.turns = len(all_tool_calls)
    result.tool_calls_made = all_tool_calls
    result.tool_results = [r[:500] for r in tool_results_for_scoring]  # truncate for logs

    if not result.error:
        scores = _score_result(case, all_tool_calls, result.final_response, tool_results_for_scoring)
        result.routing_ok = bool(scores["routing_ok"])
        result.args_ok = bool(scores["args_ok"])
        result.grounded = bool(scores["grounded"])
        result.voice_ok = bool(scores["voice_ok"])

    return result


# ---------------------------------------------------------------------------
# Log writing
# ---------------------------------------------------------------------------


def _write_case_log(result: LiveEvalResult) -> None:
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
        "scores": {
            "routing_ok": result.routing_ok,
            "args_ok": result.args_ok,
            "grounded": result.grounded,
            "voice_ok": result.voice_ok,
        },
        "perf": {
            "latency_s": round(result.latency_s, 2),
            "turns": result.turns,
            "total_output_tokens": result.total_output_tokens,
            "total_cost_usd": result.total_cost_usd,
        },
        "trace": {
            "tool_calls_made": [
                {"name": tc["name"], "args": tc["arguments"][:300]}
                for tc in result.tool_calls_made
            ],
            "plan_steps": result.plan_steps,
            "ghl_results_snippets": result.tool_results,
            "final_response": result.final_response,
        },
        "status": (
            "skipped" if result.skipped
            else "error" if result.error
            else "ok"
        ),
        "error": result.error,
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_summary(all_results: list[LiveEvalResult], ghl_tool_names: list[str]) -> None:
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)

    per_model: dict[str, dict[str, Any]] = {}
    for r in all_results:
        if r.model_label not in per_model:
            per_model[r.model_label] = {
                "model_id": r.model_id,
                "n": 0, "skipped": 0, "errors": 0,
                "routing_ok": 0, "args_ok": 0, "grounded": 0, "voice_ok": 0,
                "lat": 0.0, "cost": 0.0, "tokens": 0,
            }
        m = per_model[r.model_label]
        if r.skipped:
            m["skipped"] += 1
            continue
        if r.error:
            m["errors"] += 1
            continue
        m["n"] += 1
        m["routing_ok"] += int(r.routing_ok)
        m["args_ok"] += int(r.args_ok)
        m["grounded"] += int(r.grounded)
        m["voice_ok"] += int(r.voice_ok)
        m["lat"] += r.latency_s
        m["cost"] += r.total_cost_usd
        m["tokens"] += r.total_output_tokens

    model_summaries = {}
    for label, m in per_model.items():
        n = m["n"]
        model_summaries[label] = {
            "model_id": m["model_id"],
            "cases_run": n,
            "cases_skipped": m["skipped"],
            "cases_error": m["errors"],
            "routing_accuracy": round(m["routing_ok"] / n, 3) if n else None,
            "args_ok_rate": round(m["args_ok"] / n, 3) if n else None,
            "grounding_rate": round(m["grounded"] / n, 3) if n else None,
            "voice_ok_rate": round(m["voice_ok"] / n, 3) if n else None,
            "avg_latency_s": round(m["lat"] / n, 2) if n else None,
            "total_cost_usd": round(m["cost"], 6),
            "total_output_tokens": m["tokens"],
        }

    summary = {
        "run_timestamp": _RUN_TS,
        "eval_type": "ghl_live_mcp",
        "ghl_tools_discovered": len(ghl_tool_names),
        "ghl_tool_names": ghl_tool_names,
        "cases_total": len(_CASES),
        "models_total": len(_ALL_MODELS),
        "models": model_summaries,
    }
    (_LOG_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.live
async def test_arlo_ghl_live_all() -> None:
    """Run all models × all cases with real GHL MCP tool calls.

    Requires GHL_EVAL_TOKEN and GHL_EVAL_LOCATION_ID in .env.
    Skips individual models whose LLM API key is missing.
    Saves per-case JSON logs and a summary.json.
    """
    s = get_settings()
    if not s.ghl_eval_token:
        pytest.skip("GHL_EVAL_TOKEN not set in .env — skipping live GHL eval.")
    if not s.ghl_eval_location_id:
        pytest.skip("GHL_EVAL_LOCATION_ID not set in .env — skipping live GHL eval.")

    _LOG_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Fetch real GHL schemas once (shared across all models) ---
    print(f"\n{'=' * 80}")
    print("  ARLO × GHL LIVE EVAL — fetching real MCP schemas ...")
    print(f"  GHL MCP: {_GHL_MCP_URL}")
    lc_tools, ghl_schemas = await _fetch_ghl_schemas(s.ghl_eval_token, s.ghl_eval_location_id)
    # Exclude submit_plan_for_approval from the tool names list (it's ours, not GHL's)
    ghl_tool_names = [
        sc["function"]["name"]
        for sc in ghl_schemas
        if sc["function"]["name"] != "submit_plan_for_approval"
    ]
    print(f"  Discovered {len(ghl_tool_names)} GHL tools: {', '.join(ghl_tool_names[:10])}{'...' if len(ghl_tool_names) > 10 else ''}")
    print(f"  Log dir: {_LOG_ROOT}")
    print(f"{'=' * 80}\n")

    all_results: list[LiveEvalResult] = []

    for model in _ALL_MODELS:
        api_key = _api_key_for(model.provider)
        if api_key is None:
            print(f"  {model.label:<22} SKIP (no {model.env_key})")
            for case in _CASES:
                r = LiveEvalResult(
                    model_label=model.label,
                    model_id=model.model_id,
                    case_id=case.case_id,
                    category=case.category,
                    user_request=case.user_request,
                    expected_routing=case.expected_routing,
                    skipped=True,
                    skip_reason=f"no {model.env_key}",
                )
                all_results.append(r)
            continue

        print(f"\n  ── {model.label} ({model.model_id}) ──")
        for case in _CASES:
            print(f"    [{case.category}] {case.case_id}")
            print(f"      Q: {case.user_request[:80]}{'...' if len(case.user_request) > 80 else ''}")

            r = await _eval_one(
                model, case, api_key,
                lc_tools, ghl_schemas,
                s.ghl_eval_token, s.ghl_eval_location_id,
            )
            all_results.append(r)
            _write_case_log(r)

            if r.error:
                print(f"      ERROR: {r.error[:80]}")
            else:
                scores = (
                    f"routing={'✓' if r.routing_ok else '✗'} "
                    f"grounded={'✓' if r.grounded else '✗'} "
                    f"voice={'✓' if r.voice_ok else '✗'}"
                )
                print(f"      {scores}  {r.latency_s:.1f}s  ${r.total_cost_usd:.5f}  turns={r.turns}")
                if r.final_response:
                    snip = r.final_response[:120].replace("\n", " ")
                    print(f"      reply: {snip}")

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print("  SUMMARY BY MODEL")
    print(f"{'=' * 80}")
    hdr = "{:<22} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12}"
    print(hdr.format("Model", "Cases", "Routing %", "Grounded%", "Voice %", "Avg Lat", "Total Cost"))
    print("-" * 80)

    per: dict[str, list[LiveEvalResult]] = {}
    for r in all_results:
        per.setdefault(r.model_label, []).append(r)

    for label, results in per.items():
        active = [r for r in results if not r.skipped and not r.error]
        n = len(active)
        if n == 0:
            skipped = sum(1 for r in results if r.skipped)
            print(hdr.format(label + f" (skip={skipped})", 0, "—", "—", "—", "—", "—"))
            continue
        routing_pct = sum(r.routing_ok for r in active) / n * 100
        grounded_pct = sum(r.grounded for r in active) / n * 100
        voice_pct = sum(r.voice_ok for r in active) / n * 100
        avg_lat = sum(r.latency_s for r in active) / n
        total_cost = sum(r.total_cost_usd for r in active)
        print(hdr.format(
            label, n,
            f"{routing_pct:.0f}%", f"{grounded_pct:.0f}%", f"{voice_pct:.0f}%",
            f"{avg_lat:.1f}s", f"${total_cost:.4f}",
        ))
    print(f"{'=' * 80}")
    print(f"  Full logs → {_LOG_ROOT}\n")

    _write_summary(all_results, ghl_tool_names)

    active = [r for r in all_results if not r.skipped]
    assert active, "No models had API keys — set at least DEEPSEEK_API_KEY."
    errors = [r for r in active if r.error]
    if errors:
        msgs = "\n".join(f"  {r.model_label}/{r.case_id}: {r.error}" for r in errors[:10])
        pytest.fail(f"{len(errors)} cases errored:\n{msgs}")


# ---------------------------------------------------------------------------
# Per-model parametrized
# ---------------------------------------------------------------------------

_MODEL_IDS = [m.label for m in _ALL_MODELS]


@pytest.mark.live
@pytest.mark.parametrize("model", _ALL_MODELS, ids=_MODEL_IDS)
async def test_arlo_ghl_live_model(model: _ModelSpec) -> None:
    """Run all cases for one model with real GHL data.

    Example:
        pytest tests/eval/test_arlo_ghl_live.py -k "MiniMax M2.7" -v -s -m live
    """
    s = get_settings()
    if not s.ghl_eval_token:
        pytest.skip("GHL_EVAL_TOKEN not set.")
    if not s.ghl_eval_location_id:
        pytest.skip("GHL_EVAL_LOCATION_ID not set.")

    api_key = _api_key_for(model.provider)
    if api_key is None:
        pytest.skip(f"no {model.env_key} configured.")

    print(f"\nFetching GHL schemas ...")
    lc_tools, ghl_schemas = await _fetch_ghl_schemas(s.ghl_eval_token, s.ghl_eval_location_id)
    ghl_tool_names = [
        sc["function"]["name"] for sc in ghl_schemas
        if sc["function"]["name"] != "submit_plan_for_approval"
    ]
    print(f"Discovered {len(ghl_tool_names)} GHL tools")
    print(f"\n{'─' * 70}")
    print(f"  Model: {model.label} ({model.model_id})")
    print(f"{'─' * 70}")

    results: list[LiveEvalResult] = []
    for case in _CASES:
        print(f"\n  [{case.category}] {case.case_id}")
        print(f"  Q: {case.user_request}")
        r = await _eval_one(
            model, case, api_key,
            lc_tools, ghl_schemas,
            s.ghl_eval_token, s.ghl_eval_location_id,
        )
        results.append(r)
        _write_case_log(r)

        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            scores = f"routing={'✓' if r.routing_ok else '✗'}  grounded={'✓' if r.grounded else '✗'}  voice={'✓' if r.voice_ok else '✗'}"
            print(f"  {scores}  {r.latency_s:.1f}s  ${r.total_cost_usd:.5f}  turns={r.turns}")
            if r.tool_calls_made:
                for tc in r.tool_calls_made:
                    try:
                        args_preview = json.dumps(json.loads(tc["arguments"]), default=str)[:80]
                    except Exception:
                        args_preview = tc["arguments"][:80]
                    print(f"  → {tc['name']}({args_preview})")
            if r.plan_steps:
                for step in r.plan_steps:
                    print(f"  → plan: {step.get('tool_name')}  args={json.dumps(step.get('args', {}))[:80]}")
            if r.final_response:
                print(f"  ARLO: {r.final_response[:200]}")

    _write_summary(results, ghl_tool_names)
    errors = [r for r in results if r.error]
    if errors:
        msgs = "\n".join(f"  {r.case_id}: {r.error}" for r in errors)
        pytest.fail(f"Errors:\n{msgs}")
