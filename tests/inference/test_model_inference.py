"""Live inference benchmark — all 6 provider models.

Runs real API calls to compare latency, throughput, and cost across:

    Provider   | Quality | Model
    -----------+---------+-------------------------
    DeepSeek   | Pro     | deepseek/deepseek-v4-pro
    DeepSeek   | Flash   | deepseek/deepseek-v4-flash
    MiniMax    | Pro     | openai/MiniMax-M2.7
    MiniMax    | Flash   | openai/MiniMax-M2.5
    Kimi       | Pro     | openai/kimi-k2.6
    Kimi       | Flash   | openai/kimi-k2.5

Each model is called with an identical prompt. For every model the test
prints:
  - Latency        : wall-clock seconds to receive the full response
  - Output TPS     : output tokens ÷ latency (tokens per second)
  - Estimated cost : USD reported by LiteLLM's hidden_params
  - Response snip  : first 160 chars of the reply

Models whose provider API key is missing in .env are skipped automatically
so you can run the file even with only DeepSeek configured today.

Run (show printed output):
    pytest tests/inference/test_model_inference.py -v -s -m live

Skip live tests in CI:
    pytest -m "not live"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest
from lyra_core.common.config import get_settings
from lyra_core.common.llm import _call_resolved, estimate_cost
from lyra_core.llm.router import ResolvedModel

# ---------------------------------------------------------------------------
# Benchmark prompt — identical for every model so results are comparable.
# ---------------------------------------------------------------------------

_PROMPT = (
    "You are a concise assistant. "
    "Explain in exactly three bullet points why async/await improves Python "
    "web server throughput over synchronous I/O. Be brief."
)

_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": _PROMPT}]

# ---------------------------------------------------------------------------
# Model registry — one entry per model under test.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelSpec:
    label: str  # human-readable label shown in output
    provider: str  # catalog key (deepseek / minimax / moonshot)
    model_id: str  # LiteLLM model identifier
    api_base: str  # provider endpoint
    env_key: str  # env var name shown in skip messages


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


def _api_key_for(provider: str) -> str | None:
    """Resolve the API key for a provider from Settings."""
    s = get_settings()
    mapping = {
        "deepseek": s.deepseek_api_key,
        "minimax": s.minimax_api_key,
        "moonshot": s.kimi_api_key,
        "openai": s.openai_api_key,
        "anthropic": s.anthropic_api_key,
        "gemini": s.google_api_key,
    }
    raw = mapping.get(provider, "")
    return raw or None


def _build_resolved(spec: _ModelSpec, api_key: str) -> ResolvedModel:
    return ResolvedModel(
        tier=f"inference_bench_{spec.provider}",
        provider_key=spec.provider,
        model_id=spec.model_id,
        api_key=api_key,
        api_base=spec.api_base,
        source="env",
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    label: str
    model_id: str
    latency_s: float = 0.0
    output_tokens: int = 0
    tps: float = 0.0  # output tokens per second
    cost_usd: float = 0.0
    response_snip: str = ""
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

_SEPARATOR = "=" * 90
_COL = "{:<22} {:<26} {:>9} {:>10} {:>12}  {}"


def _print_results(results: list[_Result]) -> None:
    print("\n")
    print(_SEPARATOR)
    print("  LYRALABS — INFERENCE BENCHMARK")
    print(_SEPARATOR)
    print(
        _COL.format(
            "Model", "Model ID", "Latency", "Output TPS", "Cost (USD)", "Response (first 160 chars)"
        )
    )
    print("-" * 90)

    for r in results:
        if r.skipped:
            print(_COL.format(r.label, r.model_id, "—", "—", "—", f"[SKIPPED] {r.skip_reason}"))
        elif r.error:
            print(_COL.format(r.label, r.model_id, "—", "—", "—", f"[ERROR] {r.error[:60]}"))
        else:
            print(
                _COL.format(
                    r.label,
                    r.model_id,
                    f"{r.latency_s:.2f}s",
                    f"{r.tps:.1f} tok/s",
                    f"${r.cost_usd:.6f}",
                    r.response_snip,
                )
            )

    print(_SEPARATOR)
    print()


async def _bench_one(spec: _ModelSpec) -> _Result:
    """Call a single model and return a populated _Result."""
    result = _Result(label=spec.label, model_id=spec.model_id)

    api_key = _api_key_for(spec.provider)
    if api_key is None:
        result.skipped = True
        result.skip_reason = f"no {spec.env_key} in env"
        return result

    resolved = _build_resolved(spec, api_key)

    try:
        t0 = time.monotonic()
        response = await _call_resolved(
            resolved,
            messages=_MESSAGES,
            max_tokens=512,
            temperature=0.2,
            timeout_s=90.0,
        )
        result.latency_s = time.monotonic() - t0

        # Usage
        usage = getattr(response, "usage", None)
        result.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        result.tps = result.output_tokens / result.latency_s if result.latency_s > 0 else 0.0

        # Cost
        result.cost_usd = estimate_cost(response)

        # Response text
        try:
            text = response.choices[0].message.content or ""
        except Exception:
            text = ""
        result.response_snip = text[:160].replace("\n", " ")

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:200]

    return result


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.live
async def test_inference_benchmark() -> None:
    """Run all 6 models, print a formatted comparison table, assert no crashes.

    The test PASSES as long as at least one model responds successfully.
    Skip markers on individual models when their API key is absent — so the
    file can be run before MiniMax / Kimi keys are set.
    """
    print(f"\n\nPrompt: {_PROMPT!r}\n")

    results: list[_Result] = []
    for spec in _ALL_MODELS:
        print(f"  → calling {spec.label} ({spec.model_id}) ...", flush=True)
        r = await _bench_one(spec)
        results.append(r)

        if r.skipped:
            print(f"     skipped: {r.skip_reason}")
        elif r.error:
            print(f"     ERROR: {r.error[:80]}")
        else:
            print(f"     OK  latency={r.latency_s:.2f}s  tps={r.tps:.1f}  cost=${r.cost_usd:.6f}")

    _print_results(results)

    # At least one provider must have a configured key to consider this run useful.
    active = [r for r in results if not r.skipped]
    assert active, "All 6 models were skipped — check that at least DEEPSEEK_API_KEY is set in .env"

    # Every active model that didn't get skipped must not have errored.
    failed = [r for r in active if r.error]
    if failed:
        msgs = "\n".join(f"  {r.label}: {r.error}" for r in failed)
        pytest.fail(f"The following models returned errors:\n{msgs}")


# ---------------------------------------------------------------------------
# Individual parametrised tests — one per model, skip if no key
# ---------------------------------------------------------------------------

_PARAM_IDS = [s.label for s in _ALL_MODELS]


@pytest.mark.live
@pytest.mark.parametrize("spec", _ALL_MODELS, ids=_PARAM_IDS)
async def test_single_model(spec: _ModelSpec) -> None:
    """One isolated test per model — makes it easy to run a single model:

    pytest tests/inference/test_model_inference.py -k "DeepSeek V4-Pro" -v -s
    """
    api_key = _api_key_for(spec.provider)
    if api_key is None:
        pytest.skip(f"no {spec.env_key} configured")

    resolved = _build_resolved(spec, api_key)

    t0 = time.monotonic()
    response = await _call_resolved(
        resolved,
        messages=_MESSAGES,
        max_tokens=512,
        temperature=0.2,
        timeout_s=90.0,
    )
    latency = time.monotonic() - t0

    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    tps = output_tokens / latency if latency > 0 else 0.0
    cost = estimate_cost(response)

    try:
        text = response.choices[0].message.content or ""
    except Exception:
        text = ""

    print(
        f"\n{'─' * 60}\n"
        f"  Model     : {spec.label}\n"
        f"  Model ID  : {spec.model_id}\n"
        f"  Latency   : {latency:.2f}s\n"
        f"  Out tokens: {output_tokens}\n"
        f"  TPS       : {tps:.1f} tok/s\n"
        f"  Cost      : ${cost:.6f}\n"
        f"  Response  :\n{text}\n"
        f"{'─' * 60}"
    )

    assert text, f"{spec.label} returned an empty response"
    assert latency < 90.0, f"{spec.label} exceeded 90s timeout"
