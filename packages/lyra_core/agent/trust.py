"""Plan-step trust classification helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tools.base import RiskProfile, TrustTier

if TYPE_CHECKING:
    from ..tools.base import Tool
    from .state import PlanStep


def classify_step(step: "PlanStep", tool: "Tool") -> RiskProfile:
    """Return a RiskProfile for a single plan step.

    Explicit `trust_tier` ClassVar on the tool takes precedence.
    A tool with requires_approval=False is always LOW regardless of any other attribute.
    A tool with blast_radius="bulk" is always HIGH.
    """
    if not tool.requires_approval:
        return RiskProfile(tier=TrustTier.LOW, reversibility="reversible", blast_radius="single")
    tier: TrustTier = getattr(tool, "trust_tier", TrustTier.MEDIUM)
    blast_radius = getattr(tool, "blast_radius", "single")
    if tier == TrustTier.HIGH or blast_radius == "bulk":
        return RiskProfile(
            tier=TrustTier.HIGH,
            reversibility="irreversible",
            blast_radius=blast_radius,
        )
    return RiskProfile(tier=TrustTier.MEDIUM, reversibility="reversible", blast_radius=blast_radius)


def overall_plan_tier(profiles: list[RiskProfile]) -> TrustTier:
    """Return the highest tier across all profiles."""
    order = {TrustTier.LOW: 0, TrustTier.MEDIUM: 1, TrustTier.HIGH: 2}
    return max(profiles, key=lambda p: order[p.tier]).tier if profiles else TrustTier.LOW
