"""Canonical gate composition for the RFSN agent.

This module defines a single `gate()` function that enforces all
deterministic policies for proposal validation.  It composes
phase, file, diff and test policies from the `gate_ext` package and
checks episode budgets via `agent.loop.check_budgets`.  The gate
never learns or mutates state; it returns a `GateDecision` from
`agent.types` with a reason and optional constraint metadata.
"""

from __future__ import annotations

from typing import Any, Dict

from agent.types import AgentState, Proposal, GateDecision
from agent.profiles import Profile
from agent.loop import check_budgets

# Import policy checks from gate_ext.  Each returns (bool, reason).
from gate_ext.policy_phase import check_phase
from gate_ext.policy_files import check_files
from gate_ext.policy_diff import check_diff
from gate_ext.policy_tests import check_tests


def gate(profile: Profile, state: AgentState, proposal: Proposal) -> GateDecision:
    """Validate a proposal against all configured policies.

    The gate enforces:
    - Budget constraints (rounds, tests, patches, model calls)
    - Phase constraints (allowed proposal kinds in the current phase)
    - File constraints (vendor/CI edits, max files per episode)
    - Diff constraints (patch size and risk)
    - Test constraints (forbid test modifications in Verified profile)

    Returns a `GateDecision` with `accept=True` if the proposal is allowed,
    otherwise `accept=False` with a reason.

    Args:
        profile: Agent profile controlling budgets and policies.
        state: Current agent state.
        proposal: Proposed action from the planner.

    Returns:
        GateDecision
    """
    # First, check global budgets (rounds, patches, tests, model calls)
    within_budget, reason = check_budgets(state, profile)
    if not within_budget:
        return GateDecision(
            accept=False,
            reason=f"budget exceeded: {reason}",
            constraints={"budget": reason},
        )

    # Phase policy: is this kind allowed in the current phase?
    ok, phase_reason = check_phase(profile, state, proposal)
    if not ok:
        return GateDecision(
            accept=False,
            reason=phase_reason,
            constraints={"phase": state.phase.value, "kind": proposal.kind},
        )

    # File policy: are edited files allowed?
    ok, file_reason = check_files(profile, state, proposal)
    if not ok:
        return GateDecision(
            accept=False,
            reason=file_reason,
            constraints={"files": proposal.inputs.get("files", [])},
        )

    # Diff policy: is the patch size/risk acceptable?
    ok, diff_reason = check_diff(profile, state, proposal)
    if not ok:
        return GateDecision(
            accept=False,
            reason=diff_reason,
            constraints={"diff": proposal.inputs.get("diff", "")},
        )

    # Test policy: forbid test edits and ensure suite requirements
    ok, test_reason = check_tests(profile, state, proposal)
    if not ok:
        return GateDecision(
            accept=False,
            reason=test_reason,
            constraints={"tests": True},
        )

    # All checks passed
    return GateDecision(accept=True, reason="all constraints satisfied", constraints={})