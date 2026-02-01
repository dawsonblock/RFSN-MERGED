"""Diff policy - control patch size and risk.

Enforces:
- max_diff_lines
- Risk scoring (large deletions, broad changes)
"""

from __future__ import annotations

from typing import Tuple

from agent.types import AgentState, Proposal
from rfsn_controller.patch_hygiene import validate_patch_hygiene
from agent.profiles import Profile


def check_diff(profile: Profile, state: AgentState, proposal: Proposal) -> Tuple[bool, str]:
    """Check diff size and risk.
    
    Args:
        profile: Profile with limits
        state: Current state
        proposal: Proposed action
        
    Returns:
        (allowed, reason)
    """
    if proposal.kind != "edit":
        return True, "ok"
    
    # Check diff size
    diff_text = proposal.inputs.get("diff", "")
    
    if diff_text:
        lines = diff_text.split("\n")
        added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        total_changes = added + removed
        
        if total_changes > profile.max_diff_lines:
            return False, (
                f"diff too large: {total_changes} lines > {profile.max_diff_lines}"
            )
        
        # Risk check: large deletions are risky
        if removed > added * 2:
            # More than 2x deletions vs additions
            risk_score = removed / max(added, 1)
            if risk_score > 5.0:
                return False, (
                    f"diff too risky: {removed} deletions vs {added} additions (ratio {risk_score:.1f})"
                )
    
    # Additional hygiene checks to catch unsafe patterns and forbidden paths.
    # Validate patch hygiene using rfsn_controller.patch_hygiene. This call
    # enforces forbidden file edits, lockfile protections, test deletion bans,
    # debug print suppression, and other quality gates. If the patch violates
    # any hygiene rules, reject the proposal with a descriptive reason.
    hygiene_result = validate_patch_hygiene(diff_text)
    if not hygiene_result.is_valid:
        # Summarize the first violation as the gate reason. Detailed violations
        # are available on the result for logging.
        violation = (
            hygiene_result.violations[0]
            if hygiene_result.violations
            else "diff hygiene violation"
        )
        return False, f"diff hygiene violation: {violation}"

    return True, "ok"
