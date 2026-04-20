"""
scoring_rules.py — Post-hoc rule-based re-ranking for RCA candidates.

This module enforces the scoring contract described in the paper:

    final_score = evidence_score + topology_score + temporal_score + propagation_score

with the hard constraints:

    (H1) If a candidate has ZERO direct evidence (log_agrees=False AND
         tcb_agrees=False), its confidence is capped at CAP_NO_EVIDENCE (0.25).
         Topology-agrees alone does NOT count as evidence — topology is a
         structural hint, not an observation of anomaly.

    (H2) A candidate with evidence_score == 0 cannot be Top-1 if there
         exists any other candidate with evidence_score > 0.

    (H3) A candidate whose first-observed anomaly is LATER than the symptom
         service's first anomaly cannot be Top-1 (causes precede symptoms).
         Enforced as a sort-tier key (the later candidate drops to rank 2+).

    (H4) A candidate supported ONLY by topology (topology_agrees=True but
         both log_agrees=False and tcb_agrees=False) is capped at
         CAP_TOPOLOGY_ONLY (0.35).

Rationale: the v5 diagnosis showed four "overconfident" failures where the
LLM, with no log/TCB evidence, promoted a plausible-looking topology upstream
(e.g. adservice) to confidence ≈0.68. These rules are deterministic safeguards
against that failure mode. They do not remove LLM agency — the LLM still
proposes the candidate list and its own confidence values. The rules only
DEMOTE candidates that violate the evidence requirement.

Design choices:

- All rules are **demotion-only**: we never promote a candidate the LLM
  didn't surface, and we never raise confidence above the LLM's value. This
  preserves the LLM's final-say on content while adding a floor of rigour.
- Rules are applied in a pure function so the behaviour is unit-testable
  without LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


CAP_NO_EVIDENCE = 0.25      # H1
CAP_TOPOLOGY_ONLY = 0.35    # H4


def _support(candidate: Dict[str, Any]) -> Dict[str, bool]:
    """Extract support flags safely from a candidate dict."""
    se = candidate.get("supporting_evidence") or {}
    return {
        "log":      bool(se.get("log_agent_agrees", False)),
        "topology": bool(se.get("topology_agent_agrees", False)),
        "tcb":      bool(se.get("tcb_rca_agrees", False)),
    }


def _evidence_score(candidate: Dict[str, Any]) -> float:
    """Direct-observation evidence only (log + tcb). Topology excluded."""
    s = _support(candidate)
    return (int(s["log"]) + int(s["tcb"])) / 2.0


def apply_hard_rules(
    candidates: List[Dict[str, Any]],
    symptom_service: str,
    symptom_first_anomaly_ts: Optional[str] = None,
    tcb_temporal_gaps: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Demote candidates that violate evidence / causality requirements.

    Args:
        candidates: list of dicts with 'cause_service', 'confidence',
                    and 'supporting_evidence' at minimum.
        symptom_service: the user-visible symptom service name.
        symptom_first_anomaly_ts: ISO timestamp of the symptom's first
                    observed anomaly, if known. Used for H3.
        tcb_temporal_gaps: optional {service: gap_seconds_before_symptom}.
                    Positive = earlier than symptom (good).
                    Negative = later (violates causality, H3).

    Returns:
        New list (inputs not mutated) with:
          - confidence capped per H1/H4
          - an extra diagnostic field `_scoring_notes` on each candidate
          - re-sorted so H2/H3 violators drop below compliant ones
    """
    if not candidates:
        return []

    tcb_temporal_gaps = tcb_temporal_gaps or {}
    result: List[Dict[str, Any]] = []

    for cand in candidates:
        c = dict(cand)  # shallow copy; don't mutate caller data
        notes: List[str] = []

        support = _support(c)
        ev = _evidence_score(c)
        raw_conf = float(c.get("confidence", 0.5) or 0.0)
        new_conf = raw_conf

        # H1: zero direct evidence
        if ev == 0 and new_conf > CAP_NO_EVIDENCE:
            new_conf = CAP_NO_EVIDENCE
            notes.append(f"H1: no direct evidence → cap at {CAP_NO_EVIDENCE}")

        # H4: topology-only support
        only_topology = (
            support["topology"] and not support["log"] and not support["tcb"]
        )
        if only_topology and new_conf > CAP_TOPOLOGY_ONLY:
            new_conf = CAP_TOPOLOGY_ONLY
            notes.append(f"H4: topology-only → cap at {CAP_TOPOLOGY_ONLY}")

        # H3: temporal violation flag (used for sort tier below)
        svc = c.get("cause_service") or ""
        temporal_violation = False
        gap = tcb_temporal_gaps.get(svc)
        if gap is not None and gap < 0:
            # This candidate's first anomaly is LATER than symptom's.
            temporal_violation = True
            notes.append(
                f"H3: anomaly is {-gap:.0f}s later than symptom → causality violation"
            )

        # Self-loop sanity: symptom service cannot be its own root cause at Top-1
        # (useful for Online Boutique where symptom is always 'frontend')
        is_symptom_self = (svc == symptom_service)

        c["confidence"] = new_conf
        c["_evidence_score"] = ev
        c["_scoring_notes"] = notes
        c["_temporal_violation"] = temporal_violation
        c["_is_symptom_self"] = is_symptom_self
        result.append(c)

    # ----- Sort with tiered key (lower tier is better) -----
    # Tier 0: has direct evidence, no demerits  → best
    # Tier 1: has direct evidence but temporal/self-symptom demerit
    # Tier 2: no direct evidence (H1 cap) but also not self-symptom/temporal
    # Tier 3: no direct evidence AND (self-symptom OR temporal violation) → worst
    # Within a tier, sort by confidence desc.
    def _tier(c: Dict[str, Any]) -> int:
        ev = c.get("_evidence_score", 0.0) or 0.0
        demerited = bool(c.get("_temporal_violation") or c.get("_is_symptom_self"))
        if ev > 0:
            return 1 if demerited else 0
        return 3 if demerited else 2

    result.sort(key=lambda c: (_tier(c), -float(c.get("confidence", 0.0))))

    # ----- H2: prevent evidence=0 from being Top-1 if any compliant exists -----
    # Already handled by the tier sort (tier 2 drops below tier 0/1). Explicit
    # check below for defensive correctness.
    if result and _evidence_score(result[0]) == 0:
        for i in range(1, len(result)):
            if _evidence_score(result[i]) > 0:
                # Promote the first compliant candidate to Top-1, demote the rest.
                compliant = result.pop(i)
                result.insert(0, compliant)
                break

    # Re-number ranks 1..N
    for idx, c in enumerate(result, 1):
        c["rank"] = idx

    return result


def build_temporal_gaps(tcb_rca_output: Any, symptom_service: str) -> Dict[str, float]:
    """Extract per-service temporal gaps from TCB-RCA output.

    Returns {service_name: seconds_before_symptom_anomaly}.
    Positive value = the service's first anomaly preceded the symptom's.
    Negative value = the service's first anomaly came AFTER the symptom's
    (a causality violation — these candidates should not be Top-1).

    Services without anomaly timing data are omitted from the dict.
    """
    gaps: Dict[str, float] = {}
    try:
        candidates = getattr(tcb_rca_output, "root_cause_candidates", None) or []
        for rc in candidates:
            svc = getattr(rc, "cause_service", None)
            gap = getattr(rc, "temporal_gap_seconds", None)
            if svc and gap is not None:
                # TCB-RCA reports temporal_gap_seconds as age before symptom;
                # positive values mean the anomaly precedes the symptom.
                gaps[svc] = float(gap)
    except Exception:
        pass
    return gaps
