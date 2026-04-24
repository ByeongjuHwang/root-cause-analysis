"""
Phase 4c unit tests: Verifier 4-dimension ConsistencyChecks.

Tests the standalone helpers in agents/verifier_agent/service.py
that compute the paper-mandated 4-dimension consistency. We do NOT
spin up the full Verifier HTTP service — the integration is exercised
by the existing smoke tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from common.a2a_contract import ConsistencyChecks
from common.evidence import EvidenceCollection, EvidenceUnit, TimeRange
from agents.verifier_agent.service import (
    _check_counter_evidence,
    _compute_consistency_checks,
    _extract_evidence_collection_from_agents,
    _extract_upstream_rca_response,
    COUNTER_EVIDENCE_SEVERITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tr():
    return TimeRange(
        start="2026-04-24T13:00:00+09:00",
        end="2026-04-24T13:05:00+09:00",
    )


def _ev(service, severity=0.8, modality="metric", anomaly="resource_saturation"):
    return EvidenceUnit(
        modality=modality,
        time_range=_tr(),
        services=[service],
        anomaly_type=anomaly,
        severity=severity,
        observation={},
        source="test_fixture",
    )


def _full_signals(**overrides):
    """Default to all-positive signals (meeting cross-modality and temporal)."""
    d = {
        "has_log_evidence": True,
        "has_trace_support": True,
        "has_metric_shift": True,
        "has_topology_path": True,
        "is_temporally_prior": True,
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# _check_counter_evidence
# ---------------------------------------------------------------------------

class TestCheckCounterEvidence:
    def test_no_collection_returns_none(self):
        """None evidence collection → cannot check."""
        assert _check_counter_evidence("svc", None) is None

    def test_empty_collection_returns_none(self):
        assert _check_counter_evidence("svc", EvidenceCollection()) is None

    def test_no_evidence_for_service_returns_none(self):
        col = EvidenceCollection()
        col.add(_ev("other_service", severity=0.9))
        assert _check_counter_evidence("my_service", col) is None

    def test_strong_evidence_means_no_counter(self):
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.9))
        # severity >> threshold → True (consistent, no counter-evidence)
        assert _check_counter_evidence("svc", col) is True

    def test_weak_evidence_is_counter(self):
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.1))  # well below 0.25 threshold
        # Only weak signal → counter-evidence
        assert _check_counter_evidence("svc", col) is False

    def test_max_severity_used_not_min(self):
        """Multiple evidences — should take the max severity."""
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.1, modality="log", anomaly="volume_shift"))
        col.add(_ev("svc", severity=0.9, modality="metric"))
        # Has strong metric signal → no counter
        assert _check_counter_evidence("svc", col) is True

    def test_case_insensitive_service_match(self):
        col = EvidenceCollection()
        col.add(_ev("checkoutservice", severity=0.9))
        # EvidenceCollection.by_service handles case
        assert _check_counter_evidence("CheckoutService", col) is True

    def test_threshold_boundary(self):
        col = EvidenceCollection()
        col.add(_ev("svc", severity=COUNTER_EVIDENCE_SEVERITY_THRESHOLD))
        # exactly at threshold — >= means no counter-evidence (pass)
        assert _check_counter_evidence("svc", col) is True


# ---------------------------------------------------------------------------
# _compute_consistency_checks
# ---------------------------------------------------------------------------

class TestComputeConsistencyChecks:
    def test_free_text_candidate_returns_empty(self):
        """No cause_service → all None (can't check anything)."""
        cc = _compute_consistency_checks(
            candidate_service=None,
            signals={},
            modality_count=0,
            evidence_collection=None,
        )
        assert cc.temporal is None
        assert cc.topological is None
        assert cc.modality is None
        assert cc.counter_evidence is None

    def test_all_signals_positive_all_dimensions_pass(self):
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.9))
        cc = _compute_consistency_checks(
            candidate_service="svc",
            signals=_full_signals(),
            modality_count=3,
            evidence_collection=col,
        )
        assert cc.temporal is True
        assert cc.topological is True
        assert cc.modality is True
        assert cc.counter_evidence is True
        assert cc.passed() is True

    def test_temporal_violation_maps(self):
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.9))
        cc = _compute_consistency_checks(
            candidate_service="svc",
            signals=_full_signals(is_temporally_prior=False),
            modality_count=2,
            evidence_collection=col,
        )
        assert cc.temporal is False
        assert "temporal" in cc.failed_dimensions()
        assert cc.passed() is False

    def test_topology_missing_maps(self):
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.9))
        cc = _compute_consistency_checks(
            candidate_service="svc",
            signals=_full_signals(has_topology_path=False),
            modality_count=2,
            evidence_collection=col,
        )
        assert cc.topological is False

    def test_single_modality_fails_cross_context(self):
        """Cross-context requires >=2 modalities."""
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.9))
        cc = _compute_consistency_checks(
            candidate_service="svc",
            signals=_full_signals(),
            modality_count=1,  # only 1 modality
            evidence_collection=col,
        )
        assert cc.modality is False

    def test_zero_modality_count_yields_none(self):
        """0 modalities → can't check (would be dropped by R2 anyway)."""
        cc = _compute_consistency_checks(
            candidate_service="svc",
            signals=_full_signals(),
            modality_count=0,
            evidence_collection=None,
        )
        assert cc.modality is None

    def test_counter_evidence_weak_severity_fails(self):
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.1))  # weak
        cc = _compute_consistency_checks(
            candidate_service="svc",
            signals=_full_signals(),
            modality_count=2,
            evidence_collection=col,
        )
        assert cc.counter_evidence is False

    def test_no_evidence_for_service_yields_counter_none(self):
        """Evidence collection exists but none for candidate → cannot judge."""
        col = EvidenceCollection()
        col.add(_ev("other_svc", severity=0.9))
        cc = _compute_consistency_checks(
            candidate_service="svc",
            signals=_full_signals(),
            modality_count=2,
            evidence_collection=col,
        )
        assert cc.counter_evidence is None

    def test_multi_dim_failure_all_reported(self):
        col = EvidenceCollection()
        col.add(_ev("svc", severity=0.05))  # weak counter-ev
        cc = _compute_consistency_checks(
            candidate_service="svc",
            signals=_full_signals(
                is_temporally_prior=False,
                has_topology_path=False,
            ),
            modality_count=1,  # single modality
            evidence_collection=col,
        )
        failed = set(cc.failed_dimensions())
        assert "temporal" in failed
        assert "topological" in failed
        assert "modality" in failed
        assert "counter_evidence" in failed


# ---------------------------------------------------------------------------
# _extract_evidence_collection_from_agents
# ---------------------------------------------------------------------------

class TestExtractEvidenceCollection:
    def _ev_payload(self, service="svc", severity=0.8):
        return {
            "count": 1,
            "services_covered": [service],
            "modalities_present": ["metric"],
            "units": [
                {
                    "evidence_id": "ev_metric_abc",
                    "modality": "metric",
                    "anomaly_type": "resource_saturation",
                    "severity": severity,
                    "services": [service],
                    "time_range": {
                        "start": "2026-04-24T13:00:00+09:00",
                        "end": "2026-04-24T13:05:00+09:00",
                    },
                    "observation": {},
                    "source": "test",
                    "topology_path": None,
                    "supporting_refs": [],
                    "raw_samples": [],
                    "preceded_by": [],
                }
            ],
        }

    def test_empty_agents_returns_none(self):
        assert _extract_evidence_collection_from_agents({}, {}) is None

    def test_phase3b_fallback_legacy_payload(self):
        """When RCA has no _agent_response, fall back to log_result['evidence_collection']."""
        log_result = {"evidence_collection": self._ev_payload("svc")}
        rca_result = {}
        col = _extract_evidence_collection_from_agents(log_result, rca_result)
        assert col is not None
        assert len(col) == 1
        assert "svc" in col.services_covered()

    def test_phase4b_contract_path_preferred(self):
        """When _agent_response present on RCA, use it over Phase 3b legacy."""
        rca_result = {
            "_agent_response": {
                "agent_name": "rca_agent",
                "request_id": "INC-X",
                "candidates": [],
                "evidence_collection": self._ev_payload("from_rca_contract"),
                "completeness_score": 0.5,
                "consistency_checks": None,
                "recommended_next_actions": [],
                "reasoning": "",
            }
        }
        # Log result has a different payload — should be ignored
        log_result = {"evidence_collection": self._ev_payload("from_log_legacy")}
        col = _extract_evidence_collection_from_agents(log_result, rca_result)
        assert col is not None
        assert "from_rca_contract" in col.services_covered()
        assert "from_log_legacy" not in col.services_covered()

    def test_log_contract_used_when_rca_absent(self):
        rca_result = {}
        log_result = {
            "_agent_response": {
                "agent_name": "log_agent",
                "request_id": "INC-X",
                "candidates": [],
                "evidence_collection": self._ev_payload("from_log_contract"),
                "completeness_score": 0.5,
                "consistency_checks": None,
                "recommended_next_actions": [],
                "reasoning": "",
            },
            # Legacy field also present but should not be used when contract is
            "evidence_collection": self._ev_payload("from_log_legacy"),
        }
        col = _extract_evidence_collection_from_agents(log_result, rca_result)
        assert col is not None
        assert "from_log_contract" in col.services_covered()


# ---------------------------------------------------------------------------
# _extract_upstream_rca_response
# ---------------------------------------------------------------------------

class TestExtractUpstreamRcaResponse:
    def test_none_when_no_contract(self):
        assert _extract_upstream_rca_response({}) is None

    def test_none_when_contract_malformed(self):
        bad = {"_agent_response": {"agent_name": "garbage_value"}}
        # Missing required fields / invalid AgentName → None, not crash
        assert _extract_upstream_rca_response(bad) is None

    def test_reconstructs_valid_contract(self):
        contract = {
            "agent_name": "rca_agent",
            "request_id": "INC-Y",
            "candidates": [
                {
                    "service": "recommendationservice",
                    "confidence": 0.85,
                    "supporting_evidence": [],
                    "assumptions": [],
                    "missing_evidence": [],
                    "reasoning": "",
                    "topology_path": None,
                    "failure_mode": "resource_exhaustion",
                },
            ],
            "evidence_collection": {"units": []},
            "completeness_score": 0.7,
            "consistency_checks": None,
            "recommended_next_actions": [],
            "reasoning": "",
        }
        resp = _extract_upstream_rca_response({"_agent_response": contract})
        assert resp is not None
        assert resp.agent_name == "rca_agent"
        assert resp.candidates[0].failure_mode == "resource_exhaustion"


# ---------------------------------------------------------------------------
# Integration: build_verifier_agent_response
# ---------------------------------------------------------------------------

from common.a2a_contract import AgentResponse, Candidate
from common.response_builder import build_verifier_agent_response


class TestBuildVerifierAgentResponse:
    def test_minimal_legacy(self):
        legacy = {
            "incident_id": "INC-V1",
            "service": "frontend",
            "verdict": "accepted",
            "revised_root_cause_candidates": [
                {
                    "rank": 1,
                    "cause_service": "recommendationservice",
                    "confidence": 0.82,
                    "reasoning": "CPU saturation confirmed",
                },
            ],
            "final_confidence": 0.82,
            "explanation": "all signals agree",
            "rejected_candidates_count": 0,
        }
        resp = build_verifier_agent_response(
            legacy, request_id="INC-V1",
            consistency_by_service={},
        )
        assert resp.agent_name == "verifier_agent"
        assert len(resp.candidates) == 1
        assert resp.candidates[0].service == "recommendationservice"

    def test_consistency_attached_to_top_level(self):
        cc = ConsistencyChecks(
            temporal=True, topological=True,
            modality=True, counter_evidence=True,
        )
        legacy = {
            "incident_id": "INC-V2",
            "service": "frontend",
            "verdict": "accepted",
            "revised_root_cause_candidates": [
                {"rank": 1, "cause_service": "svc",
                 "confidence": 0.9, "reasoning": ""},
            ],
            "final_confidence": 0.9,
            "explanation": "",
            "rejected_candidates_count": 0,
        }
        resp = build_verifier_agent_response(
            legacy, request_id="INC-V2",
            consistency_by_service={"svc": cc},
        )
        # Top candidate is "svc" — its consistency should be lifted to the top
        assert resp.consistency_checks is not None
        assert resp.consistency_checks.passed() is True

    def test_per_candidate_assumptions_include_consistency(self):
        cc = ConsistencyChecks(
            temporal=True, topological=False,
            modality=True, counter_evidence=None,
        )
        legacy = {
            "incident_id": "INC-V3",
            "service": "frontend",
            "verdict": "revised",
            "revised_root_cause_candidates": [
                {"rank": 1, "cause_service": "svc",
                 "confidence": 0.5, "reasoning": ""},
            ],
            "final_confidence": 0.5,
            "explanation": "",
            "rejected_candidates_count": 0,
        }
        resp = build_verifier_agent_response(
            legacy, request_id="INC-V3",
            consistency_by_service={"svc": cc},
        )
        assump_text = " ".join(resp.candidates[0].assumptions)
        assert "temporal=OK" in assump_text
        assert "topological=FAIL" in assump_text
        assert "counter_evidence=NA" in assump_text

    def test_rejected_count_in_assumptions(self):
        legacy = {
            "incident_id": "INC-V4",
            "service": "frontend",
            "verdict": "accepted",
            "revised_root_cause_candidates": [
                {"rank": 1, "cause_service": "svc",
                 "confidence": 0.8, "reasoning": ""},
            ],
            "final_confidence": 0.8,
            "explanation": "",
            "rejected_candidates_count": 2,
        }
        resp = build_verifier_agent_response(
            legacy, request_id="INC-V4",
        )
        assump = resp.candidates[0].assumptions
        assert any("rejected_during_verify=2" in a for a in assump)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
