"""
Phase 4b unit tests: AgentResponse builder + attach helpers.

Verifies that legacy dicts from each Agent can be packaged into structured
AgentResponse objects without losing information, and that the attach helper
embeds them as JSON-serialisable dicts.

These tests do NOT spin up agents/MCP servers — they exercise the pure
builder functions in common/response_builder.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from common.a2a_contract import AgentResponse, Candidate, FailureMode
from common.evidence import EvidenceCollection, EvidenceUnit, TimeRange
from common.response_builder import (
    attach_agent_response,
    build_log_agent_response,
    build_rca_agent_response,
    build_topology_agent_response,
    _infer_failure_mode,
    _restore_evidence_collection,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _ev_payload(units):
    """Wrap unit dicts in the payload shape that get_evidence_collection_payload returns."""
    return {
        "count": len(units),
        "services_covered": sorted({s for u in units for s in u["services"]}),
        "modalities_present": sorted({u["modality"] for u in units}),
        "units": units,
    }


def _ev_unit(service, modality="metric", severity=0.7, anomaly="resource_saturation"):
    return {
        "evidence_id": f"ev_{modality}_{abs(hash((service, anomaly))) % 99999999:08x}",
        "modality": modality,
        "anomaly_type": anomaly,
        "severity": severity,
        "services": [service],
        "time_range": {
            "start": "2026-04-23T13:00:00+09:00",
            "end": "2026-04-23T13:05:00+09:00",
        },
        "observation": {},
        "source": "test_fixture",
        "topology_path": None,
        "supporting_refs": [],
        "raw_samples": [],
        "preceded_by": [],
    }


# ---------------------------------------------------------------------------
# _restore_evidence_collection
# ---------------------------------------------------------------------------

class TestRestoreEvidenceCollection:
    def test_none_payload_yields_empty(self):
        result = _restore_evidence_collection(None)
        assert isinstance(result, EvidenceCollection)
        assert len(result) == 0

    def test_empty_payload_yields_empty(self):
        result = _restore_evidence_collection({})
        assert len(result) == 0

    def test_payload_with_units_restored(self):
        payload = _ev_payload([
            _ev_unit("svc1"),
            _ev_unit("svc2", modality="log", anomaly="error_spike"),
        ])
        result = _restore_evidence_collection(payload)
        assert len(result) == 2
        assert "svc1" in result.services_covered()
        assert "svc2" in result.services_covered()

    def test_corrupt_unit_skipped_not_raised(self):
        """A malformed unit must not break the whole restoration."""
        payload = {
            "units": [
                _ev_unit("svc1"),
                {"this_is_garbage": True},  # broken
                _ev_unit("svc2"),
            ]
        }
        result = _restore_evidence_collection(payload)
        # Should restore the 2 valid ones, skip the corrupt one
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _infer_failure_mode heuristic
# ---------------------------------------------------------------------------

class TestFailureModeInference:
    def test_resource_saturation_maps_to_resource_exhaustion(self):
        assert _infer_failure_mode("resource_saturation") == "resource_exhaustion"

    def test_latency_degradation_maps_to_dependency_timeout(self):
        assert _infer_failure_mode("latency_degradation") == "dependency_timeout"

    def test_error_spike_maps_to_cascading_failure(self):
        assert _infer_failure_mode("error_spike") == "cascading_failure"

    def test_unknown_anomaly_returns_none(self):
        assert _infer_failure_mode("some_unmapped_anomaly") is None

    def test_none_input_returns_none(self):
        assert _infer_failure_mode(None) is None


# ---------------------------------------------------------------------------
# build_log_agent_response
# ---------------------------------------------------------------------------

class TestBuildLogAgentResponse:
    def _basic_legacy(self, **overrides):
        base = {
            "summary": "test summary",
            "confidence": 0.75,
            "suspected_downstream": "checkoutservice",
            "anomalous_services": ["checkoutservice"],
            "service_statistics": {
                "window": {"baseline_seconds": 480, "incident_seconds": 60},
                "mode": "dual_window",
            },
            "evidence_collection": None,
        }
        base.update(overrides)
        return base

    def test_minimal_legacy_yields_valid_response(self):
        legacy = self._basic_legacy()
        resp = build_log_agent_response(legacy, request_id="INC-001")
        assert isinstance(resp, AgentResponse)
        assert resp.agent_name == "log_agent"
        assert resp.request_id == "INC-001"
        assert len(resp.candidates) == 1
        assert resp.candidates[0].service == "checkoutservice"
        assert resp.candidates[0].confidence == 0.75

    def test_assumptions_extracted_from_stats(self):
        legacy = self._basic_legacy()
        resp = build_log_agent_response(legacy, request_id="INC-002")
        # primary candidate should have the dual_window assumption
        assumps = resp.candidates[0].assumptions
        assert any("dual_window" in a for a in assumps)
        assert any("baseline=480" in a for a in assumps)

    def test_missing_evidence_lists_absent_modalities(self):
        legacy = self._basic_legacy()
        # No evidence_collection means all modalities are missing
        resp = build_log_agent_response(legacy, request_id="INC-003")
        miss = resp.candidates[0].missing_evidence
        assert "log_evidence_units" in miss
        assert "metric_evidence_units" in miss
        assert "trace_evidence_units" in miss

    def test_evidence_collection_attached_when_present(self):
        legacy = self._basic_legacy(
            evidence_collection=_ev_payload([
                _ev_unit("checkoutservice", modality="metric", severity=0.9),
                _ev_unit("frontend", modality="log", severity=0.4, anomaly="error_spike"),
            ])
        )
        resp = build_log_agent_response(legacy, request_id="INC-004")
        assert len(resp.evidence_collection) == 2
        # Primary candidate's supporting_evidence references the metric unit
        primary = resp.candidates[0]
        assert len(primary.supporting_evidence) == 1
        assert primary.supporting_evidence[0].startswith("ev_metric_")

    def test_failure_mode_inferred_from_strongest_evidence(self):
        legacy = self._basic_legacy(
            suspected_downstream="recommendationservice",
            evidence_collection=_ev_payload([
                _ev_unit("recommendationservice", modality="metric",
                         severity=0.95, anomaly="resource_saturation"),
            ])
        )
        resp = build_log_agent_response(legacy, request_id="INC-005")
        assert resp.candidates[0].failure_mode == "resource_exhaustion"

    def test_secondary_candidates_from_anomalous_services(self):
        legacy = self._basic_legacy(
            suspected_downstream="checkoutservice",
            anomalous_services=["checkoutservice", "frontend", "currencyservice"],
        )
        resp = build_log_agent_response(legacy, request_id="INC-006")
        assert len(resp.candidates) == 3
        services = [c.service for c in resp.candidates]
        assert services[0] == "checkoutservice"  # primary first
        assert "frontend" in services
        assert "currencyservice" in services

    def test_completeness_score_computed(self):
        legacy = self._basic_legacy(
            evidence_collection=_ev_payload([
                _ev_unit("checkoutservice", modality="metric"),
                _ev_unit("checkoutservice", modality="log", anomaly="error_spike"),
            ])
        )
        resp = build_log_agent_response(legacy, request_id="INC-007")
        # Coverage of single candidate by 2 modalities ≥ 0
        assert 0.0 <= resp.completeness_score <= 1.0


# ---------------------------------------------------------------------------
# build_topology_agent_response
# ---------------------------------------------------------------------------

class TestBuildTopologyAgentResponse:
    def test_candidates_list_format(self):
        legacy = {
            "candidates": [
                {"service": "checkoutservice", "confidence": 0.7},
                {"service": "currencyservice", "confidence": 0.5},
            ],
        }
        resp = build_topology_agent_response(legacy, request_id="INC-T1")
        assert resp.agent_name == "topology_agent"
        assert len(resp.candidates) == 2
        assert resp.candidates[0].service == "checkoutservice"

    def test_singleton_suspected_downstream_fallback(self):
        legacy = {"suspected_downstream": "frontend"}
        resp = build_topology_agent_response(legacy, request_id="INC-T2")
        assert len(resp.candidates) == 1
        assert resp.candidates[0].service == "frontend"

    def test_empty_legacy_yields_empty_candidates(self):
        legacy = {}
        resp = build_topology_agent_response(legacy, request_id="INC-T3")
        assert len(resp.candidates) == 0

    def test_recommended_actions_present(self):
        legacy = {"candidates": [{"service": "x", "confidence": 0.5}]}
        resp = build_topology_agent_response(legacy, request_id="INC-T4")
        assert any("log_evidence" in a for a in resp.recommended_next_actions)
        assert any("metric_evidence" in a for a in resp.recommended_next_actions)

    def test_completeness_score_zero(self):
        """Topology alone has no observational evidence — completeness 0."""
        legacy = {"candidates": [{"service": "x", "confidence": 0.5}]}
        resp = build_topology_agent_response(legacy, request_id="INC-T5")
        assert resp.completeness_score == 0.0


# ---------------------------------------------------------------------------
# build_rca_agent_response
# ---------------------------------------------------------------------------

class TestBuildRcaAgentResponse:
    def _basic_legacy(self, **overrides):
        base = {
            "incident_id": "INC-R1",
            "service": "frontend",
            "algorithm": "LLM-Synthesis",
            "overall_confidence": 0.8,
            "evidence_convergence": "all_three_agree",
            "root_cause_candidates": [
                {
                    "rank": 1,
                    "cause_service": "recommendationservice",
                    "confidence": 0.85,
                    "reasoning": "CPU saturation observed",
                },
            ],
            "propagation_path": ["recommendationservice", "frontend"],
        }
        base.update(overrides)
        return base

    def test_minimal_legacy(self):
        legacy = self._basic_legacy()
        resp = build_rca_agent_response(legacy, request_id="INC-R1")
        assert resp.agent_name == "rca_agent"
        assert len(resp.candidates) == 1
        c = resp.candidates[0]
        assert c.service == "recommendationservice"
        assert c.confidence == 0.85

    def test_inherits_evidence_from_upstream(self):
        log_legacy = {
            "summary": "x", "confidence": 0.7,
            "suspected_downstream": "recommendationservice",
            "anomalous_services": ["recommendationservice"],
            "evidence_collection": _ev_payload([
                _ev_unit("recommendationservice",
                         modality="metric", severity=0.9,
                         anomaly="resource_saturation"),
            ]),
        }
        upstream_resp = build_log_agent_response(log_legacy, request_id="INC-R2")

        rca_legacy = self._basic_legacy()
        resp = build_rca_agent_response(
            rca_legacy, request_id="INC-R2",
            upstream_log_response=upstream_resp,
        )
        # RCA candidate should pick up evidence IDs from upstream
        assert len(resp.evidence_collection) == 1
        assert len(resp.candidates[0].supporting_evidence) == 1

    def test_failure_mode_from_upstream_evidence(self):
        log_legacy = {
            "summary": "x", "confidence": 0.7,
            "suspected_downstream": "recommendationservice",
            "anomalous_services": [],
            "evidence_collection": _ev_payload([
                _ev_unit("recommendationservice",
                         modality="metric", severity=0.95,
                         anomaly="resource_saturation"),
            ]),
        }
        upstream = build_log_agent_response(log_legacy, request_id="INC-R3")
        rca_legacy = self._basic_legacy()
        resp = build_rca_agent_response(
            rca_legacy, request_id="INC-R3",
            upstream_log_response=upstream,
        )
        # failure mode inferred from resource_saturation evidence
        assert resp.candidates[0].failure_mode == "resource_exhaustion"

    def test_no_upstream_still_works(self):
        legacy = self._basic_legacy()
        resp = build_rca_agent_response(legacy, request_id="INC-R4")
        # Without upstream, evidence_collection is empty, supporting_evidence []
        assert len(resp.evidence_collection) == 0
        assert resp.candidates[0].supporting_evidence == []

    def test_topology_path_attached_to_candidate(self):
        legacy = self._basic_legacy()
        resp = build_rca_agent_response(legacy, request_id="INC-R5")
        # propagation_path from legacy → topology_path on candidate
        assert resp.candidates[0].topology_path == ["recommendationservice", "frontend"]


# ---------------------------------------------------------------------------
# attach_agent_response
# ---------------------------------------------------------------------------

class TestAttachAgentResponse:
    def test_attaches_serialised_response(self):
        legacy = {"summary": "x", "confidence": 0.7,
                  "suspected_downstream": "svc"}
        resp = build_log_agent_response(legacy, request_id="INC-A1")
        out = attach_agent_response(legacy, resp)
        assert "_agent_response" in out
        assert isinstance(out["_agent_response"], dict)
        # Original legacy fields untouched
        assert out["confidence"] == 0.7
        assert out["suspected_downstream"] == "svc"

    def test_roundtrip_via_pydantic(self):
        legacy = {"summary": "x", "confidence": 0.7,
                  "suspected_downstream": "svc"}
        resp = build_log_agent_response(legacy, request_id="INC-A2")
        attach_agent_response(legacy, resp)
        # Reconstitute the AgentResponse from the embedded dict
        embedded = legacy["_agent_response"]
        restored = AgentResponse.model_validate(embedded)
        assert restored.agent_name == "log_agent"
        assert restored.request_id == "INC-A2"

    def test_custom_key(self):
        legacy = {"x": 1}
        resp = AgentResponse(agent_name="log_agent", request_id="INC-A3")
        attach_agent_response(legacy, resp, key="my_custom_key")
        assert "my_custom_key" in legacy
        assert "_agent_response" not in legacy


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
