"""
Phase 1 unit tests: common/a2a_contract.py

실행:
    python -m pytest tests/test_a2a_contract.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from pydantic import ValidationError

from common.a2a_contract import (
    AgentRequest,
    AgentResponse,
    Candidate,
    ConsistencyChecks,
    make_legacy_response,
)
from common.evidence import EvidenceCollection, EvidenceUnit, TimeRange


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

class TestCandidate:
    def test_minimal_construction(self):
        c = Candidate(service="checkoutservice", confidence=0.8)
        assert c.service == "checkoutservice"
        assert c.confidence == 0.8
        assert c.supporting_evidence == []
        assert c.is_evidence_backed() is False

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Candidate(service="x", confidence=1.5)
        with pytest.raises(ValidationError):
            Candidate(service="x", confidence=-0.1)

    def test_service_must_be_nonempty(self):
        with pytest.raises(ValidationError):
            Candidate(service="", confidence=0.8)
        with pytest.raises(ValidationError):
            Candidate(service="   ", confidence=0.8)

    def test_service_stripped(self):
        c = Candidate(service="  checkoutservice  ", confidence=0.8)
        assert c.service == "checkoutservice"

    def test_is_evidence_backed(self):
        c1 = Candidate(service="x", confidence=0.8)
        c2 = Candidate(
            service="x", confidence=0.8,
            supporting_evidence=["ev_log_00001"],
        )
        assert not c1.is_evidence_backed()
        assert c2.is_evidence_backed()

    def test_has_missing_evidence(self):
        c = Candidate(
            service="x", confidence=0.5,
            missing_evidence=["trace_data_for_db"],
        )
        assert c.has_missing_evidence()

    def test_json_roundtrip(self):
        c = Candidate(
            service="user-db",
            confidence=0.92,
            supporting_evidence=["ev_log_aaa", "ev_metric_bbb"],
            assumptions=["baseline_window=480s"],
            missing_evidence=["trace_data"],
            reasoning="CPU z-score extreme + log volume spike",
        )
        json_str = c.model_dump_json()
        restored = Candidate.model_validate_json(json_str)
        assert restored.service == c.service
        assert restored.supporting_evidence == c.supporting_evidence

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            Candidate(service="x", confidence=0.5, unexpected="nope")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ConsistencyChecks
# ---------------------------------------------------------------------------

class TestConsistencyChecks:
    def test_all_none_does_not_pass(self):
        cc = ConsistencyChecks()
        assert cc.passed() is False

    def test_all_true_passes(self):
        cc = ConsistencyChecks(
            temporal=True, topological=True, modality=True, counter_evidence=True,
        )
        assert cc.passed() is True

    def test_any_false_fails(self):
        cc = ConsistencyChecks(temporal=True, topological=False, modality=True)
        assert cc.passed() is False
        assert cc.failed_dimensions() == ["topological"]

    def test_none_means_skipped(self):
        cc = ConsistencyChecks(temporal=True, topological=True)
        # modality and counter_evidence not checked
        assert cc.passed() is True  # performed ones all pass
        assert set(cc.skipped_dimensions()) == {"modality", "counter_evidence"}

    def test_mixed_reports_correctly(self):
        cc = ConsistencyChecks(
            temporal=True, topological=False,
            modality=None, counter_evidence=False,
        )
        assert cc.passed() is False
        assert set(cc.failed_dimensions()) == {"topological", "counter_evidence"}
        assert cc.skipped_dimensions() == ["modality"]


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------

def _make_evidence(service, evid="ev_log_abc12345"):
    return EvidenceUnit(
        evidence_id=evid,
        modality="log",
        time_range=TimeRange(
            start="2026-04-23T13:00:00+09:00",
            end="2026-04-23T13:05:00+09:00",
        ),
        services=[service],
        anomaly_type="error_spike",
        severity=0.8,
        observation={},
        source="test",
    )


class TestAgentResponse:
    def test_minimal_response(self):
        resp = AgentResponse(
            agent_name="log_agent",
            request_id="INC-001",
        )
        assert resp.top_candidate() is None
        assert resp.needs_more_evidence() is True  # score is 0.0 default

    def test_top_candidate_by_confidence(self):
        resp = AgentResponse(
            agent_name="rca_agent",
            request_id="INC-001",
            candidates=[
                Candidate(service="a", confidence=0.5),
                Candidate(service="b", confidence=0.9),
                Candidate(service="c", confidence=0.7),
            ],
        )
        top = resp.top_candidate()
        assert top is not None
        assert top.service == "b"

    def test_evidence_backed_filtering(self):
        resp = AgentResponse(
            agent_name="log_agent",
            request_id="INC-001",
            candidates=[
                Candidate(service="a", confidence=0.5, supporting_evidence=["e1"]),
                Candidate(service="b", confidence=0.9),  # no evidence
                Candidate(service="c", confidence=0.7, supporting_evidence=["e2"]),
            ],
        )
        backed = resp.evidence_backed_candidates()
        assert [c.service for c in backed] == ["a", "c"]

    def test_needs_more_evidence(self):
        resp_low = AgentResponse(
            agent_name="log_agent", request_id="INC-001",
            completeness_score=0.3,
        )
        resp_high = AgentResponse(
            agent_name="log_agent", request_id="INC-001",
            completeness_score=0.85,
        )
        assert resp_low.needs_more_evidence(0.7)
        assert not resp_high.needs_more_evidence(0.7)

    def test_all_missing_evidence_dedup(self):
        resp = AgentResponse(
            agent_name="log_agent", request_id="INC-001",
            candidates=[
                Candidate(service="a", confidence=0.5,
                          missing_evidence=["trace", "metric"]),
                Candidate(service="b", confidence=0.7,
                          missing_evidence=["trace", "topology"]),
            ],
        )
        miss = resp.all_missing_evidence()
        # Order preserved, duplicates removed
        assert miss == ["trace", "metric", "topology"]

    def test_confidence_gap(self):
        resp = AgentResponse(
            agent_name="rca_agent", request_id="INC-001",
            candidates=[
                Candidate(service="a", confidence=0.9),
                Candidate(service="b", confidence=0.85),
                Candidate(service="c", confidence=0.5),
            ],
        )
        gap = resp.confidence_gap()
        assert gap is not None
        assert abs(gap - 0.05) < 1e-6

    def test_confidence_gap_single_candidate_none(self):
        resp = AgentResponse(
            agent_name="rca_agent", request_id="INC-001",
            candidates=[Candidate(service="a", confidence=0.9)],
        )
        assert resp.confidence_gap() is None

    def test_validate_evidence_refs_finds_dangling(self):
        ev = _make_evidence("svc1", "ev_log_aaaa1111")
        col = EvidenceCollection(units=[ev])
        resp = AgentResponse(
            agent_name="rca_agent", request_id="INC-001",
            candidates=[
                Candidate(service="svc1", confidence=0.8,
                          supporting_evidence=["ev_log_aaaa1111", "ev_ghost"]),
            ],
            evidence_collection=col,
        )
        dangling = resp.validate_evidence_refs()
        assert dangling == ["ev_ghost"]

    def test_validate_evidence_refs_clean(self):
        ev = _make_evidence("svc1", "ev_log_aaaa1111")
        col = EvidenceCollection(units=[ev])
        resp = AgentResponse(
            agent_name="rca_agent", request_id="INC-001",
            candidates=[
                Candidate(service="svc1", confidence=0.8,
                          supporting_evidence=["ev_log_aaaa1111"]),
            ],
            evidence_collection=col,
        )
        assert resp.validate_evidence_refs() == []

    def test_completeness_score_bounds(self):
        with pytest.raises(ValidationError):
            AgentResponse(
                agent_name="log_agent",
                request_id="INC-001",
                completeness_score=1.5,
            )

    def test_json_roundtrip_with_evidence(self):
        ev = _make_evidence("svc1", "ev_log_aaaa1111")
        col = EvidenceCollection(units=[ev])
        resp = AgentResponse(
            agent_name="rca_agent", request_id="INC-001",
            candidates=[
                Candidate(service="svc1", confidence=0.8,
                          supporting_evidence=["ev_log_aaaa1111"]),
            ],
            evidence_collection=col,
            completeness_score=0.75,
            reasoning="strong log evidence",
        )
        payload = resp.model_dump_json()
        restored = AgentResponse.model_validate_json(payload)
        assert restored.top_candidate().service == "svc1"
        assert len(restored.evidence_collection) == 1

    def test_verifier_with_consistency_checks(self):
        cc = ConsistencyChecks(
            temporal=True, topological=True, modality=True, counter_evidence=False,
        )
        resp = AgentResponse(
            agent_name="verifier_agent", request_id="INC-001",
            consistency_checks=cc,
        )
        assert resp.consistency_checks.passed() is False

    def test_unknown_agent_name_rejected(self):
        with pytest.raises(ValidationError):
            AgentResponse(
                agent_name="unknown_agent",  # type: ignore[arg-type]
                request_id="INC-001",
            )


# ---------------------------------------------------------------------------
# Phase 4a new fields — FailureMode on Candidate, counter_evidence on CC
# ---------------------------------------------------------------------------

class TestFailureMode:
    def test_candidate_without_failure_mode_ok(self):
        """failure_mode is optional (backward compat)."""
        c = Candidate(service="svc", confidence=0.8)
        assert c.failure_mode is None

    def test_candidate_with_failure_mode(self):
        c = Candidate(
            service="svc", confidence=0.8,
            failure_mode="resource_exhaustion",
        )
        assert c.failure_mode == "resource_exhaustion"

    def test_invalid_failure_mode_rejected(self):
        with pytest.raises(ValidationError):
            Candidate(
                service="svc", confidence=0.8,
                failure_mode="nonexistent_mode",  # type: ignore[arg-type]
            )

    def test_all_declared_failure_modes_accepted(self):
        """Every Literal value should construct without error."""
        declared = [
            "resource_exhaustion", "cascading_failure", "network_partition",
            "dependency_timeout", "deadlock_or_saturation", "configuration_error",
            "data_corruption", "retry_storm", "noisy_neighbor",
            "partial_outage", "unknown",
        ]
        for mode in declared:
            c = Candidate(service="svc", confidence=0.5, failure_mode=mode)  # type: ignore[arg-type]
            assert c.failure_mode == mode

    def test_failure_mode_json_roundtrip(self):
        c = Candidate(
            service="db", confidence=0.9,
            failure_mode="resource_exhaustion",
            supporting_evidence=["ev_metric_abc"],
        )
        js = c.model_dump_json()
        restored = Candidate.model_validate_json(js)
        assert restored.failure_mode == "resource_exhaustion"


class TestCounterEvidenceDimension:
    def test_counter_evidence_field_present(self):
        """Phase 4a renamed causal -> counter_evidence."""
        cc = ConsistencyChecks(counter_evidence=True)
        assert cc.counter_evidence is True

    def test_counter_evidence_false_detected_as_failure(self):
        cc = ConsistencyChecks(
            temporal=True, topological=True, modality=True,
            counter_evidence=False,
        )
        assert cc.passed() is False
        assert "counter_evidence" in cc.failed_dimensions()

    def test_old_causal_name_rejected(self):
        """Rename must be enforced — old kwarg should fail."""
        with pytest.raises(ValidationError):
            ConsistencyChecks(causal=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# AgentRequest
# ---------------------------------------------------------------------------

class TestAgentRequest:
    def test_minimal_request(self):
        req = AgentRequest(agent_name="log_agent", request_id="INC-001")
        assert req.iteration == 0
        assert req.focus_services is None

    def test_with_focus(self):
        req = AgentRequest(
            agent_name="log_agent",
            request_id="INC-001",
            focus_services=["user-db"],
            iteration=1,
        )
        assert req.focus_services == ["user-db"]
        assert req.iteration == 1

    def test_roundtrip(self):
        req = AgentRequest(
            agent_name="rca_agent", request_id="INC-001",
            include_evidence_refs=["ev_a", "ev_b"],
        )
        js = req.model_dump_json()
        restored = AgentRequest.model_validate_json(js)
        assert restored.include_evidence_refs == ["ev_a", "ev_b"]


# ---------------------------------------------------------------------------
# make_legacy_response
# ---------------------------------------------------------------------------

class TestLegacyResponseWrapper:
    def test_wraps_minimal(self):
        resp = make_legacy_response(
            agent_name="log_agent", request_id="INC-001",
            predicted_service="user-db", confidence=0.7,
        )
        assert resp.top_candidate().service == "user-db"
        # 레거시 응답은 completeness_score가 0 (= 재호출 대상)
        assert resp.completeness_score == 0.0
        assert resp.needs_more_evidence(0.7) is True
        top = resp.top_candidate()
        assert top.missing_evidence == ["structured_evidence_units"]
        assert top.supporting_evidence == []


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
