"""
Phase 4d-1 unit tests: Orchestrator contract parsing.

Verifies:
  1. _agent_response is extracted from log/topology/rca/verifier results
  2. agent_contracts dict is assembled when A2A_PARSE_CONTRACTS is on
  3. FinalRCAResult.agent_contracts is None when flag is off
  4. Missing _agent_response fields are handled gracefully
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from agents.orchestrator.models import (
    AgentResult,
    Evidence,
    FinalRCAResult,
    IncidentSummary,
    TimeRange,
    FinalVerdict,
    ImpactAnalysis,
    VerificationResult,
    EvidenceSummary,
    RootCauseCandidate,
)


# ---------------------------------------------------------------------------
# FinalRCAResult.agent_contracts — schema tests
# ---------------------------------------------------------------------------

class TestFinalRCAResultAgentContracts:
    def _minimal_result(self, **overrides):
        defaults = dict(
            incident_id="INC-001",
            incident_summary=IncidentSummary(
                service="frontend",
                symptom="latency degraded",
                time_range=TimeRange(
                    start="2026-04-24T13:00:00+09:00",
                    end="2026-04-24T13:05:00+09:00",
                ),
            ),
            root_cause_candidates=[],
            final_verdict=FinalVerdict(
                cause="", confidence=0.0, explanation="",
            ),
            impact_analysis=ImpactAnalysis(
                affected_services=[], related_services=[],
                propagation_path=[], blast_radius=[],
            ),
            verification=VerificationResult(
                verdict="accepted", final_confidence=0.5, notes=[],
            ),
            evidence_summary=EvidenceSummary(
                log_evidence=[], topology_evidence=[],
            ),
            agent_results=[],
        )
        defaults.update(overrides)
        return defaults

    def test_agent_contracts_field_optional(self):
        """FinalRCAResult must accept None for agent_contracts."""
        r = FinalRCAResult(**self._minimal_result())
        assert r.agent_contracts is None

    def test_agent_contracts_accepts_dict(self):
        contracts = {
            "log_agent": {"agent_name": "log_agent", "request_id": "x"},
            "rca_agent": {"agent_name": "rca_agent", "request_id": "x"},
        }
        r = FinalRCAResult(**self._minimal_result(agent_contracts=contracts))
        assert r.agent_contracts is not None
        assert "log_agent" in r.agent_contracts
        assert "rca_agent" in r.agent_contracts

    def test_adaptive_iterations_field_optional(self):
        r = FinalRCAResult(**self._minimal_result())
        assert r.adaptive_iterations is None

    def test_agent_contracts_json_roundtrip(self):
        contracts = {"log_agent": {"agent_name": "log_agent", "foo": 1}}
        r = FinalRCAResult(**self._minimal_result(agent_contracts=contracts))
        js = r.model_dump_json()
        restored = FinalRCAResult.model_validate_json(js)
        assert restored.agent_contracts == contracts


# ---------------------------------------------------------------------------
# Orchestrator's contract extraction logic
# ---------------------------------------------------------------------------
# We cannot unit-test the full analyze_incident without mocking the A2A
# client. Instead we exercise the _parse_agent_result and _parse_verification
# helpers directly to ensure _agent_response flows through.

from agents.orchestrator.service import OrchestratorService


def _fake_a2a_response(data: dict) -> dict:
    """Wrap data in the A2A task artifact envelope."""
    return {
        "result": {
            "task": {
                "artifacts": [
                    {
                        "parts": [
                            {"data": data},
                        ]
                    }
                ]
            }
        }
    }


class TestParseAgentResultExtractsContract:
    def _service(self):
        return OrchestratorService(
            log_agent_url="http://log",
            topology_agent_url="http://topo",
            rca_agent_url="http://rca",
            verifier_agent_url="http://ver",
        )

    def test_agent_response_extracted_into_metadata(self):
        svc = self._service()
        fake_contract = {
            "agent_name": "log_agent",
            "request_id": "INC-X",
            "candidates": [],
            "completeness_score": 0.7,
        }
        raw = _fake_a2a_response({
            "summary": "test",
            "confidence": 0.8,
            "evidence": [],
            "_agent_response": fake_contract,
        })
        result = svc._parse_agent_result(raw, "log_agent")
        assert result.metadata.get("_agent_response") == fake_contract

    def test_evidence_collection_extracted_into_metadata(self):
        svc = self._service()
        ev = {"units": [], "count": 0}
        raw = _fake_a2a_response({
            "summary": "test",
            "confidence": 0.8,
            "evidence_collection": ev,
        })
        result = svc._parse_agent_result(raw, "log_agent")
        assert result.metadata.get("evidence_collection") == ev

    def test_no_agent_response_gives_no_metadata_key(self):
        svc = self._service()
        raw = _fake_a2a_response({
            "summary": "test",
            "confidence": 0.8,
        })
        result = svc._parse_agent_result(raw, "log_agent")
        assert "_agent_response" not in result.metadata


class TestParseVerificationExtractsContract:
    def _service(self):
        return OrchestratorService(
            log_agent_url="http://log",
            topology_agent_url="http://topo",
            rca_agent_url="http://rca",
            verifier_agent_url="http://ver",
        )

    def test_verification_includes_agent_response(self):
        svc = self._service()
        contract = {"agent_name": "verifier_agent", "request_id": "X"}
        raw = _fake_a2a_response({
            "verdict": "accepted",
            "verification_notes": [],
            "revised_root_cause_candidates": [],
            "final_confidence": 0.9,
            "explanation": "ok",
            "_agent_response": contract,
        })
        result = svc._parse_verification(raw)
        assert result.get("_agent_response") == contract

    def test_no_artifact_fallback_has_no_agent_response(self):
        svc = self._service()
        # Missing artifact structure → fallback dict returned
        raw = {}
        result = svc._parse_verification(raw)
        assert "_agent_response" not in result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
