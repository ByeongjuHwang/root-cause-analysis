"""
Phase 4d-2 unit tests: Orchestrator adaptive execution.

Tests:
  1. _extract_completeness_score from verification dict
  2. _extract_focus_services from verification dict
  3. Full adaptive loop via mocked A2A client
     - Stops when completeness is sufficient
     - Iterates when completeness is low
     - Respects ADAPTIVE_MAX_ITERATIONS
     - No-op when feature flags off
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from agents.orchestrator.service import OrchestratorService
from agents.orchestrator.models import (
    AgentResult,
    IncidentRequest,
    TimeRange,
)


def _service():
    return OrchestratorService(
        log_agent_url="http://log",
        topology_agent_url="http://topo",
        rca_agent_url="http://rca",
        verifier_agent_url="http://ver",
    )


def _verification(completeness=None, candidates=None):
    """Build a minimal verification dict with an optional AgentResponse."""
    out = {
        "verdict": "accepted",
        "verification_notes": [],
        "revised_root_cause_candidates": [],
        "final_confidence": 0.8,
        "explanation": "ok",
    }
    if completeness is not None or candidates is not None:
        out["_agent_response"] = {
            "agent_name": "verifier_agent",
            "request_id": "INC-X",
            "candidates": candidates or [],
            "completeness_score": completeness,
        }
    return out


def _fake_a2a_response(data: dict) -> dict:
    return {
        "result": {
            "task": {
                "artifacts": [
                    {"parts": [{"data": data}]}
                ]
            }
        }
    }


# ---------------------------------------------------------------------------
# _extract_completeness_score
# ---------------------------------------------------------------------------

class TestExtractCompletenessScore:
    def test_none_when_no_contract(self):
        svc = _service()
        ver = {"verdict": "accepted"}
        assert svc._extract_completeness_score(ver) is None

    def test_none_when_score_missing(self):
        svc = _service()
        ver = {"_agent_response": {"agent_name": "verifier_agent"}}
        assert svc._extract_completeness_score(ver) is None

    def test_reads_float(self):
        svc = _service()
        ver = _verification(completeness=0.72)
        assert svc._extract_completeness_score(ver) == 0.72

    def test_converts_int_to_float(self):
        svc = _service()
        ver = _verification(completeness=1)
        score = svc._extract_completeness_score(ver)
        assert isinstance(score, float)
        assert score == 1.0

    def test_handles_invalid_type(self):
        svc = _service()
        ver = {"_agent_response": {"completeness_score": "nope"}}
        assert svc._extract_completeness_score(ver) is None


# ---------------------------------------------------------------------------
# _extract_focus_services
# ---------------------------------------------------------------------------

class TestExtractFocusServices:
    def test_empty_when_no_contract(self):
        svc = _service()
        ver = {"verdict": "accepted"}
        assert svc._extract_focus_services(ver) == []

    def test_empty_when_no_candidates(self):
        svc = _service()
        ver = _verification(completeness=0.3, candidates=[])
        assert svc._extract_focus_services(ver) == []

    def test_picks_services_with_missing_evidence(self):
        svc = _service()
        ver = _verification(completeness=0.3, candidates=[
            {"service": "svc_a", "missing_evidence": ["log_evidence_units"]},
            {"service": "svc_b", "missing_evidence": []},  # no missing → skipped
            {"service": "svc_c", "missing_evidence": ["trace_evidence_units"]},
        ])
        result = svc._extract_focus_services(ver, top_k=5)
        assert result == ["svc_a", "svc_c"]

    def test_respects_top_k(self):
        svc = _service()
        cands = [
            {"service": f"svc_{i}", "missing_evidence": ["x"]}
            for i in range(10)
        ]
        ver = _verification(completeness=0.3, candidates=cands)
        result = svc._extract_focus_services(ver, top_k=3)
        assert len(result) == 3
        assert result == ["svc_0", "svc_1", "svc_2"]

    def test_dedups(self):
        svc = _service()
        ver = _verification(completeness=0.3, candidates=[
            {"service": "dup", "missing_evidence": ["x"]},
            {"service": "dup", "missing_evidence": ["y"]},
            {"service": "other", "missing_evidence": ["z"]},
        ])
        result = svc._extract_focus_services(ver)
        assert result == ["dup", "other"]


# ---------------------------------------------------------------------------
# Full adaptive loop — mocked A2A client
# ---------------------------------------------------------------------------

class _MockA2AClient:
    """Records every send_message call and returns pre-programmed responses."""

    def __init__(self, responses_queue):
        self.responses = list(responses_queue)
        self.calls = []  # (url, metadata)

    async def send_message(self, agent_base_url, text, metadata=None):
        self.calls.append({"url": agent_base_url, "metadata": metadata or {}})
        if not self.responses:
            raise RuntimeError(f"MockA2AClient: no more responses queued (url={agent_base_url})")
        return self.responses.pop(0)


def _incident():
    return IncidentRequest(
        incident_id="INC-TEST",
        service="frontend",
        time_range=TimeRange(
            start="2026-04-24T13:00:00+09:00",
            end="2026-04-24T13:05:00+09:00",
        ),
        symptom="latency",
    )


def _log_result_fixture(focus_marker: str = "initial"):
    """Create an AgentResult simulating a log_agent response."""
    return AgentResult(
        agent="log_agent",
        summary=f"log analysis — {focus_marker}",
        confidence=0.7,
        evidence=[],
        metadata={},
    )


def _topology_result_fixture():
    return AgentResult(
        agent="topology_agent",
        summary="topology",
        confidence=0.6,
        evidence=[],
        metadata={"propagation_path": ["frontend"], "blast_radius": []},
    )


class TestAdaptiveLoop:
    @pytest.mark.asyncio
    async def test_no_iteration_when_score_sufficient(self):
        """completeness_score >= threshold → stop immediately."""
        svc = _service()
        svc.a2a_client = _MockA2AClient([])  # no calls expected

        ver = _verification(completeness=0.9)  # above 0.5 default
        os.environ["ADAPTIVE_THRESHOLD"] = "0.5"
        os.environ["ADAPTIVE_MAX_ITERATIONS"] = "3"

        try:
            log_result, rca_result, verification, iters = (
                await svc._run_adaptive_iterations(
                    incident=_incident(),
                    log_result=_log_result_fixture(),
                    topology_result=_topology_result_fixture(),
                    rca_result={"root_cause_candidates": []},
                    verification=ver,
                    agent_errors=[],
                )
            )
            assert len(iters) == 1
            assert iters[0]["action"] == "stop"
            assert iters[0]["reason"] == "completeness_sufficient"
            assert len(svc.a2a_client.calls) == 0
        finally:
            os.environ.pop("ADAPTIVE_THRESHOLD", None)
            os.environ.pop("ADAPTIVE_MAX_ITERATIONS", None)

    @pytest.mark.asyncio
    async def test_stops_when_no_contract(self):
        """If completeness unreadable → stop immediately."""
        svc = _service()
        svc.a2a_client = _MockA2AClient([])

        ver = {"verdict": "accepted"}  # no _agent_response

        log_result, rca_result, verification, iters = (
            await svc._run_adaptive_iterations(
                incident=_incident(),
                log_result=_log_result_fixture(),
                topology_result=_topology_result_fixture(),
                rca_result={"root_cause_candidates": []},
                verification=ver,
                agent_errors=[],
            )
        )
        assert len(iters) == 1
        assert iters[0]["reason"] == "no_contract_available"
        assert iters[0]["action"] == "stop"

    @pytest.mark.asyncio
    async def test_stops_when_no_focus_derivable(self):
        """Score is low but no candidates have missing_evidence → stop."""
        svc = _service()
        svc.a2a_client = _MockA2AClient([])

        ver = _verification(completeness=0.2, candidates=[
            {"service": "svc", "missing_evidence": []},
        ])
        os.environ["ADAPTIVE_THRESHOLD"] = "0.5"

        try:
            _, _, _, iters = await svc._run_adaptive_iterations(
                incident=_incident(),
                log_result=_log_result_fixture(),
                topology_result=_topology_result_fixture(),
                rca_result={"root_cause_candidates": []},
                verification=ver,
                agent_errors=[],
            )
            assert len(iters) == 1
            assert iters[0]["reason"] == "no_focus_services_derivable"
        finally:
            os.environ.pop("ADAPTIVE_THRESHOLD", None)

    @pytest.mark.asyncio
    async def test_iterates_and_passes_focus(self):
        """Low score → re-invoke log_agent with focus_services metadata."""
        svc = _service()

        # Prepare: 3 mock responses for Log → RCA → Verifier
        # 2nd iteration: completeness improves to 0.8, stops.
        log_response_2 = _fake_a2a_response({
            "summary": "improved log analysis",
            "confidence": 0.85,
            "evidence": [],
        })
        rca_response_2 = _fake_a2a_response({
            "summary": "improved rca",
            "confidence": 0.9,
            "root_cause_candidates": [],
        })
        verifier_response_2 = _fake_a2a_response({
            "verdict": "accepted",
            "verification_notes": [],
            "revised_root_cause_candidates": [],
            "final_confidence": 0.9,
            "explanation": "now-ok",
            "_agent_response": {
                "agent_name": "verifier_agent",
                "request_id": "INC-X",
                "candidates": [],
                "completeness_score": 0.8,
            },
        })

        svc.a2a_client = _MockA2AClient([
            log_response_2, rca_response_2, verifier_response_2,
        ])

        initial_ver = _verification(
            completeness=0.3,
            candidates=[{"service": "svc_x", "missing_evidence": ["log"]}],
        )

        os.environ["ADAPTIVE_THRESHOLD"] = "0.5"
        os.environ["ADAPTIVE_MAX_ITERATIONS"] = "3"

        try:
            log_result, rca_result, verification, iters = (
                await svc._run_adaptive_iterations(
                    incident=_incident(),
                    log_result=_log_result_fixture(),
                    topology_result=_topology_result_fixture(),
                    rca_result={"root_cause_candidates": []},
                    verification=initial_ver,
                    agent_errors=[],
                )
            )

            # Should have 2 iter records: (1) re-invoke, (2) sufficient
            assert len(iters) == 2
            assert iters[0]["action"] == "re_invoke_log_agent"
            assert iters[0]["focus_services"] == ["svc_x"]
            assert iters[0]["new_completeness_score"] == 0.8
            assert iters[1]["reason"] == "completeness_sufficient"

            # Verify focus_services was included in log_agent call metadata
            log_call = svc.a2a_client.calls[0]
            assert log_call["url"] == "http://log"
            assert log_call["metadata"].get("focus_services") == ["svc_x"]
        finally:
            os.environ.pop("ADAPTIVE_THRESHOLD", None)
            os.environ.pop("ADAPTIVE_MAX_ITERATIONS", None)

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self):
        """Even if score stays low, max iterations caps the loop."""
        svc = _service()

        # Build 2 iterations' worth of "still bad" responses
        def _bad_triplet():
            return [
                _fake_a2a_response({
                    "summary": "still bad",
                    "confidence": 0.5, "evidence": [],
                }),
                _fake_a2a_response({
                    "summary": "still bad",
                    "confidence": 0.5,
                    "root_cause_candidates": [],
                }),
                _fake_a2a_response({
                    "verdict": "rejected",
                    "verification_notes": [],
                    "revised_root_cause_candidates": [],
                    "final_confidence": 0.3,
                    "explanation": "still-bad",
                    "_agent_response": {
                        "agent_name": "verifier_agent",
                        "request_id": "INC-X",
                        "candidates": [
                            {"service": "svc_x", "missing_evidence": ["log"]}
                        ],
                        "completeness_score": 0.2,
                    },
                }),
            ]

        # Max iter = 2 → should do 2 re-invocations, total 6 mock calls
        svc.a2a_client = _MockA2AClient(_bad_triplet() + _bad_triplet())

        initial_ver = _verification(
            completeness=0.2,
            candidates=[{"service": "svc_x", "missing_evidence": ["log"]}],
        )

        os.environ["ADAPTIVE_THRESHOLD"] = "0.5"
        os.environ["ADAPTIVE_MAX_ITERATIONS"] = "2"

        try:
            _, _, _, iters = await svc._run_adaptive_iterations(
                incident=_incident(),
                log_result=_log_result_fixture(),
                topology_result=_topology_result_fixture(),
                rca_result={"root_cause_candidates": []},
                verification=initial_ver,
                agent_errors=[],
            )

            # Should have done 2 re-invocations (iter_num 1 and 2)
            reinvokes = [i for i in iters if i.get("action") == "re_invoke_log_agent"]
            assert len(reinvokes) == 2
            # 6 calls total (2 iterations × 3 agents)
            assert len(svc.a2a_client.calls) == 6
        finally:
            os.environ.pop("ADAPTIVE_THRESHOLD", None)
            os.environ.pop("ADAPTIVE_MAX_ITERATIONS", None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
