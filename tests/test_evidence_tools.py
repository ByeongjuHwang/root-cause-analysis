"""
Phase 2 integration tests: mcp_servers/observability_mcp/app/evidence_tools.py

실행:
    python -m pytest tests/test_evidence_tools.py -v

실제 MCP server를 띄우지 않고, repository 함수를 mock하여 evidence_tools의
로직만 격리 테스트한다. 실전 smoke test는 run_rcaeval.py로 수행.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from common.evidence import EvidenceCollection, TimeRange
from mcp_servers.observability_mcp.app.evidence_tools import (
    build_evidence_collection,
    build_log_evidence,
    build_metric_evidence,
    build_topology_evidence,
    get_evidence_collection_payload,
    get_log_evidence_payload,
    get_metric_evidence_payload,
    get_topology_evidence_payload,
)


# ---------------------------------------------------------------------------
# Mock data builders
# ---------------------------------------------------------------------------

WINDOW_START = "2026-04-23T13:00:00+09:00"
WINDOW_END   = "2026-04-23T13:05:00+09:00"

BASELINE_START = "2026-04-23T12:55:00+09:00"
BASELINE_END   = "2026-04-23T13:00:00+09:00"

INCIDENT_START = "2026-04-23T13:03:00+09:00"
INCIDENT_END   = "2026-04-23T13:05:00+09:00"


def _mock_stats_result():
    """Mimics repo_get_service_statistics() return shape."""
    return {
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "mode": "dual",
        "total_log_count": 1000,
        "services": {
            "checkoutservice": {
                "baseline_count": 100, "incident_count": 380,
                "volume_delta": 2.17, "error_ratio": 0.05,
                "timeout_hits": 12, "retry_hits": 3, "reset_hits": 1,
                "has_activity": True,
            },
            "frontend": {
                "baseline_count": 200, "incident_count": 220,
                "volume_delta": 0.1, "error_ratio": 0.0,
                "timeout_hits": 0, "retry_hits": 0, "reset_hits": 0,
                "has_activity": True,
            },
            "idle_service": {
                "baseline_count": 0, "incident_count": 0,
                "volume_delta": 0.0, "error_ratio": 0.0,
                "timeout_hits": 0, "retry_hits": 0, "reset_hits": 0,
                "has_activity": False,
            },
        },
    }


def _mock_log_records(service: str, **kwargs):
    """Mimics repo_search_logs() signature including start/end/log_file kwargs."""
    # Only checkoutservice gets upstream errors
    if service != "checkoutservice":
        return []
    return [
        SimpleNamespace(
            model_dump=lambda: {
                "timestamp": "2026-04-23T13:03:10+09:00",
                "level": "ERROR", "message": "upstream connection reset",
                "upstream": "redis", "status_code": 503,
            }
        ),
        SimpleNamespace(
            model_dump=lambda: {
                "timestamp": "2026-04-23T13:03:15+09:00",
                "level": "ERROR", "message": "upstream timeout",
                "upstream": "redis", "status_code": 504,
            }
        ),
    ]


def _mock_metric_result():
    """Mimics repo_get_all_service_metric_summaries() return shape."""
    return {
        "window": {"start": INCIDENT_START, "end": INCIDENT_END},
        "services": {
            "checkoutservice": {
                "metric": {"has_data": True, "cpu_spike_zscore": 289.6,
                           "cpu_max": 5.16, "mem_jump_ratio": 0.40,
                           "mem_max": 0.78},
                "latency": {"has_data": True, "p95_delta_ms": 26.5,
                            "p99_delta_ms": -45.0, "p50_ms": 10, "p95_ms": 80},
                "retry_timeout": {"has_data": False},
            },
            "frontend": {
                "metric": {"has_data": True, "cpu_spike_zscore": 1.2,
                           "cpu_max": 0.3, "mem_jump_ratio": 0.0},
                "latency": {"has_data": True, "p95_delta_ms": 349.5,
                            "p99_delta_ms": 512.0, "p50_ms": 30, "p95_ms": 450},
                "retry_timeout": {"has_data": False},
            },
            "quiet_svc": {
                "metric": {"has_data": False},
                "latency": {"has_data": False},
                "retry_timeout": {"has_data": False},
            },
        },
    }


# ---------------------------------------------------------------------------
# build_log_evidence
# ---------------------------------------------------------------------------

class TestBuildLogEvidence:
    def test_basic_log_evidence_generation(self):
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_service_statistics",
            return_value=_mock_stats_result(),
        ), patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_search_logs",
            side_effect=_mock_log_records,
        ):
            col = build_log_evidence(
                start=WINDOW_START, end=WINDOW_END,
                baseline_start=BASELINE_START, baseline_end=BASELINE_END,
                incident_start=INCIDENT_START, incident_end=INCIDENT_END,
            )

        assert isinstance(col, EvidenceCollection)
        # checkoutservice has: volume_shift + error_spike + keyword_distress + dependency_failure
        # frontend: none (no anomalies)
        # idle_service: skipped (has_activity=False)
        services = col.services_covered()
        assert "checkoutservice" in services
        # dependency_failure is logged against checkoutservice but also lists upstream
        assert any("redis" in u.services for u in col.by_service("checkoutservice"))

    def test_focus_services_filters(self):
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_service_statistics",
            return_value=_mock_stats_result(),
        ), patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_search_logs",
            side_effect=_mock_log_records,
        ):
            col = build_log_evidence(
                start=WINDOW_START, end=WINDOW_END,
                focus_services=["frontend"],
            )
        # frontend has no anomalies in mock → empty collection
        # (and checkoutservice filtered out)
        assert "checkoutservice" not in col.services_covered()

    def test_returns_empty_collection_gracefully(self):
        empty_stats = {"services": {}, "mode": "dual", "window": {}}
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_service_statistics",
            return_value=empty_stats,
        ):
            col = build_log_evidence(start=WINDOW_START, end=WINDOW_END)
        assert len(col) == 0

    def test_dependency_failure_resilient_to_exceptions(self):
        """If repo_search_logs raises, we should still return whatever
        log stats evidence we got."""
        def _raise(*args, **kwargs):
            raise RuntimeError("simulated IO failure")

        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_service_statistics",
            return_value=_mock_stats_result(),
        ), patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_search_logs",
            side_effect=_raise,
        ):
            col = build_log_evidence(start=WINDOW_START, end=WINDOW_END)
        # Stats-based evidence still produced, only dep_failure slice missing
        assert len(col) > 0


# ---------------------------------------------------------------------------
# build_metric_evidence
# ---------------------------------------------------------------------------

class TestBuildMetricEvidence:
    def test_metric_evidence_generation(self):
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_all_metric",
            return_value=_mock_metric_result(),
        ):
            col = build_metric_evidence(
                start=WINDOW_START, end=WINDOW_END,
                baseline_start=BASELINE_START, baseline_end=BASELINE_END,
            )
        services = col.services_covered()
        # checkoutservice: resource_saturation (CPU z=289)
        # frontend:        latency_degradation (p95_delta=349)
        # quiet_svc:       nothing (has_data all False)
        assert "checkoutservice" in services
        assert "frontend" in services
        assert "quiet_svc" not in services

    def test_metric_severity_high_for_cpu_fault(self):
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_all_metric",
            return_value=_mock_metric_result(),
        ):
            col = build_metric_evidence(start=WINDOW_START, end=WINDOW_END)
        checkout_evs = col.by_service("checkoutservice")
        sat_evs = [e for e in checkout_evs if e.anomaly_type == "resource_saturation"]
        assert len(sat_evs) == 1
        # z=289 maps to ~1.0
        assert sat_evs[0].severity > 0.95

    def test_metric_repo_exception_returns_empty(self):
        def _raise(*args, **kwargs):
            raise RuntimeError("no metric file")
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_all_metric",
            side_effect=_raise,
        ):
            col = build_metric_evidence(start=WINDOW_START, end=WINDOW_END)
        assert len(col) == 0


# ---------------------------------------------------------------------------
# build_topology_evidence
# ---------------------------------------------------------------------------

class TestBuildTopologyEvidence:
    def test_basic_topology_evidence(self):
        col = build_topology_evidence(
            symptom_service="api-gateway",
            candidate_services=["auth-service", "user-db"],
            path=["api-gateway", "auth-service", "user-db"],
            start=WINDOW_START, end=WINDOW_END,
        )
        assert len(col) == 2
        # severity <= 0.5 for all topology evidence
        for u in col.units:
            assert u.severity <= 0.5

    def test_candidate_not_in_path_skipped(self):
        col = build_topology_evidence(
            symptom_service="api-gateway",
            candidate_services=["auth-service", "unrelated-svc"],
            path=["api-gateway", "auth-service", "user-db"],
            start=WINDOW_START, end=WINDOW_END,
        )
        # only auth-service produces evidence
        assert len(col) == 1


# ---------------------------------------------------------------------------
# build_evidence_collection (unified)
# ---------------------------------------------------------------------------

class TestBuildEvidenceCollection:
    def test_unified_all_modalities(self):
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_service_statistics",
            return_value=_mock_stats_result(),
        ), patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_search_logs",
            side_effect=_mock_log_records,
        ), patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_all_metric",
            return_value=_mock_metric_result(),
        ):
            col = build_evidence_collection(
                start=WINDOW_START, end=WINDOW_END,
                baseline_start=BASELINE_START, baseline_end=BASELINE_END,
                symptom_service="frontend",
                topology_path=["frontend", "checkoutservice", "redis"],
                candidate_services=["checkoutservice", "redis"],
            )
        mods = col.modalities_present()
        assert "log" in mods
        assert "metric" in mods
        assert "topology" in mods

    def test_topology_skipped_when_args_missing(self):
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_service_statistics",
            return_value=_mock_stats_result(),
        ), patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_search_logs",
            side_effect=_mock_log_records,
        ), patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_all_metric",
            return_value=_mock_metric_result(),
        ):
            col = build_evidence_collection(
                start=WINDOW_START, end=WINDOW_END,
                # no topology_path / symptom_service
            )
        assert "topology" not in col.modalities_present()


# ---------------------------------------------------------------------------
# Payload wrappers — the shape MCP clients receive
# ---------------------------------------------------------------------------

class TestPayloadShape:
    def test_log_payload_shape(self):
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_service_statistics",
            return_value=_mock_stats_result(),
        ), patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_search_logs",
            side_effect=_mock_log_records,
        ):
            payload = get_log_evidence_payload(
                start=WINDOW_START, end=WINDOW_END,
            )
        assert set(payload.keys()) == {
            "count", "services_covered", "modalities_present", "units"
        }
        assert isinstance(payload["units"], list)
        # Each unit must be JSON-serialisable dict, not a pydantic model
        for u in payload["units"]:
            assert isinstance(u, dict)
            assert "evidence_id" in u
            assert "modality" in u
            assert "severity" in u

    def test_payload_roundtrip_via_evidence_collection(self):
        """Ensure the payload can be loaded back into EvidenceCollection."""
        import json
        from common.evidence import EvidenceUnit

        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_all_metric",
            return_value=_mock_metric_result(),
        ):
            payload = get_metric_evidence_payload(start=WINDOW_START, end=WINDOW_END)

        # Mimic what the client (Phase 3 Agent) will do
        restored = [EvidenceUnit.model_validate(u) for u in payload["units"]]
        assert len(restored) == payload["count"]
        for u in restored:
            assert u.modality == "metric"

    def test_empty_payload_is_serialisable(self):
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.repo_get_service_statistics",
            return_value={"services": {}, "mode": "dual", "window": {}},
        ):
            payload = get_log_evidence_payload(start=WINDOW_START, end=WINDOW_END)
        assert payload["count"] == 0
        assert payload["units"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
