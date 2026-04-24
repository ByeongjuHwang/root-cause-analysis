"""
Phase 1 unit tests: common/evidence_factory.py

실행:
    python -m pytest tests/test_evidence_factory.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from common.evidence import TimeRange
from common.evidence_factory import (
    build_collection_from_mcp_outputs,
    evidence_from_log_records,
    evidence_from_metric_summary,
    evidence_from_service_statistics,
    evidence_from_topology_path,
)


TR = TimeRange(
    start="2026-04-23T13:00:00+09:00",
    end="2026-04-23T13:05:00+09:00",
)


# ---------------------------------------------------------------------------
# evidence_from_service_statistics
# ---------------------------------------------------------------------------

class TestLogStatsFactory:
    def test_all_three_kinds(self):
        stats = {
            "baseline_count": 100, "incident_count": 380,
            "volume_delta": 2.17, "error_ratio": 0.05,
            "timeout_hits": 12, "retry_hits": 3, "reset_hits": 1,
        }
        units = evidence_from_service_statistics("checkoutservice", stats, TR)
        types = {u.anomaly_type for u in units}
        assert types == {"volume_shift", "error_spike", "keyword_distress"}

    def test_low_volume_ignores_volume_delta(self):
        """v6 regression에서 배운 것: 저용량 서비스의 volume_delta는 증거화하지 않는다."""
        stats = {
            "baseline_count": 5, "incident_count": 5,  # too small
            "volume_delta": 2.20, "error_ratio": 0.0,
            "timeout_hits": 0, "retry_hits": 0, "reset_hits": 0,
        }
        units = evidence_from_service_statistics("redis", stats, TR)
        assert len(units) == 0

    def test_empty_stats_no_evidence(self):
        stats = {
            "baseline_count": 100, "incident_count": 100,
            "volume_delta": 0.01, "error_ratio": 0.0,
            "timeout_hits": 0, "retry_hits": 0, "reset_hits": 0,
        }
        units = evidence_from_service_statistics("frontend", stats, TR)
        assert units == []

    def test_only_keyword_distress(self):
        stats = {
            "baseline_count": 200, "incident_count": 200,
            "volume_delta": 0.0, "error_ratio": 0.0,
            "timeout_hits": 15, "retry_hits": 0, "reset_hits": 0,
        }
        units = evidence_from_service_statistics("gateway", stats, TR)
        assert len(units) == 1
        assert units[0].anomaly_type == "keyword_distress"
        assert units[0].observation["keyword_hits"]["timeout"] == 15

    def test_severity_bounded(self):
        stats = {
            "baseline_count": 100, "incident_count": 10000,
            "volume_delta": 100.0, "error_ratio": 1.0,
            "timeout_hits": 9999, "retry_hits": 9999, "reset_hits": 9999,
        }
        units = evidence_from_service_statistics("storm", stats, TR)
        for u in units:
            assert 0.0 <= u.severity <= 1.0


# ---------------------------------------------------------------------------
# evidence_from_log_records
# ---------------------------------------------------------------------------

class TestLogRecordsFactory:
    def test_upstream_errors_produce_dependency_failure(self):
        records = [
            {"timestamp": "2026-04-23T13:02:00+09:00", "level": "ERROR",
             "message": "upstream timeout",
             "upstream": "user-db", "status_code": 504},
            {"timestamp": "2026-04-23T13:02:10+09:00", "level": "ERROR",
             "message": "connection refused",
             "upstream": "user-db", "status_code": 503},
        ]
        ev = evidence_from_log_records("auth-service", records, TR)
        assert ev is not None
        assert ev.anomaly_type == "dependency_failure"
        assert "auth-service" in ev.services
        assert "user-db" in ev.services
        assert ev.observation["upstream_error_count"] == 2

    def test_no_upstream_records_no_evidence(self):
        records = [
            {"timestamp": "2026-04-23T13:02:00+09:00", "level": "ERROR",
             "message": "internal error"},  # no upstream
        ]
        ev = evidence_from_log_records("auth-service", records, TR)
        assert ev is None

    def test_raw_samples_bounded(self):
        records = [
            {"timestamp": f"t{i}", "level": "ERROR",
             "message": "e" * 500,
             "upstream": "db", "status_code": 500}
            for i in range(10)
        ]
        ev = evidence_from_log_records("svc", records, TR, max_samples=3)
        assert ev is not None
        assert len(ev.raw_samples) == 3
        for s in ev.raw_samples:
            # 240 char cap enforced by EvidenceUnit validator
            assert len(s) <= 240


# ---------------------------------------------------------------------------
# evidence_from_metric_summary
# ---------------------------------------------------------------------------

class TestMetricFactory:
    def test_cpu_spike_creates_resource_saturation(self):
        summary = {
            "metric": {"has_data": True, "cpu_spike_zscore": 289.6,
                       "cpu_max": 5.16, "mem_jump_ratio": 0.0},
            "latency": {"has_data": False},
            "retry_timeout": {"has_data": False},
        }
        units = evidence_from_metric_summary("checkoutservice", summary, TR)
        assert len(units) == 1
        assert units[0].anomaly_type == "resource_saturation"
        assert units[0].severity > 0.9  # z=289 should map high

    def test_latency_spike_creates_latency_degradation(self):
        summary = {
            "metric": {"has_data": False},
            "latency": {"has_data": True, "p95_delta_ms": 349.5,
                        "p99_delta_ms": 512.0, "p50_ms": 20, "p95_ms": 500},
            "retry_timeout": {"has_data": False},
        }
        units = evidence_from_metric_summary("frontend", summary, TR)
        assert len(units) == 1
        assert units[0].anomaly_type == "latency_degradation"

    def test_packet_drops_create_network_degradation(self):
        summary = {
            "metric": {"has_data": False},
            "latency": {"has_data": False},
            "retry_timeout": {"has_data": True, "rx_drop_delta": 250,
                              "tx_drop_delta": 0, "error_delta": 0,
                              "sockets_max": 1000},
        }
        units = evidence_from_metric_summary("emailservice", summary, TR)
        assert len(units) == 1
        assert units[0].anomaly_type == "network_degradation"

    def test_noise_below_threshold_ignored(self):
        summary = {
            "metric": {"has_data": True, "cpu_spike_zscore": 0.5,
                       "cpu_max": 0.5, "mem_jump_ratio": 0.01},
            "latency": {"has_data": True, "p95_delta_ms": 5.0,
                        "p99_delta_ms": 10.0},
            "retry_timeout": {"has_data": True, "rx_drop_delta": 0,
                              "tx_drop_delta": 0, "error_delta": 0,
                              "sockets_max": 100},
        }
        units = evidence_from_metric_summary("quiet-svc", summary, TR)
        assert units == []

    def test_multiple_kinds_at_once(self):
        summary = {
            "metric": {"has_data": True, "cpu_spike_zscore": 150,
                       "cpu_max": 4.0, "mem_jump_ratio": 0.6},
            "latency": {"has_data": True, "p95_delta_ms": 400,
                        "p99_delta_ms": 600},
            "retry_timeout": {"has_data": True, "rx_drop_delta": 500,
                              "tx_drop_delta": 200, "error_delta": 30,
                              "sockets_max": 1500},
        }
        units = evidence_from_metric_summary("busy", summary, TR)
        types = {u.anomaly_type for u in units}
        assert types == {"resource_saturation", "latency_degradation", "network_degradation"}


# ---------------------------------------------------------------------------
# evidence_from_topology_path
# ---------------------------------------------------------------------------

class TestTopologyFactory:
    def test_direct_upstream(self):
        ev = evidence_from_topology_path(
            symptom_service="api-gateway",
            candidate_service="auth-service",
            path=["api-gateway", "auth-service", "user-db"],
            time_range=TR,
        )
        assert ev is not None
        assert ev.anomaly_type == "topology_proximity"
        assert ev.observation["path_distance"] == 1
        # severity bounded at 0.5 (v7 H4 principle)
        assert ev.severity <= 0.5

    def test_candidate_is_symptom_no_evidence(self):
        ev = evidence_from_topology_path(
            symptom_service="api-gateway",
            candidate_service="api-gateway",
            path=["api-gateway", "auth-service"],
            time_range=TR,
        )
        assert ev is None

    def test_candidate_not_in_path(self):
        ev = evidence_from_topology_path(
            symptom_service="api-gateway",
            candidate_service="unrelated-svc",
            path=["api-gateway", "auth-service"],
            time_range=TR,
        )
        assert ev is None

    def test_distant_has_lower_severity(self):
        ev_near = evidence_from_topology_path(
            "api-gateway", "auth-service",
            ["api-gateway", "auth-service", "user-db"], TR,
        )
        ev_far = evidence_from_topology_path(
            "api-gateway", "user-db",
            ["api-gateway", "auth-service", "user-db"], TR,
        )
        assert ev_near.severity > ev_far.severity


# ---------------------------------------------------------------------------
# build_collection_from_mcp_outputs — integration
# ---------------------------------------------------------------------------

class TestBuildCollection:
    def test_end_to_end(self):
        stats = {
            "checkoutservice": {
                "baseline_count": 100, "incident_count": 380,
                "volume_delta": 2.17, "error_ratio": 0.05,
                "timeout_hits": 12, "retry_hits": 3, "reset_hits": 1,
            },
            "frontend": {
                "baseline_count": 200, "incident_count": 220,
                "volume_delta": 0.1, "error_ratio": 0.0,
                "timeout_hits": 0, "retry_hits": 0, "reset_hits": 0,
            },
        }
        metrics = {
            "services": {
                "checkoutservice": {
                    "metric": {"has_data": True, "cpu_spike_zscore": 289.6,
                               "cpu_max": 5.16, "mem_jump_ratio": 0.0},
                    "latency": {"has_data": False},
                    "retry_timeout": {"has_data": False},
                },
            }
        }
        col = build_collection_from_mcp_outputs(stats, metrics, time_range=TR)
        # checkoutservice: 3 log + 1 metric = 4 evidence
        # frontend: no evidence
        assert "checkoutservice" in col.services_covered()
        assert "frontend" not in col.services_covered()
        assert len(col.by_modality("log")) >= 3
        assert len(col.by_modality("metric")) == 1

    def test_time_range_required(self):
        with pytest.raises(ValueError):
            build_collection_from_mcp_outputs({}, time_range=None)


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
