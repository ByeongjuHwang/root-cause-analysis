"""
Phase 3a tests: agents/log_agent/shadow.py

실행:
    python -m pytest tests/test_shadow.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from agents.log_agent.shadow import (
    _compare_legacy_vs_evidence,
    _summarise_collection,
    run_shadow_evidence_collection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Change cwd so llm_logs/ is written under tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_payload(services_with_severity):
    """Build a payload shaped like get_evidence_collection_payload output.

    services_with_severity: list of (service, modality, severity, anomaly).
    """
    units = []
    for i, (svc, modality, sev, anom) in enumerate(services_with_severity):
        units.append({
            "evidence_id": f"ev_{modality}_{i:08x}",
            "modality": modality,
            "anomaly_type": anom,
            "severity": sev,
            "services": [svc],
            "time_range": {"start": "2026-04-23T13:00:00+09:00",
                           "end": "2026-04-23T13:05:00+09:00"},
            "observation": {},
            "source": "test",
            "topology_path": None,
            "supporting_refs": [], "raw_samples": [],
        })
    return {
        "count": len(units),
        "services_covered": sorted({u["services"][0] for u in units}),
        "modalities_present": sorted({u["modality"] for u in units}),
        "units": units,
    }


# ---------------------------------------------------------------------------
# _summarise_collection
# ---------------------------------------------------------------------------

class TestSummariseCollection:
    def test_empty_payload(self):
        summary = _summarise_collection({"count": 0, "services_covered": [],
                                          "modalities_present": [], "units": []})
        assert summary["total_units"] == 0
        assert summary["dominant_services"] == []

    def test_single_service_multiple_modalities(self):
        payload = _fake_payload([
            ("checkoutservice", "log", 0.8, "error_spike"),
            ("checkoutservice", "metric", 0.95, "resource_saturation"),
        ])
        summary = _summarise_collection(payload)
        assert summary["total_units"] == 2
        assert summary["dominant_services"][0]["service"] == "checkoutservice"
        # The metric evidence has higher severity, so top_anomaly should be resource_saturation
        assert summary["dominant_services"][0]["top_anomaly"] == "resource_saturation"
        assert summary["dominant_services"][0]["max_severity"] == 0.95

    def test_dominant_ranking_by_severity(self):
        payload = _fake_payload([
            ("low-sev", "log", 0.3, "volume_shift"),
            ("high-sev", "metric", 0.9, "resource_saturation"),
            ("mid-sev", "log", 0.6, "error_spike"),
        ])
        summary = _summarise_collection(payload)
        order = [d["service"] for d in summary["dominant_services"]]
        assert order == ["high-sev", "mid-sev", "low-sev"]

    def test_dominant_caps_at_3(self):
        payload = _fake_payload([
            (f"svc{i}", "log", 0.5 + i * 0.01, "error_spike") for i in range(5)
        ])
        summary = _summarise_collection(payload)
        assert len(summary["dominant_services"]) == 3

    def test_per_service_counts_correctly(self):
        payload = _fake_payload([
            ("a", "log", 0.5, "error_spike"),
            ("a", "log", 0.6, "volume_shift"),
            ("a", "metric", 0.7, "resource_saturation"),
            ("b", "log", 0.4, "error_spike"),
        ])
        summary = _summarise_collection(payload)
        a_info = summary["per_service"]["a"]
        assert a_info["log_count"] == 2
        assert a_info["metric_count"] == 1
        assert a_info["trace_count"] == 0


# ---------------------------------------------------------------------------
# _compare_legacy_vs_evidence
# ---------------------------------------------------------------------------

class TestComparison:
    def test_agreement_detected(self):
        legacy = {
            "suspected_downstream": "checkoutservice",
            "anomalous_services": ["checkoutservice", "frontend"],
        }
        evidence_summary = {
            "dominant_services": [
                {"service": "checkoutservice", "max_severity": 0.95},
                {"service": "frontend", "max_severity": 0.5},
            ]
        }
        comp = _compare_legacy_vs_evidence(legacy, evidence_summary)
        assert comp["top_matches"] is True
        assert comp["legacy_top_in_evidence_top3"] is True

    def test_disagreement(self):
        legacy = {
            "suspected_downstream": "redis",
            "anomalous_services": ["redis"],
        }
        evidence_summary = {
            "dominant_services": [
                {"service": "checkoutservice", "max_severity": 0.95},
            ]
        }
        comp = _compare_legacy_vs_evidence(legacy, evidence_summary)
        assert comp["top_matches"] is False
        assert comp["legacy_top"] == "redis"
        assert comp["evidence_top"] == "checkoutservice"

    def test_both_empty(self):
        legacy = {"suspected_downstream": None, "anomalous_services": []}
        evidence_summary = {"dominant_services": []}
        comp = _compare_legacy_vs_evidence(legacy, evidence_summary)
        assert comp["top_matches"] is False  # both None — count as no match
        assert comp["legacy_top"] is None
        assert comp["evidence_top"] is None


# ---------------------------------------------------------------------------
# run_shadow_evidence_collection — integration
# ---------------------------------------------------------------------------

class TestRunShadow:
    def _fake_legacy_result(self):
        return {
            "suspected_downstream": "checkoutservice",
            "anomalous_services": ["checkoutservice", "frontend"],
            "confidence": 0.75,
            "evidence": [{"dummy": 1}, {"dummy": 2}],
            "hypothesis": "something about checkoutservice",
        }

    def test_writes_shadow_file(self, tmp_cwd):
        fake_payload = _fake_payload([
            ("checkoutservice", "metric", 0.95, "resource_saturation"),
        ])
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.get_evidence_collection_payload",
            return_value=fake_payload,
        ):
            out = run_shadow_evidence_collection(
                legacy_result=self._fake_legacy_result(),
                incident_id="TEST_INC_001",
                symptom_service="frontend",
                start="2026-04-23T13:00:00+09:00",
                end="2026-04-23T13:05:00+09:00",
            )

        assert out is not None
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["schema_version"] == "shadow.v1"
        assert data["incident_id"] == "TEST_INC_001"
        assert data["evidence_aware"]["summary"]["total_units"] == 1
        assert data["comparison"]["top_matches"] is True

    def test_disabled_via_env_returns_none(self, tmp_cwd, monkeypatch):
        monkeypatch.setenv("LOG_AGENT_SHADOW_DISABLE", "1")
        out = run_shadow_evidence_collection(
            legacy_result=self._fake_legacy_result(),
            incident_id="TEST_INC_001",
            symptom_service="frontend",
            start="2026-04-23T13:00:00+09:00",
            end="2026-04-23T13:05:00+09:00",
        )
        assert out is None

    def test_evidence_failure_still_writes_record(self, tmp_cwd):
        """If evidence collection raises, we still log (with error field)."""
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.get_evidence_collection_payload",
            side_effect=RuntimeError("simulated"),
        ):
            out = run_shadow_evidence_collection(
                legacy_result=self._fake_legacy_result(),
                incident_id="TEST_INC_002",
                symptom_service="frontend",
                start="2026-04-23T13:00:00+09:00",
                end="2026-04-23T13:05:00+09:00",
            )
        assert out is not None
        data = json.loads(out.read_text(encoding="utf-8"))
        # Should record the error in evidence_aware.full_payload
        assert "error" in data["evidence_aware"]["full_payload"]
        assert data["evidence_aware"]["summary"]["total_units"] == 0

    def test_filename_pattern(self, tmp_cwd):
        fake_payload = _fake_payload([])
        with patch(
            "mcp_servers.observability_mcp.app.evidence_tools.get_evidence_collection_payload",
            return_value=fake_payload,
        ):
            out = run_shadow_evidence_collection(
                legacy_result=self._fake_legacy_result(),
                incident_id="INC/ABC:001",  # with forbidden chars
                symptom_service="svc",
                start="2026-04-23T13:00:00+09:00",
                end="2026-04-23T13:05:00+09:00",
            )
        # Slashes/colons replaced
        assert "/" not in out.name
        assert ":" not in out.name
        assert out.name.startswith(out.name[:8])  # timestamp-like prefix
        assert "log_agent_shadow" in out.name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
