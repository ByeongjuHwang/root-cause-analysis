"""
Phase 3b unit tests: evidence injection into Log Agent prompt.

These tests exercise build_log_agent_user_prompt() with the new
evidence_collection argument. They do NOT spin up agents/MCP servers.
End-to-end behaviour is verified by RCAEval smoke separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from agents.log_agent.skills_llm import build_log_agent_user_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _basic_prompt_kwargs():
    """Minimal kwargs to build a prompt without evidence injection."""
    return dict(
        symptom_service="frontend",
        symptom="latency degraded",
        time_range_start="2026-04-23T13:00:00+09:00",
        time_range_end="2026-04-23T13:05:00+09:00",
        service_error_summary={},
        error_evidence_samples=[],
    )


def _evidence_payload_with(units):
    """Build an EvidenceCollection-shaped payload for prompt injection."""
    return {
        "count": len(units),
        "services_covered": sorted({s for u in units for s in u["services"]}),
        "modalities_present": sorted({u["modality"] for u in units}),
        "units": units,
    }


def _make_unit(service, modality, severity, anomaly):
    return {
        "evidence_id": f"ev_{modality}_{abs(hash((service, modality, anomaly))):08x}"[:14],
        "modality": modality,
        "anomaly_type": anomaly,
        "severity": severity,
        "services": [service],
        "time_range": {
            "start": "2026-04-23T13:00:00+09:00",
            "end": "2026-04-23T13:05:00+09:00",
        },
        "observation": {},
        "source": "test",
        "topology_path": None,
        "supporting_refs": [],
        "raw_samples": [],
    }


# ---------------------------------------------------------------------------
# Backward compatibility — no evidence_collection passed
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_no_evidence_arg_works(self):
        """Phase 3b must not break callers that don't pass evidence_collection."""
        prompt = build_log_agent_user_prompt(**_basic_prompt_kwargs())
        assert isinstance(prompt, str)
        assert "Evidence-Aware Summary" not in prompt

    def test_evidence_none_no_section(self):
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=None,
        )
        assert "Evidence-Aware Summary" not in prompt

    def test_evidence_empty_units_no_section(self):
        empty_payload = _evidence_payload_with([])
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=empty_payload,
        )
        # No units → no section at all (factory-style guard, not just an empty header)
        assert "Evidence-Aware Summary" not in prompt


# ---------------------------------------------------------------------------
# Evidence injection — section content
# ---------------------------------------------------------------------------

class TestEvidenceSectionContent:
    def test_dominant_service_appears(self):
        units = [
            _make_unit("recommendationservice", "metric", 0.95, "resource_saturation"),
        ]
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=_evidence_payload_with(units),
        )
        assert "Evidence-Aware Summary" in prompt
        assert "recommendationservice" in prompt
        assert "0.95" in prompt
        assert "resource_saturation" in prompt

    def test_multiple_services_ranked_by_severity(self):
        units = [
            _make_unit("low-svc", "metric", 0.40, "latency_degradation"),
            _make_unit("high-svc", "metric", 0.95, "resource_saturation"),
            _make_unit("mid-svc", "metric", 0.60, "network_degradation"),
        ]
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=_evidence_payload_with(units),
        )
        # high-svc must appear earlier in the prompt than low-svc
        idx_high = prompt.find("high-svc")
        idx_mid = prompt.find("mid-svc")
        idx_low = prompt.find("low-svc")
        assert -1 < idx_high < idx_mid < idx_low

    def test_modalities_listed_in_section_header(self):
        units = [
            _make_unit("svc", "metric", 0.5, "latency_degradation"),
            _make_unit("svc", "log", 0.4, "error_spike"),
        ]
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=_evidence_payload_with(units),
        )
        assert "modalities:" in prompt
        # Both modalities mentioned in the section header
        ev_section_start = prompt.find("Evidence-Aware Summary")
        next_section_start = prompt.find("##", ev_section_start + 5)
        section = prompt[ev_section_start:next_section_start]
        assert "metric" in section
        assert "log" in section

    def test_dominant_caps_at_5(self):
        units = [
            _make_unit(f"svc{i}", "metric", 0.5 + i * 0.01, "resource_saturation")
            for i in range(10)
        ]
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=_evidence_payload_with(units),
        )
        # Top 5 must appear; the rest should not
        for i in range(5, 10):
            top_5_threshold = 9 - i  # services with severity 0.59..0.55 should be in top 5
            # svc9, svc8, svc7, svc6, svc5 should be present (top by severity)
        # Verify top 5 (highest severity) are listed:
        for i in range(9, 4, -1):
            assert f"svc{i}" in prompt
        # Lowest 5 (svc0..svc4) should NOT be in the dominant list
        # (but they might appear if mentioned in another section — here there's none)
        for i in range(5):
            assert f"svc{i}" not in prompt

    def test_section_includes_guidance_on_hub_bias(self):
        """The prompt should instruct the LLM how to handle evidence-vs-stats
        conflicts (hub bias, symptom service interpretation, multi-modality)."""
        units = [_make_unit("svc", "metric", 0.9, "resource_saturation")]
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=_evidence_payload_with(units),
        )
        # Look for the three guidance bullets
        assert "Hub service" in prompt
        assert "Symptom service" in prompt
        assert "multi-modality" in prompt or "Multi-modality" in prompt


# ---------------------------------------------------------------------------
# Multi-modality aggregation per service
# ---------------------------------------------------------------------------

class TestMultiModalityAggregation:
    def test_max_severity_used(self):
        units = [
            _make_unit("svc", "log", 0.4, "error_spike"),
            _make_unit("svc", "metric", 0.9, "resource_saturation"),
        ]
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=_evidence_payload_with(units),
        )
        # Highest sev (0.90) and its associated anomaly should appear
        assert "0.90" in prompt
        assert "resource_saturation" in prompt

    def test_evidence_count_aggregated(self):
        units = [
            _make_unit("svc", "log", 0.5, "error_spike"),
            _make_unit("svc", "log", 0.6, "volume_shift"),
            _make_unit("svc", "metric", 0.7, "resource_saturation"),
        ]
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            evidence_collection=_evidence_payload_with(units),
        )
        assert "evidence_count=3" in prompt


# ---------------------------------------------------------------------------
# Section ordering — must come AFTER metric_summaries, BEFORE error summary
# ---------------------------------------------------------------------------

class TestSectionOrdering:
    def test_evidence_after_metric_before_error_summary(self):
        units = [_make_unit("svc", "metric", 0.9, "resource_saturation")]
        metric_summaries = {
            "services": {
                "svc": {
                    "metric": {"has_data": True, "cpu_spike_zscore": 100.0,
                               "cpu_max": 4.0, "mem_jump_ratio": 0.0},
                    "latency": {"has_data": False},
                    "retry_timeout": {"has_data": False},
                }
            }
        }
        prompt = build_log_agent_user_prompt(
            **_basic_prompt_kwargs(),
            metric_summaries=metric_summaries,
            evidence_collection=_evidence_payload_with(units),
        )
        idx_metric = prompt.find("Per-service Metric") if "Per-service Metric" in prompt else \
                     prompt.find("Metric Summary") if "Metric Summary" in prompt else \
                     prompt.find("metric")  # loose match — actual heading varies
        idx_evidence = prompt.find("Evidence-Aware Summary")
        idx_error = prompt.find("Service-level Error Summary")

        assert idx_evidence > 0
        assert idx_error > 0
        assert idx_evidence < idx_error
        # Best-effort: evidence is after some kind of metric mention
        if idx_metric > 0:
            assert idx_metric < idx_evidence


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
