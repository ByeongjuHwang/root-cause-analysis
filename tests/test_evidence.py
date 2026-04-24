"""
Phase 1 unit tests: common/evidence.py

실행:
    cd C:\\Users\\hwang\\Documents\\GitHub\\rca_work
    python -m pytest tests/test_evidence.py -v

또는 스크립트처럼:
    python tests/test_evidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running both via pytest and as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from pydantic import ValidationError

from common.evidence import (
    EvidenceCollection,
    EvidenceUnit,
    TimeRange,
    completeness_score,
    make_evidence_id,
)


# ---------------------------------------------------------------------------
# TimeRange
# ---------------------------------------------------------------------------

class TestTimeRange:
    def test_valid_construction(self):
        tr = TimeRange(
            start="2026-04-23T13:00:00+09:00",
            end="2026-04-23T13:05:00+09:00",
        )
        assert tr.duration_seconds() == 300

    def test_malformed_start_rejected(self):
        with pytest.raises(ValidationError):
            TimeRange(start="not-a-date", end="2026-04-23T13:05:00+09:00")

    def test_malformed_end_rejected(self):
        with pytest.raises(ValidationError):
            TimeRange(start="2026-04-23T13:00:00+09:00", end="bogus")

    def test_contains(self):
        tr = TimeRange(
            start="2026-04-23T13:00:00+09:00",
            end="2026-04-23T13:05:00+09:00",
        )
        assert tr.contains("2026-04-23T13:02:00+09:00")
        assert not tr.contains("2026-04-23T12:59:00+09:00")
        assert not tr.contains("2026-04-23T13:06:00+09:00")

    def test_contains_malformed_returns_false(self):
        tr = TimeRange(
            start="2026-04-23T13:00:00+09:00",
            end="2026-04-23T13:05:00+09:00",
        )
        assert tr.contains("not-a-date") is False

    def test_overlaps(self):
        a = TimeRange(start="2026-04-23T13:00:00+09:00", end="2026-04-23T13:05:00+09:00")
        b = TimeRange(start="2026-04-23T13:04:00+09:00", end="2026-04-23T13:08:00+09:00")
        c = TimeRange(start="2026-04-23T13:10:00+09:00", end="2026-04-23T13:15:00+09:00")
        assert a.overlaps(b)
        assert b.overlaps(a)  # symmetric
        assert not a.overlaps(c)

    def test_frozen(self):
        tr = TimeRange(start="2026-04-23T13:00:00+09:00", end="2026-04-23T13:05:00+09:00")
        with pytest.raises((ValidationError, TypeError)):
            tr.start = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvidenceUnit
# ---------------------------------------------------------------------------

class TestEvidenceUnit:
    @staticmethod
    def _make(**overrides):
        defaults = dict(
            modality="log",
            time_range=TimeRange(
                start="2026-04-23T13:00:00+09:00",
                end="2026-04-23T13:05:00+09:00",
            ),
            services=["checkoutservice"],
            anomaly_type="error_spike",
            severity=0.8,
            observation={"error_count": 142, "baseline": 3},
            source="observability_mcp/log_repository",
        )
        defaults.update(overrides)
        return EvidenceUnit(**defaults)

    def test_basic_construction(self):
        ev = self._make()
        assert ev.modality == "log"
        assert ev.services == ["checkoutservice"]
        assert ev.severity == 0.8
        assert ev.evidence_id.startswith("ev_")

    def test_severity_bounds_rejected(self):
        with pytest.raises(ValidationError):
            self._make(severity=1.5)
        with pytest.raises(ValidationError):
            self._make(severity=-0.1)

    def test_empty_services_rejected(self):
        with pytest.raises(ValidationError):
            self._make(services=[])
        with pytest.raises(ValidationError):
            self._make(services=["", "  "])

    def test_services_stripped(self):
        ev = self._make(services=["  checkoutservice  ", "frontend"])
        assert ev.services == ["checkoutservice", "frontend"]

    def test_raw_samples_bounded(self):
        long_msg = "x" * 500
        ev = self._make(raw_samples=[long_msg] * 10)
        # Capped at 5 samples, 240 chars each
        assert len(ev.raw_samples) <= 5
        for s in ev.raw_samples:
            assert len(s) <= 240

    def test_involves_service_case_insensitive(self):
        ev = self._make(services=["checkoutservice"])
        assert ev.involves_service("CheckoutService")
        assert ev.involves_service("  checkoutservice  ")
        assert not ev.involves_service("frontend")

    def test_frozen(self):
        ev = self._make()
        with pytest.raises((ValidationError, TypeError)):
            ev.severity = 0.5  # type: ignore[misc]

    def test_to_prompt_snippet_length_bounded(self):
        ev = self._make(
            observation={f"key_{i}": "x" * 50 for i in range(20)},
        )
        snippet = ev.to_prompt_snippet(max_len=200)
        assert len(snippet) <= 200
        # 핵심 정보는 여전히 포함
        assert ev.evidence_id in snippet or snippet.endswith("...")

    def test_with_supporting_creates_new_unit(self):
        ev = self._make()
        ev2 = ev.with_supporting("ev_xyz_00001", "ev_abc_00002")
        assert "ev_xyz_00001" in ev2.supporting_refs
        assert "ev_abc_00002" in ev2.supporting_refs
        # Original unchanged (immutability)
        assert ev.supporting_refs == []

    def test_with_supporting_deduplicates(self):
        ev = self._make()
        ev2 = ev.with_supporting("ev_x", "ev_x", "ev_y")
        assert ev2.supporting_refs == ["ev_x", "ev_y"]

    def test_json_roundtrip(self):
        ev = self._make()
        json_str = ev.model_dump_json()
        restored = EvidenceUnit.model_validate_json(json_str)
        assert restored.evidence_id == ev.evidence_id
        assert restored.services == ev.services
        assert restored.severity == ev.severity
        assert restored.observation == ev.observation

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceUnit(
                modality="log",
                time_range=TimeRange(
                    start="2026-04-23T13:00:00+09:00",
                    end="2026-04-23T13:05:00+09:00",
                ),
                services=["x"],
                anomaly_type="error_spike",
                severity=0.5,
                observation={},
                source="src",
                bogus_field="nope",  # type: ignore[call-arg]
            )


class TestPrecededBy:
    """Phase 4a: temporal ordering between evidences."""

    @staticmethod
    def _make(**overrides):
        defaults = dict(
            modality="log",
            time_range=TimeRange(
                start="2026-04-23T13:00:00+09:00",
                end="2026-04-23T13:05:00+09:00",
            ),
            services=["x"],
            anomaly_type="error_spike",
            severity=0.5,
            observation={},
            source="t",
        )
        defaults.update(overrides)
        return EvidenceUnit(**defaults)

    def test_default_empty(self):
        ev = self._make()
        assert ev.preceded_by == []

    def test_with_predecessors(self):
        ev = self._make(preceded_by=["ev_log_predecessor1", "ev_log_predecessor2"])
        assert len(ev.preceded_by) == 2
        assert "ev_log_predecessor1" in ev.preceded_by

    def test_json_roundtrip_preserves_preceded_by(self):
        ev = self._make(preceded_by=["ev_log_0001", "ev_metric_0002"])
        restored = EvidenceUnit.model_validate_json(ev.model_dump_json())
        assert restored.preceded_by == ev.preceded_by


# ---------------------------------------------------------------------------
# make_evidence_id (deterministic)
# ---------------------------------------------------------------------------

class TestDeterministicID:
    def test_same_inputs_same_id(self):
        tr = TimeRange(start="2026-04-23T13:00:00+09:00", end="2026-04-23T13:05:00+09:00")
        id1 = make_evidence_id("log", ["checkoutservice"], tr, "error_spike")
        id2 = make_evidence_id("log", ["checkoutservice"], tr, "error_spike")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        tr = TimeRange(start="2026-04-23T13:00:00+09:00", end="2026-04-23T13:05:00+09:00")
        id1 = make_evidence_id("log", ["checkoutservice"], tr, "error_spike")
        id2 = make_evidence_id("log", ["frontend"],      tr, "error_spike")
        id3 = make_evidence_id("metric", ["checkoutservice"], tr, "error_spike")
        assert id1 != id2
        assert id1 != id3

    def test_service_order_normalised(self):
        tr = TimeRange(start="2026-04-23T13:00:00+09:00", end="2026-04-23T13:05:00+09:00")
        id1 = make_evidence_id("log", ["a", "b"], tr, "error_spike")
        id2 = make_evidence_id("log", ["b", "a"], tr, "error_spike")
        assert id1 == id2  # sorted internally

    def test_id_format(self):
        tr = TimeRange(start="2026-04-23T13:00:00+09:00", end="2026-04-23T13:05:00+09:00")
        ident = make_evidence_id("log", ["x"], tr, "error_spike")
        assert ident.startswith("ev_log_")
        assert len(ident.split("_")[-1]) == 8  # 8 hex chars


# ---------------------------------------------------------------------------
# EvidenceCollection
# ---------------------------------------------------------------------------

class TestEvidenceCollection:
    @staticmethod
    def _make_unit(service, modality="log", anomaly="error_spike", severity=0.5):
        return EvidenceUnit(
            modality=modality,
            time_range=TimeRange(
                start="2026-04-23T13:00:00+09:00",
                end="2026-04-23T13:05:00+09:00",
            ),
            services=[service],
            anomaly_type=anomaly,
            severity=severity,
            observation={},
            source="test",
        )

    def test_add_and_len(self):
        col = EvidenceCollection()
        col.add(self._make_unit("svc1"))
        col.add(self._make_unit("svc2"))
        assert len(col) == 2

    def test_duplicate_id_ignored(self):
        col = EvidenceCollection()
        u = self._make_unit("svc1")
        col.add(u)
        col.add(u)  # same object, same id
        assert len(col) == 1

    def test_by_service(self):
        col = EvidenceCollection()
        col.add(self._make_unit("svc1"))
        col.add(self._make_unit("svc1", modality="metric", anomaly="resource_saturation"))
        col.add(self._make_unit("svc2"))
        evs = col.by_service("svc1")
        assert len(evs) == 2

    def test_by_modality(self):
        col = EvidenceCollection()
        col.add(self._make_unit("svc1", modality="log"))
        col.add(self._make_unit("svc2", modality="metric", anomaly="resource_saturation"))
        col.add(self._make_unit("svc3", modality="metric", anomaly="latency_degradation"))
        assert len(col.by_modality("log")) == 1
        assert len(col.by_modality("metric")) == 2

    def test_services_covered_preserves_order(self):
        col = EvidenceCollection()
        col.add(self._make_unit("alpha"))
        col.add(self._make_unit("bravo"))
        col.add(self._make_unit("alpha"))  # duplicate service, different evidence
        assert col.services_covered() == ["alpha", "bravo"]

    def test_strongest_by_service(self):
        col = EvidenceCollection()
        col.add(self._make_unit("svc1", severity=0.3))
        col.add(self._make_unit("svc1", anomaly="volume_shift", severity=0.8))
        col.add(self._make_unit("svc1", anomaly="keyword_distress", severity=0.5))
        strongest = col.strongest_by_service("svc1")
        assert strongest is not None
        assert strongest.severity == 0.8

    def test_strongest_by_service_missing_returns_none(self):
        col = EvidenceCollection()
        assert col.strongest_by_service("nonexistent") is None


# ---------------------------------------------------------------------------
# completeness_score
# ---------------------------------------------------------------------------

class TestCompletenessScore:
    @staticmethod
    def _make_unit(service, modality="log", severity=0.5):
        return EvidenceUnit(
            modality=modality,
            time_range=TimeRange(
                start="2026-04-23T13:00:00+09:00",
                end="2026-04-23T13:05:00+09:00",
            ),
            services=[service],
            anomaly_type="error_spike" if modality == "log" else "resource_saturation",
            severity=severity,
            observation={},
            source="test",
        )

    def test_empty_candidates(self):
        assert completeness_score(EvidenceCollection(), []) == 0.0

    def test_full_coverage(self):
        col = EvidenceCollection()
        col.add(self._make_unit("svc1"))
        col.add(self._make_unit("svc1", modality="metric"))
        col.add(self._make_unit("svc2"))
        col.add(self._make_unit("svc2", modality="metric"))
        score = completeness_score(col, ["svc1", "svc2"])
        assert score >= 0.8

    def test_partial_coverage(self):
        col = EvidenceCollection()
        col.add(self._make_unit("svc1"))
        # svc2 has no evidence
        score = completeness_score(col, ["svc1", "svc2"])
        assert 0.0 < score < 1.0

    def test_modality_requirement(self):
        col = EvidenceCollection()
        col.add(self._make_unit("svc1", modality="log"))
        low = completeness_score(col, ["svc1"], required_modalities=["log", "metric"])
        col.add(self._make_unit("svc1", modality="metric"))
        high = completeness_score(col, ["svc1"], required_modalities=["log", "metric"])
        assert high > low


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
