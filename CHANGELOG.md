# TCB-RCA Framework — Paper Artifact Refactoring Changelog

## Phase 1: Core Structure Fixes

### 1. Dynamic Topology Resolution (Critical)
**Files**: `agents/rca_agent/service.py`, `agents/topology_agent/skills.py`

**Before**: RCA engine used hardcoded `_TOPOLOGY_GRAPH` constant. Topology Agent output was ignored by TCB-RCA.

**After**: 3-level fallback chain:
1. `topology_result["topology_graph"]` — Topology Agent passes full graph
2. Architecture MCP repository — direct load respecting `topology_file` env
3. Default hardcoded — last resort for backward compatibility

Topology Agent now returns `topology_graph` + `service_metadata` in every response.

### 2. Orchestrator Graceful Degradation
**Files**: `agents/orchestrator/service.py`, `agents/orchestrator/models.py`

**Before**: Any agent failure crashed the entire pipeline.

**After**: Each of 4 agent calls wrapped in try/except. On failure:
- Creates degraded `AgentResult` with error info
- Pipeline continues with remaining agents
- `agent_errors` list tracked in `FinalRCAResult`

### 3. Unified Experiment Runner
**Files**: `run_experiment.py`, `run_all_experiments.py`

**Before**: Only `ours`/`b3` in run_experiment.py. B1/B2 required separate scripts.

**After**: All 4 system variants (ours/b1/b2/b3) in single `SYSTEMS` dict.
- `--json` flag: pure JSON to stdout for programmatic use
- `--quiet` flag: suppress agent subprocess output
- `--case-study` auto-detected from scenario definition
- stderr/stdout properly separated

### 4. RCA Output `cause_service` Field
**Files**: `agents/rca_agent/service.py`

**Before**: `_format_output` only included `cause` (description text), not `cause_service`.

**After**: Every candidate includes `cause_service` (e.g., "user-db") for precise evaluation matching.

### 5. Verifier Dynamic Service Discovery
**Files**: `agents/verifier_agent/service.py`

**Before**: Hardcoded `_KNOWN_SERVICES` set.

**After**: `_extract_known_services()` dynamically extracts from agent results + default fallback.

### 6. E2E Test Stability
**Files**: `run_demo.py`, `test_end_to_end.py`

**Before**: run_demo.py mixed print() and JSON. E2E test fragile.

**After**: `run_demo.py --json` outputs pure JSON. E2E test uses `--json` flag. Added `cause_service` checks.

### 7. CSV Export
**Files**: `run_all_experiments.py`

Added `--csv` flag generating:
- `results_{timestamp}.csv` — per-experiment raw data
- `summary_table_{timestamp}.csv` — per-system aggregated metrics (for paper Table 2)

### 8. Project Metadata
**Files**: `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`

- `requires-python` relaxed to `>=3.11`
- Dependency lists synchronized
- .gitignore excludes dev artifacts
- README: full reproduction guide with system/scenario tables

## Phase 2: Evaluation & LLM Compatibility

### 9. RCAEval Benchmark Runner
**Files**: `run_rcaeval.py`

**Before**: Delegated to separate runner scripts per system variant. CLI broken for b1/b2.

**After**: Self-contained with inline service management. All 4 systems × 90 cases.
- `cause_service` matching for evaluation
- CSV export with `--csv` flag
- Service start/stop per experiment

### 10. LLM Mode Dynamic Topology
**Files**: `agents/rca_agent/skills_llm.py`, `agents/topology_agent/skills_llm.py`

**Before**: LLM RCA skills had own hardcoded `_TOPOLOGY_GRAPH`. LLM topology agent didn't return full graph.

**After**: LLM skills import and reuse `_resolve_topology()` from deterministic service. Both LLM topology return paths include `topology_graph`/`service_metadata`.

### 11. OpenRCA Separation
**Files**: `supplementary/openrca_adapter.py`, `supplementary/openrca_runner.py`

Moved OpenRCA files to `supplementary/` directory — clearly marked as auxiliary experiment, not part of main evaluation.

## Test Results

| Test Category | Count | Status |
|---|---|---|
| TCB-RCA algorithm correctness | 5 | ✅ ALL PASS |
| Dynamic topology resolution | 4 | ✅ ALL PASS |
| RCA Service integration | 3 | ✅ ALL PASS |
| Topology Agent full graph | 4 | ✅ ALL PASS |
| Verifier dynamic services | 3 | ✅ ALL PASS |
| LLM skills structure | 6 | ✅ ALL PASS |
| Topology LLM full graph | 3 | ✅ ALL PASS |
| OpenRCA separation | 3 | ✅ ALL PASS |
| RCAEval structure | 5 | ✅ ALL PASS |
| Graceful degradation | 6 | ✅ ALL PASS |
| Scenario completeness | 10 | ✅ ALL PASS |
| **TOTAL** | **43** | **✅ ALL PASS** |

---

## Phase 3: Surgical Refactoring for Paper Artifact Quality

### 12. Single Source of Truth: `experiment_core.py`
**New file**: `experiment_core.py`

**Before**: Three places defined the same data
- `SYSTEMS` was in `run_experiment.py` AND `run_rcaeval.py` (copy-paste)
- `SCENARIOS` / `SCENARIO_CONFIGS` was in `run_experiment.py` AND `run_monolithic.py` AND `run_parallel.py`
- Each runner had its own `evaluate_result()` with different matching rules

**After**: All shared definitions live in `experiment_core.py`:
- `SYSTEMS`: system variant registry (ours/b1/b2/b3) with port_base, module path, uses_verifier, uses_mcp, is_monolithic flags
- `SCENARIOS`: synthetic + case-study scenario registry with ground-truth fields
- `resolve_services(system_key)`: deterministic service list for a system
- `build_env_overrides(system_key)`: URL environment for inter-agent discovery
- `resolve_scenario_inputs(scenario_key)`: unified input file resolution for all scenario kinds
- `scenario_ground_truth(scenario_key)`: extract GT fields for evaluation
- `candidate_matches_truth(candidate, gt)`: normalized matching (handles `cart-service` ↔ `cartservice` variants)
- `evaluate_prediction(prediction, ground_truth, elapsed)`: SINGLE evaluation function

### 13. Unified Evaluation Pipeline
**Files**: `run_experiment.py`, `run_rcaeval.py`

Synthetic scenarios, case studies, and public benchmarks now all call
`evaluate_prediction()`. Matching rules are identical across all three:
hyphens/underscores are normalized, cause_service field preferred over
free-text cause description, alternate ground-truth paths supported
(for multi-fault scenarios like S7).

### 14. Backward Compatibility Preserved
**Files**: `run_experiment.py`

- `run_single(system_key, scenario_key, json_mode, quiet)` signature unchanged
- `evaluate_result(result, scenario, elapsed)` preserved as backward-compat
  wrapper around `evaluate_prediction` (external callers unaffected)
- `SYSTEMS` and `SCENARIOS` module-level re-exports preserved

### 15. Legacy Runner Deprecation Notice
**Files**: `run_monolithic.py`, `run_parallel.py`, `run_no_verifier.py`, `run_case_study.py`

Deprecation notice in module docstring pointing to the unified entry point
(`run_experiment.py --system X --scenario Y`). No behavioral change —
they still work for anyone invoking them directly, but the paper
experiments use the unified harness.

### 16. JSON Serializability Audit
`evaluate_prediction` return value audited — every field is a
JSON-primitive (bool, int, float, str, None, list). `None` is used for
"not applicable" metrics (e.g. `path_accuracy=None` when no path GT,
`fp_handled=None` when scenario has a real root cause). This ensures
`json.dumps()` works without `default=str` hacks and downstream
analysis scripts can pandas-read the CSVs without type coercion.

### 17. Removed Demo Output from `--json` Path
**Files**: `run_experiment.py`

The `=== Config: USE_LLM_AGENT=... ===` preamble in `main()` has been
removed. All diagnostics go to stderr unconditionally; only the
structured experiment record goes to stdout in `--json` mode.

## Phase 3 Test Results

All 16 regression tests pass:

| Test Category | Count | Status |
|---|---|---|
| TCB-RCA algorithm correctness | 2 | ✅ ALL PASS |
| experiment_core registry integrity | 4 | ✅ ALL PASS |
| Backward-compat export verification | 4 | ✅ ALL PASS |
| Unified evaluation (synthetic / RCAEval / FP / JSON) | 6 | ✅ ALL PASS |
| **TOTAL** | **16** | **✅ ALL PASS** |

Combined with Phase 1+2: **59/59 PASS** across all refactoring phases.
