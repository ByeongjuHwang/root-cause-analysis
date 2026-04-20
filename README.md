# TCB-RCA: Multi-Agent Root Cause Analysis Framework

> MCP + A2A 기반 멀티에이전트 마이크로서비스 장애 RCA 프레임워크
> 핵심 알고리즘: **TCB-RCA** (Topology-Constrained Temporal Backtracking RCA)

## Architecture

```
┌─────────────┐      A2A       ┌────────────────┐
│ Orchestrator │◄──────────────►│   Log Agent    │──► Observability MCP
│              │                └────────────────┘
│              │      A2A       ┌────────────────┐
│              │◄──────────────►│ Topology Agent │──► Architecture MCP
│              │                └────────────────┘
│              │      A2A       ┌────────────────┐
│              │◄──────────────►│   RCA Agent    │    (TCB-RCA Engine)
│              │                └────────────────┘
│              │      A2A       ┌────────────────┐
│              │◄──────────────►│ Verifier Agent │
└─────────────┘                └────────────────┘
```

## Quick Start

### 권장: uv 사용 (빠르고 재현성 높음)

```bash
# 1. uv 설치 (최초 한 번)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 가상환경 생성 + 활성화
uv venv --python 3.11
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. 의존성 설치
uv pip install -r requirements.txt

# 4. 데모 실행 (결정론적 모드, LLM 불필요)
python run_demo.py

# 5. TCB-RCA 단위 테스트
python test_tcb_rca.py

# 6. E2E 검증
python test_end_to_end.py
```

자세한 환경 구성은 **[SETUP.md](SETUP.md)** 참조.

### 대안: pip + venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
```

## Experiment Reproduction

### Single Experiment

```bash
# System variants: ours, b1 (monolithic), b2 (no-MCP), b3 (no-verifier)
# Scenarios: s1~s8 (synthetic), case1/case2 (real-world)

python run_experiment.py --system ours --scenario s1
python run_experiment.py --system b1   --scenario s3
python run_experiment.py --system ours --scenario case1

# Pure JSON output (for scripting)
python run_experiment.py --system ours --scenario s1 --json --quiet
```

### Full Experiment Suite

```bash
# Run all 34 experiments (ours/b1/b2/b3 × s1~s8 + case studies)
python run_all_experiments.py

# With CSV export for paper tables
python run_all_experiments.py --csv

# Selective execution
python run_all_experiments.py --systems ours b3 --scenarios s1 s2 s3
```

### RCAEval Benchmark (Optional)

```bash
# Requires RCAEval RE2-OB dataset
python run_rcaeval.py --data-dir <path_to_rcaeval_data>
```

## System Variants (Paper Table 1)

| Key   | Description                              | Agents | MCP | Verifier |
|-------|------------------------------------------|--------|-----|----------|
| ours  | Full framework (Multi-Agent + MCP)       | 5      | ✓   | ✓        |
| b1    | Monolithic single-process                | 1      | ✗   | ✗        |
| b2    | No-MCP parallel pipeline                 | 5      | ✗   | ✓        |
| b3    | No-verifier pipeline                     | 4      | ✓   | ✗        |

## Synthetic Scenarios (S1~S8)

| ID  | Fault Type                    | Root Cause       | Challenge          |
|-----|-------------------------------|------------------|--------------------|
| S1  | DB Connection Pool Exhaustion | user-db          | Multi-hop cascade  |
| S2  | Worker Crash + Queue Backlog  | worker-service   | Async propagation  |
| S3  | Catalog Slow Query            | catalog-service  | Latency-only fault |
| S4  | DB Disk Full (Fan-out)        | order-db         | Fan-out impact     |
| S5  | GC Pause (False Positive)     | (none)           | False alarm        |
| S6  | Noisy DB Fault                | user-db          | Misleading noise   |
| S7  | Concurrent Dual Faults        | user-db + worker | Multiple roots     |
| S8  | Partial Observability         | user-db          | Missing logs       |

## Configuration

| Environment Variable          | Default | Description                        |
|-------------------------------|---------|------------------------------------|
| `USE_LLM_AGENT`              | false   | Enable LLM-augmented agents        |
| `LLM_MODEL`                  | —       | OpenAI model name (if LLM enabled) |
| `OPENAI_API_KEY`             | —       | API key (if LLM enabled)           |
| `OBSERVABILITY_LOG_FILE`     | —       | Override log data source            |
| `ARCHITECTURE_TOPOLOGY_FILE` | —       | Override topology data source       |

## Deterministic vs LLM Mode

```bash
# Deterministic mode (default, no API key needed)
python run_demo.py

# LLM mode (requires OPENAI_API_KEY)
USE_LLM_AGENT=true python run_demo.py
```

## Output Format

Results are saved in `experiments/` as JSON:

```
experiments/
├── ours_s1.json         # Individual experiment results
├── b1_s1.json
├── ...
└── summaries/
    ├── summary_latest.json    # Latest run summary
    ├── results_*.csv          # Per-experiment CSV (--csv flag)
    └── summary_table_*.csv    # Aggregated metrics CSV
```

## Project Structure

```
├── agents/
│   ├── log_agent/           # Log analysis (Observability MCP client)
│   ├── topology_agent/      # Topology analysis (Architecture MCP client)
│   ├── rca_agent/           # TCB-RCA engine
│   │   ├── tcb_rca.py       # Core algorithm implementation
│   │   └── service.py       # Dynamic topology resolution
│   ├── verifier_agent/      # Cross-validation
│   ├── orchestrator/        # A2A coordination (graceful degradation)
│   └── monolithic/          # B1 baseline (single-process)
├── mcp_servers/
│   ├── observability_mcp/   # Log data server
│   └── architecture_mcp/    # Topology data server
├── scenarios/               # S2~S8 synthetic log files
├── experiments/             # Output directory
├── experiment_core.py       # Single source of truth: SYSTEMS, SCENARIOS, evaluate_prediction
├── run_demo.py              # Quick demo
├── run_experiment.py        # Single experiment runner (uses experiment_core)
├── run_all_experiments.py   # Batch runner + summary
├── run_rcaeval.py           # RCAEval benchmark (uses experiment_core)
├── test_tcb_rca.py          # TCB-RCA unit test
└── test_end_to_end.py       # E2E integration test
```
