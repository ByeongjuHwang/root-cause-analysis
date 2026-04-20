# 논문 실험 실행 가이드 (Experiment Execution Guide)

이 문서는 논문에 포함될 실험 결과를 재현하기 위한 **단계별 실행 가이드**입니다.

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [실험 구조 개요](#2-실험-구조-개요)
3. [Step 1: 사전 검증 (Sanity Check)](#3-step-1-사전-검증)
4. [Step 2: 논문 Table 2 — 합성 시나리오 실험](#4-step-2-논문-table-2)
5. [Step 3: 논문 Table 3 — Case Study 실험](#5-step-3-논문-table-3)
6. [Step 4: 전체 일괄 실행 + CSV 생성](#6-step-4-전체-일괄-실행)
7. [Step 5: Ablation Study (B1/B2/B3 비교)](#7-step-5-ablation-study)
8. [Step 6: RCAEval RE2-OB 벤치마크 (선택)](#8-step-6-rcaeval-벤치마크)
9. [Step 7: LLM 모드 실험 (선택)](#9-step-7-llm-모드-실험)
10. [결과 해석 가이드](#10-결과-해석-가이드)
11. [트러블슈팅](#11-트러블슈팅)

---

## 1. 사전 준비

### 1.1 환경 설정

```bash
# Python 3.11 이상 필요
python --version  # Python 3.11+

# 의존성 설치
pip install -r requirements.txt

# 또는 uv 사용 시
uv pip install -r requirements.txt
```

### 1.2 디렉터리 구조 확인

```bash
# 다음 파일/폴더가 존재해야 합니다
ls run_experiment.py          # 단일 실험 러너
ls run_all_experiments.py     # 일괄 실험 러너
ls run_demo.py                # 빠른 데모
ls scenarios/                 # 합성 시나리오 로그 (S2~S8)
ls mcp_servers/               # MCP 서버 데이터
ls agents/                    # 에이전트 코드
ls case_study_logs.jsonl      # Case Study 로그
ls case_study_topology.json   # Case Study 토폴로지
```

### 1.3 포트 확인

실험은 다음 포트 범위를 사용합니다. 충돌이 없어야 합니다:

| 시스템 | 포트 범위 |
|--------|-----------|
| ours   | 18000~18004 |
| b3     | 19000~19004 |
| b1     | 20000      |
| b2     | 21000~21004 |

```bash
# 포트 사용 확인 (모두 비어있어야 함)
for port in 18000 18001 18002 18003 18004 19000 20000 21000; do
  lsof -i :$port 2>/dev/null && echo "WARNING: Port $port in use!"
done
```

---

## 2. 실험 구조 개요

### 2.1 시스템 변형 (논문 Table 1)

| Key  | 논문 명칭 | 설명 | 에이전트 수 | MCP | Verifier |
|------|----------|------|------------|-----|----------|
| `ours` | **Proposed** | 전체 프레임워크 (Multi-Agent + MCP + Verifier) | 5 | ✓ | ✓ |
| `b1` | **B1: Monolithic** | 단일 프로세스, A2A/MCP 미사용 | 1 | ✗ | ✗ |
| `b2` | **B2: No-MCP** | 멀티에이전트이나 MCP 미경유 | 5 | ✗ | ✓ |
| `b3` | **B3: No-Verifier** | MCP 사용하나 검증 에이전트 미사용 | 4 | ✓ | ✗ |

### 2.2 합성 시나리오 (S1~S8)

| ID | 장애 유형 | Ground Truth Root Cause | 전파 경로 | 난이도 특성 |
|----|----------|------------------------|-----------|------------|
| S1 | DB 커넥션 풀 고갈 | `user-db` | user-db → auth-service → api-gateway | 다단계 캐스케이드 |
| S2 | 워커 크래시 + 큐 백로그 | `worker-service` | worker-service → message-queue → order-service → api-gateway | 비동기 전파 |
| S3 | 카탈로그 슬로우 쿼리 | `catalog-service` | catalog-service → api-gateway | 지연만 관측 |
| S4 | DB 디스크 풀 | `order-db` | order-db → order-service → api-gateway | 팬아웃 영향 |
| S5 | GC 일시 정지 (**오탐**) | **(없음)** | — | False Positive 탐지 |
| S6 | Noisy DB 장애 | `user-db` | user-db → auth-service → api-gateway | 교란 에러 존재 |
| S7 | 동시 다중 장애 | `user-db` + `worker-service` | 2개 경로 동시 | 복수 근본 원인 |
| S8 | 부분 관측성 | `user-db` | user-db → auth-service → api-gateway | 근본 원인 로그 누락 |

### 2.3 실세계 Case Study

| ID | 출처 | 장애 | Ground Truth |
|----|------|------|-------------|
| case1 | AppInstallTimeout | 앱 설치 타임아웃 | `device-state-gateway` |
| case2 | AppUninstallTimeout | 앱 삭제 타임아웃 | `device-state-gateway` |

### 2.4 평가 지표

| 지표 | 설명 |
|------|------|
| **AC@1** | Top-1 후보가 ground truth root cause와 일치 |
| **AC@3** | Top-3 후보 중 하나가 ground truth와 일치 |
| **Path Accuracy** | 추론된 전파 경로가 ground truth 경로와 일치 |
| **FP Handled** | False positive 시나리오(S5)에서 낮은 신뢰도 또는 거부 |
| **Elapsed** | 분석 완료까지 소요 시간 (초) |

---

## 3. Step 1: 사전 검증

논문 실험 전에 프레임워크가 정상 동작하는지 확인합니다.

### 3.1 TCB-RCA 알고리즘 단위 테스트

```bash
python test_tcb_rca.py
```

**기대 결과**: 모든 PASS (Root cause = user-db, Confidence > 0.7)

### 3.2 E2E 통합 테스트

```bash
python test_end_to_end.py
```

**기대 결과**: `PASS: end-to-end demo validation succeeded`

### 3.3 빠른 데모 실행

```bash
# 사람이 읽기 편한 출력
python run_demo.py

# JSON 출력 확인
python run_demo.py --json | python -m json.tool | head -30
```

**기대 결과**: JSON에 `root_cause_candidates`, `verification`, `impact_analysis` 포함

---

## 4. Step 2: 논문 Table 2 — 합성 시나리오 실험

### 4.1 개별 실행 (디버깅/확인용)

```bash
# Proposed (ours) 시스템으로 각 시나리오 개별 실행
python run_experiment.py --system ours --scenario s1
python run_experiment.py --system ours --scenario s2
python run_experiment.py --system ours --scenario s3
python run_experiment.py --system ours --scenario s4
python run_experiment.py --system ours --scenario s5
python run_experiment.py --system ours --scenario s6
python run_experiment.py --system ours --scenario s7
python run_experiment.py --system ours --scenario s8
```

각 실행 후 `experiments/ours_s1.json` ~ `experiments/ours_s8.json`이 생성됩니다.

### 4.2 결과 확인 방법

```bash
# 단일 결과 확인
python -c "
import json
d = json.load(open('experiments/ours_s1.json'))
ev = d['evaluation']
print(f\"Scenario: {d['scenario_description']}\")
print(f\"AC@1: {ev['ac_at_1']}, AC@3: {ev['ac_at_3']}\")
print(f\"Path Accuracy: {ev['path_accuracy']}\")
print(f\"Confidence: {ev['top_confidence']:.2f}\")
print(f\"Predicted: {ev['predicted_top_cause_service']}\")
print(f\"Ground Truth: {ev['ground_truth_root_cause']}\")
print(f\"Elapsed: {ev['elapsed_seconds']:.1f}s\")
"
```

### 4.3 JSON 모드 (스크립트 연동용)

```bash
# 결과를 stdout에 pure JSON으로 받기
python run_experiment.py --system ours --scenario s1 --json --quiet > result_s1.json

# jq로 평가 결과만 추출
python run_experiment.py --system ours --scenario s1 --json --quiet \
  | python -c "import json,sys; print(json.dumps(json.load(sys.stdin)['evaluation'], indent=2))"
```

---

## 5. Step 3: 논문 Table 3 — Case Study 실험

### 5.1 Case Study 실행

```bash
# Case Study 1: AppInstallTimeout
python run_experiment.py --system ours --scenario case1

# Case Study 2: AppUninstallTimeout
python run_experiment.py --system ours --scenario case2
```

**참고**: Case Study는 별도의 토폴로지(`case_study_topology.json`)와 로그(`case_study_logs.jsonl`)를 사용합니다. `--scenario case1`을 지정하면 자동으로 인식됩니다.

### 5.2 결과 확인

```bash
python -c "
import json
for cs in ['case1', 'case2']:
    d = json.load(open(f'experiments/ours_{cs}.json'))
    ev = d['evaluation']
    print(f\"{d['scenario_description']}:\")
    print(f\"  AC@1={ev['ac_at_1']} Predicted={ev['predicted_top_cause_service']} GT={ev['ground_truth_root_cause']}\")
    print()
"
```

---

## 6. Step 4: 전체 일괄 실행

### 6.1 전체 실험 (논문 전체 표 생성)

```bash
# 34개 실험 일괄 실행 (약 15~25분 소요)
# ours × S1~S8 + case1,case2 = 10
# b1 × S1~S8 = 8
# b2 × S1~S8 = 8
# b3 × S1~S8 = 8
python run_all_experiments.py
```

**출력 예시**:
```
  Plan: 34 experiments
    1. ours x s1
    2. ours x s2
    ...

  [1/34] Running ours x s1...
  ours  s1     | AC@1:✓ AC@3:✓ Path:✓ FP:- | conf=0.95 |  30.5s | GT:user-db              Pred:user-db
  ...

  ====================================================================
    EXPERIMENT SUMMARY
  ====================================================================

    System: ours
      Total experiments: 10
      AC@1:  9/10 (90.0%)
      AC@3:  9/10 (90.0%)
      Path:  7/7 (100.0%)
      FP:    1/1 (100.0%)
      Avg elapsed: 30.9s

    System: b1
      ...
```

### 6.2 CSV 포함 실행 (논문 표 작성용)

```bash
python run_all_experiments.py --csv
```

**생성 파일**:
```
experiments/summaries/
├── summary_20260418_143000.json       # 전체 결과 JSON
├── summary_latest.json                # 최신 결과 (고정 이름)
├── results_20260418_143000.csv        # 개별 실험 결과 CSV
└── summary_table_20260418_143000.csv  # 시스템별 집계 CSV (논문 표 직접 사용)
```

### 6.3 부분 실행 (특정 시스템/시나리오만)

```bash
# ours와 b3만 비교
python run_all_experiments.py --systems ours b3

# S1~S4만 빠르게 테스트
python run_all_experiments.py --scenarios s1 s2 s3 s4

# 조합
python run_all_experiments.py --systems ours b1 --scenarios s1 s5 s7

# 실행 계획만 확인 (실제 실행 안 함)
python run_all_experiments.py --dry-run

# 첫 실패에서 중단
python run_all_experiments.py --stop-on-fail
```

---

## 7. Step 5: Ablation Study

논문의 Ablation Study는 시스템 변형 간 비교를 통해 각 구성요소의 기여를 측정합니다.

### 7.1 비교 실험 실행

```bash
# 동일 시나리오를 4개 시스템으로 각각 실행
for sys in ours b1 b2 b3; do
  for sc in s1 s2 s3 s4 s5 s6 s7 s8; do
    echo "Running $sys x $sc..."
    python run_experiment.py --system $sys --scenario $sc --quiet
  done
done
```

또는 간단히:

```bash
python run_all_experiments.py --systems ours b1 b2 b3 --csv
```

### 7.2 Ablation 결과 비교 스크립트

```bash
python -c "
import json, glob

systems = {}
for f in sorted(glob.glob('experiments/summaries/summary_latest.json')):
    data = json.load(open(f))
    for sk, s in data.get('summary_by_system', {}).items():
        systems[sk] = s

print(f'{\"System\":<8} {\"AC@1\":>8} {\"AC@3\":>8} {\"Path\":>8} {\"Avg(s)\":>8}')
print('-' * 42)
for sk in ['ours', 'b1', 'b2', 'b3']:
    s = systems.get(sk)
    if not s: continue
    t = max(s['total'], 1)
    print(f'{sk:<8} {s[\"ac_at_1_rate\"]:>7.1%} {s[\"ac_at_3_rate\"]:>7.1%} {s[\"path_accuracy_rate\"]:>7.1%} {s[\"avg_elapsed\"]:>7.1f}')
"
```

**기대 출력 형태** (논문 Table 2):
```
System     AC@1     AC@3     Path    Avg(s)
------------------------------------------
ours      90.0%    90.0%   100.0%     30.9
b1        75.0%    87.5%    85.7%     12.3
b2        82.5%    87.5%    85.7%     25.1
b3        87.5%    87.5%   100.0%     31.4
```

### 7.3 각 Ablation의 논문 해석 포인트

| 비교 | 측정 대상 | 기대 결과 |
|------|----------|----------|
| ours vs b1 | 멀티에이전트 분업의 가치 | ours의 AC@1 ≥ b1 |
| ours vs b2 | MCP 통합의 가치 | ours의 Path Accuracy ≥ b2 |
| ours vs b3 | Verifier의 가치 | ours의 FP 처리 ≥ b3 |
| b1 vs b3 | 아키텍처 vs 검증 | b3가 b1보다 높아야 함 |

---

## 8. Step 6: RCAEval RE2-OB 벤치마크

### 8.1 데이터 준비

```bash
# RCAEval RE2-OB 데이터셋 다운로드
# https://figshare.com/articles/dataset/RCAEval_..._Microservice_Systems/31048672
# 압축 해제 후:
ls <rcaeval_data>/
# cartservice_cpu_1/
# cartservice_cpu_2/
# cartservice_delay_1/
# ... (90개 케이스)
```

### 8.2 실행

```bash
# 전체 실행 (4시스템 × 90케이스 = 360실험, 약 2~3시간)
python run_rcaeval.py --data-dir /path/to/rcaeval_data

# 특정 시스템만
python run_rcaeval.py --data-dir /path/to/rcaeval_data --systems ours,b3

# 특정 결함 유형만 (빠른 테스트)
python run_rcaeval.py --data-dir /path/to/rcaeval_data --fault-types cpu,mem

# 각 (서비스, 결함) 쌍의 첫 번째 반복만 (30케이스)
python run_rcaeval.py --data-dir /path/to/rcaeval_data --first-only

# CSV 포함
python run_rcaeval.py --data-dir /path/to/rcaeval_data --csv

# 실행 계획 확인만
python run_rcaeval.py --data-dir /path/to/rcaeval_data --dry-run
```

### 8.3 결과

```
experiments/rcaeval/
├── ours_cartservice_cpu_1.json    # 개별 결과
├── ...
├── rcaeval_summary_*.json         # 집계 결과
└── rcaeval_results_*.csv          # CSV (논문 표)
```

---

## 9. Step 7: LLM 모드 실험

### 9.1 환경 설정

```bash
# .env 파일 생성 또는 환경변수 설정
echo "USE_LLM_AGENT=true" >> .env
echo "OPENAI_API_KEY=sk-..." >> .env
echo "LLM_MODEL=gpt-4o" >> .env
```

### 9.2 LLM 모드 실행

```bash
# 단일 실험
USE_LLM_AGENT=true python run_experiment.py --system ours --scenario s1

# 전체 실행
USE_LLM_AGENT=true python run_all_experiments.py --csv
```

### 9.3 결정론적 vs LLM 비교

```bash
# 결정론적 모드 (기본값)
python run_all_experiments.py --systems ours --csv
# → experiments/summaries/summary_*.csv 저장

# LLM 모드
USE_LLM_AGENT=true python run_all_experiments.py --systems ours --csv
# → 기존 결과를 덮어쓰므로 먼저 결정론적 결과를 별도 저장할 것

# 권장: 결과 디렉토리를 분리
cp -r experiments/summaries experiments/summaries_deterministic
USE_LLM_AGENT=true python run_all_experiments.py --systems ours --csv
cp -r experiments/summaries experiments/summaries_llm
```

---

## 10. 결과 해석 가이드

### 10.1 개별 실험 결과 JSON 구조

```jsonc
{
  "system": "ours",
  "scenario": "s1",
  "scenario_description": "S1: DB Connection Pool Exhaustion",
  "elapsed_seconds": 30.56,
  "evaluation": {
    "ac_at_1": true,           // Top-1 정확도
    "ac_at_3": true,           // Top-3 정확도
    "path_accuracy": true,     // 전파 경로 일치
    "fp_handled": null,        // S5에서만 해당
    "top_confidence": 0.95,    // 최고 후보 신뢰도
    "verdict": "accepted",     // Verifier 판정
    "ground_truth_root_cause": "user-db",
    "predicted_top_cause_service": "user-db"
  },
  "result": { /* 전체 RCA 결과 */ }
}
```

### 10.2 summary_table CSV 구조 (논문 표 직접 사용)

| system | total | ac_at_1_count | ac_at_1_rate | ac_at_3_count | ac_at_3_rate | path_accuracy_rate | avg_elapsed |
|--------|-------|---------------|--------------|---------------|--------------|-------------------|-------------|
| ours   | 10    | 9             | 0.900        | 9             | 0.900        | 1.000             | 30.92       |
| b1     | 8     | 6             | 0.750        | 7             | 0.875        | 0.857             | 12.30       |

### 10.3 특수 시나리오 해석

**S5 (False Positive)**: `ground_truth_root_cause`가 `null`입니다.
- `fp_handled = true` → 올바르게 거부/낮은 신뢰도
- `fp_handled = false` → 오탐을 잘못 수용

**S7 (Concurrent Faults)**: 복수 ground truth가 있습니다.
- `user-db` 또는 `worker-service` 중 하나를 예측하면 AC@1 = true

### 10.4 결과를 LaTeX 표로 변환

```bash
python -c "
import csv, sys

reader = csv.DictReader(open('experiments/summaries/summary_table_*.csv'))
print(r'\begin{tabular}{lcccc}')
print(r'\toprule')
print(r'System & AC@1 & AC@3 & Path Acc. & Avg. Time \\\\')
print(r'\midrule')
for row in reader:
    name = {'ours': r'\textbf{Ours}', 'b1': 'B1 (Mono.)', 'b2': 'B2 (No-MCP)', 'b3': 'B3 (No-Ver.)'}
    sys_name = name.get(row['system'], row['system'])
    ac1 = float(row['ac_at_1_rate']) * 100
    ac3 = float(row['ac_at_3_rate']) * 100
    path = float(row['path_accuracy_rate']) * 100
    elapsed = float(row['avg_elapsed'])
    print(f'{sys_name} & {ac1:.1f}\% & {ac3:.1f}\% & {path:.1f}\% & {elapsed:.1f}s \\\\\\\\')
print(r'\bottomrule')
print(r'\end{tabular}')
" 2>/dev/null || echo "(CSV 파일 생성 후 실행하세요)"
```

---

## 11. 트러블슈팅

### 11.1 포트 충돌

```bash
# 이전 실험의 프로세스가 남아있을 때
pkill -f "agents.orchestrator.main"
pkill -f "agents.log_agent.main"
pkill -f "agents.topology_agent.main"
pkill -f "agents.rca_agent.main"
pkill -f "agents.verifier_agent.main"
pkill -f "agents.monolithic.main"
```

### 11.2 실험 중 에이전트 실패

Orchestrator는 graceful degradation을 지원합니다. 에이전트 하나가 실패해도 나머지로 계속 분석합니다. 결과 JSON의 `agent_errors` 필드를 확인하세요:

```bash
python -c "
import json
d = json.load(open('experiments/ours_s1.json'))
errors = d['result'].get('agent_errors')
if errors:
    for e in errors:
        print(f\"  Agent: {e['agent']}, Error: {e['error']}\")
else:
    print('No agent errors')
"
```

### 11.3 이전 결과 초기화

```bash
# 모든 실험 결과 삭제
rm -rf experiments/*.json experiments/summaries/*

# 특정 시스템만
rm experiments/b1_*.json
```

### 11.4 타임아웃 조정

`run_experiment.py`에서 요청 타임아웃은 120초, `run_all_experiments.py`에서 프로세스 타임아웃은 180초입니다. LLM 모드에서 느리면:

```bash
# 환경변수로 A2A 클라이언트 타임아웃 조정은 코드 수정 필요
# agents/orchestrator/a2a_client.py의 timeout 파라미터
```

### 11.5 결정론적 모드 확인

LLM 모드가 의도치 않게 켜져 있을 수 있습니다:

```bash
# 현재 모드 확인
echo $USE_LLM_AGENT  # 비어있거나 "false"여야 결정론적 모드

# 명시적으로 결정론적 모드 강제
USE_LLM_AGENT=false python run_experiment.py --system ours --scenario s1
```

---

## 빠른 참조: 논문에 필요한 최소 실행 명령

```bash
# 1) 검증
python test_tcb_rca.py
python test_end_to_end.py

# 2) 전체 실험 + CSV
python run_all_experiments.py --csv

# 3) 결과 확인
cat experiments/summaries/summary_latest.json | python -m json.tool | head -50

# 끝! experiments/summaries/summary_table_*.csv를 논문에 사용하세요.
```
