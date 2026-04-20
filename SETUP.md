# 환경 구성 가이드 (Setup Guide with uv)

> 이 프로젝트는 **uv**(https://docs.astral.sh/uv/)를 권장합니다.
> uv는 기존 pip/venv보다 10~100배 빠른 Python 패키지 관리자이며, 재현성이 높습니다.

---

## 목차

1. [uv 설치](#1-uv-설치)
2. [가상환경 생성 및 활성화](#2-가상환경-생성-및-활성화)
3. [의존성 설치](#3-의존성-설치)
4. [일상 사용 명령어](#4-일상-사용-명령어)
5. [LLM 모드 추가 설정](#5-llm-모드-추가-설정)
6. [논문 실험 실행 예시](#6-논문-실험-실행-예시)
7. [대안: 전통적 pip/venv 방식](#7-대안-전통적-pipvenv-방식)
8. [환경 초기화 및 재생성](#8-환경-초기화-및-재생성)
9. [.gitignore 주요 항목](#9-gitignore-주요-항목)

---

## 1. uv 설치

### macOS / Linux

```bash
# 공식 설치 스크립트 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew
brew install uv

# 또는 pipx
pipx install uv
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 설치 확인

```bash
uv --version
# 예: uv 0.5.x
```

---

## 2. 가상환경 생성 및 활성화

### 2.1 최초 한 번 — `.venv` 생성

프로젝트 루트(`pyproject.toml`이 있는 곳)에서:

```bash
cd /path/to/rca_final

# Python 3.11 이상 필요. 버전 지정 가능:
uv venv --python 3.11       # Python 3.11
uv venv --python 3.12       # Python 3.12
uv venv                     # 시스템 기본 Python 사용

# 결과: .venv/ 디렉터리 생성됨 (이 디렉터리는 .gitignore로 제외됨)
```

### 2.2 매번 — 활성화

**macOS / Linux (bash, zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```

**Fish shell:**
```fish
source .venv/bin/activate.fish
```

### 2.3 활성화 확인

```bash
# 프롬프트 앞에 (rca_final) 또는 (.venv)가 붙어야 함
# Python 경로 확인
which python
# 예상: /path/to/rca_final/.venv/bin/python

python --version
# Python 3.11.x 이상
```

### 2.4 비활성화

```bash
deactivate
```

---

## 3. 의존성 설치

### 3.1 기본 설치 (결정론적 모드만)

```bash
# 활성화된 상태에서:
uv pip install -r requirements.txt

# 또는 pyproject.toml 기반:
uv pip install -e .
```

### 3.2 평가용 추가 패키지 (numpy, pandas)

```bash
uv pip install -e ".[eval]"
```

### 3.3 개발용 추가 패키지 (pytest)

```bash
uv pip install -e ".[dev]"
```

### 3.4 모든 추가 패키지

```bash
uv pip install -e ".[eval,dev]"
```

### 3.5 LLM 모드 추가 (선택)

```bash
# LLM 기능 사용 시
uv pip install openai>=1.50
```

### 3.6 설치 확인

```bash
python -c "import fastapi, httpx, pydantic, uvicorn, dotenv; print('All deps OK')"
```

---

## 4. 일상 사용 명령어

### 4.1 가상환경 안에서 스크립트 실행

```bash
# 활성화된 상태에서:
python test_tcb_rca.py
python run_demo.py
python run_experiment.py --system ours --scenario s1
```

### 4.2 활성화 없이 실행 (uv run)

```bash
# uv가 자동으로 .venv를 사용
uv run python test_tcb_rca.py
uv run python run_demo.py
uv run python run_experiment.py --system ours --scenario s1
uv run python run_all_experiments.py --csv
```

### 4.3 패키지 추가

```bash
# 활성화된 상태에서:
uv pip install <package>

# pyproject.toml에도 반영하려면 수동 편집 후:
uv pip install -e .
```

### 4.4 설치된 패키지 목록

```bash
uv pip list
uv pip freeze > requirements_snapshot.txt  # 현재 스냅샷 저장 (커밋 X)
```

---

## 5. LLM 모드 추가 설정

### 5.1 OpenAI API 키 설정

프로젝트 루트에 `.env` 파일 생성 (**이 파일은 `.gitignore`로 제외됨**):

```bash
cat > .env << 'EOF'
USE_LLM_AGENT=true
OPENAI_API_KEY=sk-...your-key-here...
LLM_MODEL=gpt-4o
EOF
```

### 5.2 LLM 의존성 설치

```bash
uv pip install openai>=1.50
```

### 5.3 LLM 모드로 실행

```bash
# .env가 자동 로드됨
uv run python run_demo.py

# 또는 일회성 환경변수로
USE_LLM_AGENT=true uv run python run_experiment.py --system ours --scenario s1
```

---

## 6. 논문 실험 실행 예시

완전한 실험 재현 워크플로우:

```bash
# 1. 프로젝트로 이동
cd rca_final

# 2. 가상환경 생성 (최초 한 번만)
uv venv --python 3.11

# 3. 활성화
source .venv/bin/activate

# 4. 의존성 설치 (평가용 포함)
uv pip install -r requirements.txt

# 5. 사전 검증
python test_tcb_rca.py
python test_end_to_end.py

# 6. 전체 실험 (34개, ~15-25분)
python run_all_experiments.py --csv

# 7. 논문 표 생성
python show_results.py --latex --per-scenario

# 8. (선택) LLM 모드 실험
uv pip install openai
# .env 파일에 OPENAI_API_KEY 설정
python run_all_experiments.py --systems ours --csv
```

---

## 7. 대안: 전통적 pip/venv 방식

uv를 쓸 수 없는 환경에서는 표준 `venv`를 사용할 수 있습니다:

```bash
# 1. venv 생성
python3.11 -m venv .venv

# 2. 활성화
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\activate             # Windows

# 3. 의존성 설치 (pip 사용)
pip install --upgrade pip
pip install -r requirements.txt

# 4. 이후 사용법은 uv와 동일
python run_demo.py
```

---

## 8. 환경 초기화 및 재생성

### 8.1 완전 초기화

```bash
# 활성화 해제
deactivate

# .venv 삭제
rm -rf .venv

# 캐시 삭제 (선택)
uv cache clean
```

### 8.2 재생성

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 8.3 다른 컴퓨터에서 재현

```bash
# git clone 후
cd rca_final
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 끝. 이제 동일한 환경.
```

---

## 9. .gitignore 주요 항목

다음 항목들이 `.gitignore`에 포함되어 있어 **자동으로 git 추적에서 제외**됩니다:

```
.venv/              # uv가 생성한 가상환경
venv/               # 전통적 venv
uv.lock             # uv 잠금 파일 (논문 artifact에서는 재현을 위해 제외)
.python-version     # pyenv 로컬 버전
__pycache__/        # Python 바이트코드 캐시
*.pyc               # 컴파일된 Python
.env                # API 키 등 민감 정보
llm_logs/           # LLM 호출 로그
.pytest_cache/      # pytest 캐시
.mypy_cache/        # mypy 캐시
.ruff_cache/        # ruff 캐시
```

### 확인 방법

```bash
# .venv/는 git status에 나타나지 않아야 함
git status

# 만약 이미 추적 중이면 (실수로 커밋된 경우):
git rm -r --cached .venv
git rm --cached .env  # 있을 경우
git commit -m "chore: remove tracked build artifacts"
```

### .gitignore 전체 목록 확인

```bash
cat .gitignore
```

---

## 트러블슈팅

### Q. `uv: command not found`
설치 후 셸을 재시작하거나 PATH를 확인하세요:
```bash
echo $PATH | tr ':' '\n' | grep -i uv
# 없으면:
export PATH="$HOME/.local/bin:$PATH"  # 또는 설치 스크립트가 알려준 경로
```

### Q. `python: command not found` (활성화 후)
```bash
# Python이 .venv에 제대로 설치되었는지 확인
ls .venv/bin/python*
# 없으면 재생성:
rm -rf .venv && uv venv --python 3.11
```

### Q. Python 3.11이 시스템에 없을 때
```bash
# uv가 자동으로 Python을 다운로드:
uv python install 3.11
uv venv --python 3.11
```

### Q. 기존 pip 환경과 충돌
```bash
# 완전히 격리된 .venv 사용:
deactivate  # 기존 환경 비활성화
rm -rf .venv
uv venv
source .venv/bin/activate
```

### Q. 포트가 이미 사용 중
```bash
# 이전 실험 프로세스가 남아있을 수 있음
pkill -f "agents.orchestrator"
pkill -f "agents.log_agent"
pkill -f "agents.topology_agent"
pkill -f "agents.rca_agent"
pkill -f "agents.verifier_agent"
pkill -f "agents.monolithic"
```

---

## 참고 문서

- uv 공식 문서: https://docs.astral.sh/uv/
- 프로젝트 구조: [README.md](README.md)
- 실험 실행 가이드: [EXPERIMENTS.md](EXPERIMENTS.md)
- 리팩토링 변경 내역: [CHANGELOG.md](CHANGELOG.md)
