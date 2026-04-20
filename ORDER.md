# 2. 현재 위치 확인 — 아래 파일들이 보여야 정상
Get-ChildItem -Name run_experiment.py, run_all_experiments.py, experiment_core.py, pyproject.toml

# 기대 출력:
#   experiment_core.py
#   pyproject.toml
#   run_all_experiments.py
#   run_experiment.py

# 3. 이전 실험 결과 완전 삭제
if (Test-Path experiments) {
    Remove-Item -Recurse -Force experiments\*.json -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force experiments\summaries -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force experiments\rcaeval -ErrorAction SilentlyContinue
    Write-Host "[OK] 이전 experiments/ 결과 삭제됨"
}

# 4. LLM 호출 로그 삭제 (있었다면)
if (Test-Path llm_logs) {
    Remove-Item -Recurse -Force llm_logs
    Write-Host "[OK] llm_logs/ 삭제됨"
}

# 5. Python 캐시 삭제
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Write-Host "[OK] __pycache__ 정리 완료"


# Agent 관련 Python 프로세스만 찾아서 종료
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "agents\.(orchestrator|log_agent|topology_agent|rca_agent|verifier_agent|monolithic)" } |
    ForEach-Object {
        Write-Host "Killing PID $($_.ProcessId): $($_.CommandLine.Substring(0, [Math]::Min(100, $_.CommandLine.Length)))..."
        Stop-Process -Id $_.ProcessId -Force
    }

Write-Host "[OK] Agent 프로세스 정리 완료"