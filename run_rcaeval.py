"""
02_run_rcaeval_patch.py

목적: run_rcaeval.py 에서 A2A_CONTRACT_MODE / ADAPTIVE_THRESHOLD / ADAPTIVE_MAX_ITERATIONS
환경변수를 명시적으로 활성화.

이 파일은 *직접 실행되는 코드*가 아니라, 본인이 run_rcaeval.py 의 service-start 부분을
수정할 때 참고할 패치 명세입니다.

================================================================================
변경 위치: run_rcaeval.py line 381-395 부근 (서비스 기동 시 환경변수 set 하는 곳)
================================================================================

기존 (line 381-395 부근):

    \"\"\"Start all services for a system with appropriate env vars.\"\"\"
    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pypath if pypath else "")
    env["OBSERVABILITY_LOG_FILE"] = str(log_file)
    env["ARCHITECTURE_TOPOLOGY_FILE"] = str(topology_file)
    env.update(build_env_overrides(system_key))

변경 후 (env.update(build_env_overrides(system_key)) 바로 다음에 4줄 추가):

    \"\"\"Start all services for a system with appropriate env vars.\"\"\"
    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pypath if pypath else "")
    env["OBSERVABILITY_LOG_FILE"] = str(log_file)
    env["ARCHITECTURE_TOPOLOGY_FILE"] = str(topology_file)
    env.update(build_env_overrides(system_key))

    # >>> ADAPTIVE-FIX: enable A2A contracts and adaptive re-invocation
    env["A2A_CONTRACT_MODE"] = os.environ.get("A2A_CONTRACT_MODE", "on")
    env["ADAPTIVE_THRESHOLD"] = os.environ.get("ADAPTIVE_THRESHOLD", "0.5")
    env["ADAPTIVE_MAX_ITERATIONS"] = os.environ.get("ADAPTIVE_MAX_ITERATIONS", "3")
    # <<< ADAPTIVE-FIX


================================================================================
중요: 동일한 패치가 line 500 부근의 두 번째 env setup 위치에도 필요합니다
================================================================================

run_rcaeval.py 에는 env를 두 번 만드는 코드가 있습니다:
  - line 381 부근: 서비스 기동용
  - line 500 부근: 분석(analyze) 호출용

두 곳 모두 같은 4줄 추가가 필요합니다.

기존 (line 500 부근):

    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pypath if pypath else "")
    env["ARCHITECTURE_TOPOLOGY_FILE"] = str(topology_file)
    env.update(build_env_overrides(system_key))

변경 후:

    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pypath if pypath else "")
    env["ARCHITECTURE_TOPOLOGY_FILE"] = str(topology_file)
    env.update(build_env_overrides(system_key))

    # >>> ADAPTIVE-FIX: enable A2A contracts and adaptive re-invocation
    env["A2A_CONTRACT_MODE"] = os.environ.get("A2A_CONTRACT_MODE", "on")
    env["ADAPTIVE_THRESHOLD"] = os.environ.get("ADAPTIVE_THRESHOLD", "0.5")
    env["ADAPTIVE_MAX_ITERATIONS"] = os.environ.get("ADAPTIVE_MAX_ITERATIONS", "3")
    # <<< ADAPTIVE-FIX


================================================================================
왜 os.environ.get(...) 으로 감싸는가
================================================================================

Ablation 실험을 위함입니다. 본격 실험 시에는:

  python run_rcaeval.py ...                              # adaptive ON (기본)

Ablation을 위해 adaptive를 끄려면:

  ADAPTIVE_MAX_ITERATIONS=0 python run_rcaeval.py ...    # adaptive OFF

ADAPTIVE_MAX_ITERATIONS=0 이면 _run_adaptive_iterations 의 for 루프가 한 번도
돌지 않아 single-pass 실행이 됩니다. 본인 thesis 의 Adaptive ablation 섹션이
이 한 줄 차이만으로 구해집니다.

A2A_CONTRACT_MODE 만 끄고 싶을 때도:

  A2A_CONTRACT_MODE=off python run_rcaeval.py ...

이렇게 외부에서 set한 값이 우선됩니다.
"""
