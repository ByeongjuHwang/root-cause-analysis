"""
End-to-end validation test.

Runs run_demo.py --json and validates the JSON output against expected criteria.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    result = subprocess.run(
        [sys.executable, "run_demo.py", "--json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"FAIL: run_demo.py exited with code {result.returncode}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr[-500:]}")
        return 1

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"FAIL: Cannot parse JSON output: {exc}")
        print(f"STDOUT (last 500 chars):\n{result.stdout[-500:]}")
        return 1

    candidates = payload.get("root_cause_candidates", [])
    verification = payload.get("verification", {})
    impact = payload.get("impact_analysis", {})

    checks = [
        (
            "At least one root cause candidate",
            len(candidates) >= 1,
        ),
        (
            "Top cause mentions user-db (cause or cause_service)",
            candidates and (
                "user-db" in candidates[0].get("cause", "")
                or candidates[0].get("cause_service") == "user-db"
            ),
        ),
        (
            "Verification not rejected",
            verification.get("verdict") != "rejected",
        ),
        (
            "Propagation path starts with user-db",
            impact.get("propagation_path", [None])[0] == "user-db",
        ),
        (
            "api-gateway in affected services",
            "api-gateway" in impact.get("affected_services", []),
        ),
    ]

    failed = [desc for desc, ok in checks if not ok]
    if failed:
        for desc in failed:
            print(f"FAIL: {desc}")
        print(f"\nFull output:\n{json.dumps(payload, indent=2)[:2000]}")
        return 1

    print("PASS: end-to-end demo validation succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
