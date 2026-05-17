"""
01_verifier_service_patch.py

목적: agents/verifier_agent/service.py 에 AgentResponse contract 부착 로직을 추가.

이 파일은 *직접 실행되는 코드*가 아니라, 본인이 `agents/verifier_agent/service.py`
를 수정할 때 참고할 패치 명세입니다. 두 가지 변경이 필요합니다.

================================================================================
변경 1: import 추가 (파일 상단, line 25 부근)
================================================================================

기존 (line 22-25):

    from __future__ import annotations

    from typing import Any, Dict, List, Optional, Set, Tuple

변경 후:

    from __future__ import annotations

    import os
    from typing import Any, Dict, List, Optional, Set, Tuple


================================================================================
변경 2: verify() 함수의 return 직전에 contract 부착 (line 594 부근)
================================================================================

기존 (line 587-603):

        final_confidence = revised_candidates[0]["confidence"] if revised_candidates else 0.0
        explanation = (
            verification_notes[0]
            if verification_notes
            else "No verification note available."
        )

        return {
            "incident_id": incident_id,
            "service": service,
            "verdict": verdict,
            "verification_notes": verification_notes,
            "revised_root_cause_candidates": revised_candidates,
            "final_confidence": final_confidence,
            "explanation": explanation,
            "rejected_candidates_count": len(rejected_candidates),
        }

변경 후 (return 직전에 _maybe_attach_verifier_response 호출 추가):

        final_confidence = revised_candidates[0]["confidence"] if revised_candidates else 0.0
        explanation = (
            verification_notes[0]
            if verification_notes
            else "No verification note available."
        )

        result = {
            "incident_id": incident_id,
            "service": service,
            "verdict": verdict,
            "verification_notes": verification_notes,
            "revised_root_cause_candidates": revised_candidates,
            "final_confidence": final_confidence,
            "explanation": explanation,
            "rejected_candidates_count": len(rejected_candidates),
        }

        # >>> ADAPTIVE-FIX: attach AgentResponse contract when A2A_CONTRACT_MODE=on
        _maybe_attach_verifier_response(
            result=result,
            incident_id=incident_id,
            agent_results=agent_results,
        )
        # <<< ADAPTIVE-FIX

        return result


================================================================================
변경 3: 모듈 하단(또는 _derive_final_verdict 위)에 헬퍼 함수 추가
================================================================================

다음 함수를 `class VerifierService` 정의 바로 위, 또는 모듈 하단에 추가:


def _maybe_attach_verifier_response(
    result: Dict[str, Any],
    incident_id: str,
    agent_results: Dict[str, Any],
) -> None:
    \"\"\"Attach AgentResponse to verifier's result dict when A2A_CONTRACT_MODE is on.

    The Orchestrator's _run_adaptive_iterations() reads verifier's _agent_response
    to extract completeness_score and decide whether to re-invoke. Without this
    attachment, adaptive re-invocation cannot trigger.

    We extract the upstream RCA AgentResponse (if present) so that the verifier
    contract inherits the evidence_collection. ConsistencyChecks are left as None
    (the thesis uses the 5-signal scheme; 4D consistency is schema-only).
    \"\"\"
    if os.getenv("A2A_CONTRACT_MODE", "off") == "off":
        return

    try:
        from common.response_builder import (
            build_verifier_agent_response,
            attach_agent_response,
        )
        from common.a2a_contract import AgentResponse

        # Inherit evidence_collection from upstream RCA contract if present.
        # The RCA agent attaches its AgentResponse as _agent_response key.
        upstream_rca_response: Optional[AgentResponse] = None
        rca_result = agent_results.get("rca_agent") if isinstance(agent_results, dict) else None
        if isinstance(rca_result, dict):
            rca_contract = rca_result.get("_agent_response")
            if isinstance(rca_contract, dict):
                try:
                    upstream_rca_response = AgentResponse.model_validate(rca_contract)
                except Exception:
                    upstream_rca_response = None

        agent_resp = build_verifier_agent_response(
            legacy_result=result,
            request_id=incident_id or "UNKNOWN",
            upstream_rca_response=upstream_rca_response,
            consistency_by_service={},  # 5-signal scheme: 4D consistency left empty
        )
        attach_agent_response(result, agent_resp)
    except Exception as exc:
        # Defensive: contract attachment must never break the verifier itself.
        # If it fails, adaptive loop will see None completeness_score and stop —
        # which is the current behavior, so we degrade safely.
        result["_agent_response_error"] = str(exc)
"""
