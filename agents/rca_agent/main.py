
import json
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# LLM 모드와 결정론적 모드를 환경변수로 전환
# USE_LLM_AGENT=true 면 LLM 버전 사용, 기본값은 결정론적(TCB-RCA) 버전
_USE_LLM = os.getenv("USE_LLM_AGENT", "false").lower() in ("true", "1", "yes")

if _USE_LLM:
    from .skills_llm import RCAServiceLLM as _ServiceClass
    _MODE = "LLM-Synthesis"
else:
    from .service import RCAService as _ServiceClass
    _MODE = "deterministic-TCB-RCA"

app = FastAPI(title=f"RCA Synthesis Agent ({_MODE})")
service = _ServiceClass()
BASE_DIR = Path(__file__).resolve().parent
AGENT_CARD_PATH = BASE_DIR / "sample_agent_card.json"


@app.get("/health")
async def health():
    return {"status": "ok", "mode": _MODE}


@app.get("/.well-known/agent-card.json")
async def agent_card():
    with AGENT_CARD_PATH.open("r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.post("/a2a")
async def a2a_endpoint(request: Request):
    body = await request.json()
    rpc_id = body.get("id")
    method = body.get("method")
    if method != "message/send":
        return JSONResponse(status_code=400, content={
            "jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "Unsupported method"}
        })

    try:
        params = body.get("params", {})
        message = params.get("message", {})
        metadata = params.get("metadata", {})
        
        # LLM과 결정론적 버전 모두 동일한 인터페이스
        result = await service.synthesize(
            incident_id=metadata.get("incident_id"),
            service=metadata.get("service"),
            log_result=metadata.get("log_result", {}),
            topology_result=metadata.get("topology_result", {}),
        )
        
        task_id = message.get("taskId") or str(uuid.uuid4())
        context_id = message.get("contextId") or str(uuid.uuid4())
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "task": {
                    "id": task_id,
                    "contextId": context_id,
                    "status": {"state": "completed", "timestamp": "2026-03-29T11:00:00+09:00"},
                    "artifacts": [{"name": "rca-synthesis-result", "parts": [{"kind": "data", "data": result}]}],
                }
            },
        })
    except Exception as exc:
        import traceback
        error_detail = f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
        return JSONResponse(status_code=500, content={
            "jsonrpc": "2.0", "id": rpc_id, 
            "error": {"code": -32000, "message": str(exc), "data": error_detail}
        })


if __name__ == "__main__":
    uvicorn.run("agents.rca_agent.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "9004")), reload=False)
