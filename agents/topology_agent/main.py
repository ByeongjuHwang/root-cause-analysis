
import json
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# LLM 모드와 결정론적 모드를 환경변수로 전환
_USE_LLM = os.getenv("USE_LLM_AGENT", "false").lower() in ("true", "1", "yes")

if _USE_LLM:
    from .skills_llm import TopologyAnalysisServiceLLM as _ServiceClass
    _MODE = "LLM"
else:
    from .skills import TopologyAnalysisService as _ServiceClass
    _MODE = "deterministic"

app = FastAPI(title=f"Topology Analysis Agent ({_MODE})")
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

    params = body.get("params", {})
    message = params.get("message", {})
    metadata = params.get("metadata", {})

    # LLM 버전은 incident_id 파라미터 추가로 받음
    kwargs = {
        "service": metadata.get("service"),
        "suspected_downstream": metadata.get("suspected_downstream"),
        "diagram_uri": metadata.get("diagram_uri", "arch://system/latest"),
        "topology_file": metadata.get("topology_file"),
    }
    if _USE_LLM:
        kwargs["incident_id"] = metadata.get("incident_id")

    result = await service.analyze(**kwargs)

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
                "artifacts": [{"name": "topology-analysis-result", "parts": [{"kind": "data", "data": result}]}],
            }
        },
    })


if __name__ == "__main__":
    uvicorn.run("agents.topology_agent.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "9002")), reload=False)
