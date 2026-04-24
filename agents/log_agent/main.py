
import json
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# LLM 모드와 결정론적 모드를 환경변수로 전환
# USE_LLM_AGENT=true 면 LLM 버전 사용, 기본값은 결정론적 버전
_USE_LLM = os.getenv("USE_LLM_AGENT", "false").lower() in ("true", "1", "yes")

if _USE_LLM:
    from .skills_llm import LogAnalysisServiceLLM as _ServiceClass
    _MODE = "LLM"
else:
    from .skills import LogAnalysisService as _ServiceClass
    _MODE = "deterministic"

app = FastAPI(title=f"Log Analysis Agent ({_MODE})")
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
    method = body.get("method")
    rpc_id = body.get("id")

    if method != "message/send":
        return JSONResponse(status_code=400, content={
            "jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "Unsupported method"}
        })

    params = body.get("params", {})
    message = params.get("message", {})
    metadata = params.get("metadata", {})

    # LLM 버전은 incident_id 파라미터 추가로 받음
    kwargs = {
        "service": metadata.get("service", "unknown-service"),
        "start": metadata.get("start"),
        "end": metadata.get("end"),
        "trace_id": metadata.get("trace_id"),
        "symptom": metadata.get("symptom", ""),
        "log_file": metadata.get("log_file"),
    }
    if _USE_LLM:
        kwargs["incident_id"] = metadata.get("incident_id")
        # v6: asymmetric dual-window analysis. Only pass if both are present.
        baseline_range = metadata.get("baseline_range")
        incident_range = metadata.get("incident_range")
        if baseline_range and incident_range:
            kwargs["baseline_range"] = tuple(baseline_range)
            kwargs["incident_range"] = tuple(incident_range)
        # v8: metrics CSV path for metric tool calls
        metrics_file = metadata.get("metrics_file")
        if metrics_file:
            kwargs["metrics_file"] = metrics_file
        # Phase 4d-2: adaptive focus_services (from orchestrator re-invocation)
        focus_services = metadata.get("focus_services")
        if focus_services and isinstance(focus_services, list):
            kwargs["focus_services"] = [str(s) for s in focus_services if s]

    analysis = await service.analyze(**kwargs)

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
                "artifacts": [{"name": "log-analysis-result", "parts": [{"kind": "data", "data": analysis}]}],
            }
        },
    })


if __name__ == "__main__":
    uvicorn.run("agents.log_agent.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "9001")), reload=False)
