
import json
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .service import VerifierService

app = FastAPI(title="Verifier Agent")
service = VerifierService()
BASE_DIR = Path(__file__).resolve().parent
AGENT_CARD_PATH = BASE_DIR / "sample_agent_card.json"


@app.get("/health")
async def health():
    return {"status": "ok"}


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

    result = await service.verify(
        incident_id=metadata.get("incident_id"),
        service=metadata.get("service"),
        draft_rca=metadata.get("draft_rca", {}),
        agent_results=metadata.get("agent_results", {}),
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
                "artifacts": [{"name": "verification-result", "parts": [{"kind": "data", "data": result}]}],
            }
        },
    })


if __name__ == "__main__":
    uvicorn.run("agents.verifier_agent.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "9003")), reload=False)
