"""
B1 Monolithic Agent — 단일 프로세스 RCA 서비스.

모든 로직을 하나의 HTTP 서버에서 실행. A2A 미사용, MCP 미사용.
"""

import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .service import MonolithicRCAService

app = FastAPI(title="B1 Monolithic RCA Agent")

service = MonolithicRCAService(
    log_file=os.getenv("OBSERVABILITY_LOG_FILE"),
    topology_file=os.getenv("ARCHITECTURE_TOPOLOGY_FILE"),
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: Request):
    incident = await request.json()
    result = await service.analyze_incident(incident)
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run(
        "agents.monolithic.main:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "20000")),
        reload=False,
    )
