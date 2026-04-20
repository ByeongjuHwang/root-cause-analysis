"""
B2 Parallel Orchestrator — 에이전트를 병렬 독립 실행하는 오케스트레이터.
"""

import os

import uvicorn
from fastapi import FastAPI

from .models import IncidentRequest
from .service_parallel import ParallelOrchestratorService

app = FastAPI(title="Orchestrator Agent (B2: Parallel Independent)")

service = ParallelOrchestratorService(
    log_agent_url=os.getenv("LOG_AGENT_URL", "http://127.0.0.1:21001"),
    topology_agent_url=os.getenv("TOPOLOGY_AGENT_URL", "http://127.0.0.1:21002"),
    rca_agent_url=os.getenv("RCA_AGENT_URL", "http://127.0.0.1:21004"),
    verifier_agent_url=os.getenv("VERIFIER_AGENT_URL", "http://127.0.0.1:21003"),
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(incident: IncidentRequest):
    result = await service.analyze_incident(incident)
    return result.model_dump()


if __name__ == "__main__":
    uvicorn.run(
        "agents.orchestrator.main_parallel:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "21000")),
        reload=False,
    )
