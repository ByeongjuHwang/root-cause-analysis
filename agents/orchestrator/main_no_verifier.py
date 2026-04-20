"""
B3 Orchestrator — 교차 검증 에이전트를 제거한 오케스트레이터.

원본 main.py와 동일하되 NoVerifierOrchestratorService를 사용하며,
verifier_agent_url이 필요 없다.
"""

import os

import uvicorn
from fastapi import FastAPI

from .models import IncidentRequest
from .service_no_verifier import NoVerifierOrchestratorService

app = FastAPI(title="Orchestrator Agent (B3: No-Verifier)")

service = NoVerifierOrchestratorService(
    log_agent_url=os.getenv("LOG_AGENT_URL", "http://127.0.0.1:19001"),
    topology_agent_url=os.getenv("TOPOLOGY_AGENT_URL", "http://127.0.0.1:19002"),
    rca_agent_url=os.getenv("RCA_AGENT_URL", "http://127.0.0.1:19004"),
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
        "agents.orchestrator.main_no_verifier:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "19000")),
        reload=False,
    )
