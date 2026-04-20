
import os

import uvicorn
from fastapi import FastAPI

from .models import IncidentRequest
from .service import OrchestratorService

app = FastAPI(title="Orchestrator Agent")

service = OrchestratorService(
    log_agent_url=os.getenv("LOG_AGENT_URL", "http://127.0.0.1:18001"),
    topology_agent_url=os.getenv("TOPOLOGY_AGENT_URL", "http://127.0.0.1:18002"),
    rca_agent_url=os.getenv("RCA_AGENT_URL", "http://127.0.0.1:18004"),
    verifier_agent_url=os.getenv("VERIFIER_AGENT_URL", "http://127.0.0.1:18003"),
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(incident: IncidentRequest):
    result = await service.analyze_incident(incident)
    return result.model_dump()


if __name__ == "__main__":
    uvicorn.run("agents.orchestrator.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "18000")), reload=False)
