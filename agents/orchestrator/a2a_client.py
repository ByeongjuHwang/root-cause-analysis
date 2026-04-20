import uuid
import httpx
from typing import Dict, Any, Optional


class A2AClient:
    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    async def fetch_agent_card(self, base_url: str) -> Dict[str, Any]:
        url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot connect to agent card endpoint: {url}") from e

    async def send_message(
        self,
        agent_base_url: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rpc_id = str(uuid.uuid4())

        message = {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": text,
                }
            ],
            "messageId": str(uuid.uuid4()),
        }

        if task_id:
            message["taskId"] = task_id
        if context_id:
            message["contextId"] = context_id

        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "message/send",
            "params": {
                "message": message,
                "metadata": metadata or {}
            }
        }

        target_url = f"{agent_base_url.rstrip('/')}/a2a"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(target_url, json=payload)

                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"A2A call failed: url={target_url}, "
                        f"status={resp.status_code}, body={resp.text}"
                    )

                return resp.json()
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot connect to agent endpoint: {target_url}") from e