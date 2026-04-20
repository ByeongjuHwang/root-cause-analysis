import sys
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .repository import search_logs as repo_search_logs
from .repository import get_error_summary as repo_get_error_summary
from .repository import get_trace_logs as repo_get_trace_logs


# STDIO transport에서는 stdout에 임의 로그를 쓰면 JSON-RPC가 깨질 수 있으므로 stderr 로깅 사용
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

mcp = FastMCP("observability-mcp")


@mcp.resource("resource://incident-schema", mime_type="application/json")
def incident_schema() -> dict:
    """Return the incident request schema used by the RCA system."""
    return {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string"},
            "service": {"type": "string"},
            "time_range": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"}
                },
                "required": ["start", "end"]
            },
            "symptom": {"type": "string"},
            "trace_id": {"type": ["string", "null"]}
        },
        "required": ["incident_id", "service", "time_range", "symptom"]
    }


@mcp.resource("resource://sample-log-fields", mime_type="application/json")
def sample_log_fields() -> dict:
    """Return the supported log fields for analysis."""
    return {
        "fields": [
            "timestamp",
            "service",
            "level",
            "trace_id",
            "message",
            "upstream",
            "status_code",
            "latency_ms",
            "error_type"
        ]
    }


@mcp.tool()
def search_logs(service: str, start: str, end: str, keyword: Optional[str] = None) -> dict:
    """
    Search logs for a target service within a time range.

    Args:
        service: Target microservice name
        start: ISO-8601 start timestamp
        end: ISO-8601 end timestamp
        keyword: Optional keyword filter
    """
    logging.info("search_logs called: service=%s start=%s end=%s keyword=%s", service, start, end, keyword)

    rows = repo_search_logs(service=service, start=start, end=end, keyword=keyword)
    return {
        "service": service,
        "count": len(rows),
        "logs": [r.model_dump() for r in rows[:100]]
    }


@mcp.tool()
def get_error_summary(service: str, start: str, end: str) -> dict:
    """
    Summarize error patterns for a target service in a time window.

    Args:
        service: Target microservice name
        start: ISO-8601 start timestamp
        end: ISO-8601 end timestamp
    """
    logging.info("get_error_summary called: service=%s start=%s end=%s", service, start, end)

    summary = repo_get_error_summary(service=service, start=start, end=end)
    return summary.model_dump()


@mcp.tool()
def get_trace_logs(trace_id: str) -> dict:
    """
    Retrieve all logs associated with a trace_id.

    Args:
        trace_id: Distributed trace identifier
    """
    logging.info("get_trace_logs called: trace_id=%s", trace_id)

    rows = repo_get_trace_logs(trace_id=trace_id)
    return {
        "trace_id": trace_id,
        "count": len(rows),
        "logs": [r.model_dump() for r in rows[:200]]
    }


def main():
    # 논문 1차 PoC는 로컬 실행이 쉬운 stdio로 시작
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()