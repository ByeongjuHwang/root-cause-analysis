from pydantic import BaseModel, Field
from typing import Optional, List


class LogRecord(BaseModel):
    timestamp: str
    service: str
    level: str
    trace_id: Optional[str] = None
    message: str
    upstream: Optional[str] = None
    status_code: Optional[int] = None
    latency_ms: Optional[int] = None
    error_type: Optional[str] = None


class SearchLogsInput(BaseModel):
    service: str = Field(..., description="Target microservice name")
    start: str = Field(..., description="ISO-8601 start timestamp")
    end: str = Field(..., description="ISO-8601 end timestamp")
    keyword: Optional[str] = Field(default=None, description="Optional keyword filter")


class ErrorSummaryInput(BaseModel):
    service: str
    start: str
    end: str


class TraceLogsInput(BaseModel):
    trace_id: str


class ErrorSummary(BaseModel):
    service: str
    total_logs: int
    error_logs: int
    top_error_types: List[dict]
    top_upstreams: List[dict]