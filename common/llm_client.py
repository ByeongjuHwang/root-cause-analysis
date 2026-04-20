"""
LLM Client — gpt-5-nano 호출을 위한 공통 모듈.

주요 기능:
1. OpenAI SDK 래퍼 (gpt-5-nano 최적화)
2. 프롬프트/응답 자동 로깅 (재현성 확보)
3. 재시도 로직 (네트워크 에러 대응)
4. 토큰 사용량 추적 (비용 모니터링)
5. JSON 응답 파싱 및 에러 핸들링

gpt-5-nano 제약 사항 반영:
- temperature 파라미터 미지원 → 사용 안 함
- reasoning_effort="low" 기본
- max_completion_tokens 사용 (max_tokens 아님)
- seed 파라미터로 재현성 확보

사용 예:
    from common.llm_client import LLMClient
    
    client = LLMClient(log_dir="llm_logs")
    result = await client.call_json(
        agent_name="log_agent",
        system_prompt="You are a log analysis expert...",
        user_prompt="Analyze these logs...",
        schema={"anomalies": "list", "hypothesis": "string"},
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")
DEFAULT_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
DEFAULT_SEED = int(os.getenv("LLM_SEED", "42"))
DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))
DEFAULT_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# 가격 (USD per 1M tokens) - gpt-5-nano 기준
PRICE_INPUT_PER_1M = 0.05
PRICE_OUTPUT_PER_1M = 0.40


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class LLMCallRecord:
    """단일 LLM 호출의 전체 기록. 재현성과 디버깅 용도."""
    # Metadata
    call_id: str
    timestamp: str
    agent_name: str
    incident_id: Optional[str]
    
    # Request
    model: str
    reasoning_effort: str
    seed: int
    max_completion_tokens: int
    system_prompt: str
    user_prompt: str
    
    # Response
    raw_response: str
    parsed_response: Optional[Dict[str, Any]]
    parse_error: Optional[str]
    
    # Usage
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    
    # Performance
    elapsed_seconds: float
    retry_count: int
    
    # Error (if failed)
    error: Optional[str] = None


# =========================================================================
# LLM Client
# =========================================================================

class LLMClient:
    """OpenAI gpt-5-nano 래퍼 — 에이전트에서 공통 사용."""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        seed: int = DEFAULT_SEED,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        log_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        
        # OpenAI SDK는 동적 import (의존성 설치 전에도 모듈 import 가능하도록)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai 패키지가 필요합니다. 'pip install openai' 또는 "
                "'uv pip install openai'로 설치하세요."
            ) from e
        
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=timeout,
        )
        
        # 로깅 디렉토리 설정
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 세션 통계
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_tokens = 0
        self._total_calls = 0
        self._total_errors = 0
    
    # ---- Public API ----
    
    async def call_json(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        incident_id: Optional[str] = None,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """LLM 호출 후 JSON 응답 파싱.
        
        Args:
            agent_name: 로깅용 에이전트 이름 (예: "log_agent")
            system_prompt: 시스템 프롬프트 (역할 정의)
            user_prompt: 사용자 프롬프트 (입력 데이터 + 질의)
            incident_id: 인시던트 ID (로깅용)
            schema_hint: JSON 스키마 힌트 (프롬프트에 포함됨)
        
        Returns:
            파싱된 JSON dict. 파싱 실패 시 {"_error": "...", "_raw": "..."}
        """
        # 스키마 힌트를 user prompt 끝에 추가
        full_user_prompt = user_prompt
        if schema_hint:
            full_user_prompt += f"\n\n출력 형식 (JSON만 출력하고 다른 텍스트는 포함하지 마세요):\n{schema_hint}"
        
        record = await self._call_with_retry(
            agent_name=agent_name,
            system_prompt=system_prompt,
            user_prompt=full_user_prompt,
            incident_id=incident_id,
        )
        
        if record.error:
            logger.error(f"[{agent_name}] LLM call failed: {record.error}")
            return {"_error": record.error, "_raw": record.raw_response}
        
        return record.parsed_response or {"_error": record.parse_error, "_raw": record.raw_response}
    
    # ---- 통계 / 리포팅 ----
    
    def get_stats(self) -> Dict[str, Any]:
        """세션 전체의 토큰 사용량과 비용 통계."""
        cost = (
            self._total_input_tokens * PRICE_INPUT_PER_1M / 1_000_000
            + self._total_output_tokens * PRICE_OUTPUT_PER_1M / 1_000_000
        )
        return {
            "model": self.model,
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "estimated_cost_usd": round(cost, 4),
        }
    
    def reset_stats(self):
        """통계 초기화."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_tokens = 0
        self._total_calls = 0
        self._total_errors = 0
    
    # ---- Internal ----
    
    async def _call_with_retry(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        incident_id: Optional[str],
    ) -> LLMCallRecord:
        """재시도 로직이 포함된 LLM 호출."""
        call_id = self._generate_call_id(agent_name, user_prompt)
        start_time = time.time()
        last_error = None
        
        for retry in range(self.max_retries):
            try:
                record = await self._call_once(
                    call_id=call_id,
                    agent_name=agent_name,
                    incident_id=incident_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    retry_count=retry,
                )
                
                # 통계 업데이트
                self._total_calls += 1
                self._total_input_tokens += record.input_tokens
                self._total_output_tokens += record.output_tokens
                self._total_reasoning_tokens += record.reasoning_tokens
                
                # 로깅
                if self.log_dir:
                    self._save_record(record)
                
                return record
            
            except Exception as e:
                last_error = e
                wait_time = 2 ** retry  # 지수 백오프
                logger.warning(
                    f"[{agent_name}] LLM call failed (retry {retry+1}/{self.max_retries}): {e}. "
                    f"Waiting {wait_time}s..."
                )
                if retry < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
        
        # 모든 재시도 실패
        self._total_errors += 1
        elapsed = time.time() - start_time
        
        error_record = LLMCallRecord(
            call_id=call_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=agent_name,
            incident_id=incident_id,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            seed=self.seed,
            max_completion_tokens=self.max_completion_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response="",
            parsed_response=None,
            parse_error=None,
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            elapsed_seconds=elapsed,
            retry_count=self.max_retries,
            error=str(last_error),
        )
        
        if self.log_dir:
            self._save_record(error_record)
        
        return error_record
    
    async def _call_once(
        self,
        call_id: str,
        agent_name: str,
        incident_id: Optional[str],
        system_prompt: str,
        user_prompt: str,
        retry_count: int,
    ) -> LLMCallRecord:
        """단일 LLM 호출."""
        start_time = time.time()
        
        # gpt-5-nano 호출
        # 주의: temperature 파라미터는 지원 안 함
        # reasoning_effort와 max_completion_tokens 사용
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_completion_tokens,
            "seed": self.seed,
        }
        
        # gpt-5 계열은 reasoning_effort 지원. 다른 모델은 무시
        if self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3"):
            kwargs["reasoning_effort"] = self.reasoning_effort
        
        response = await self.client.chat.completions.create(**kwargs)
        
        elapsed = time.time() - start_time
        
        # 응답 추출
        raw_response = response.choices[0].message.content or ""
        
        # 토큰 사용량
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        # reasoning_tokens는 gpt-5 계열에서 completion_tokens에 포함되지만
        # 별도로 reported되기도 함
        reasoning_tokens = 0
        if usage and hasattr(usage, 'completion_tokens_details'):
            details = usage.completion_tokens_details
            if details and hasattr(details, 'reasoning_tokens'):
                reasoning_tokens = details.reasoning_tokens or 0
        
        # 비용 계산
        cost = (
            input_tokens * PRICE_INPUT_PER_1M / 1_000_000
            + output_tokens * PRICE_OUTPUT_PER_1M / 1_000_000
        )
        
        # JSON 파싱 시도
        parsed = None
        parse_error = None
        try:
            parsed = self._parse_json_response(raw_response)
        except Exception as e:
            parse_error = str(e)
        
        return LLMCallRecord(
            call_id=call_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=agent_name,
            incident_id=incident_id,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            seed=self.seed,
            max_completion_tokens=self.max_completion_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw_response,
            parsed_response=parsed,
            parse_error=parse_error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=round(cost, 6),
            elapsed_seconds=round(elapsed, 2),
            retry_count=retry_count,
        )
    
    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 추출 및 파싱.
        
        LLM이 종종 ```json ... ``` 코드 블록으로 감싸거나
        앞뒤에 설명을 덧붙이므로 robust하게 파싱.
        """
        if not raw.strip():
            raise ValueError("Empty response")
        
        # 1. 코드 블록 제거
        # ```json ... ``` 또는 ``` ... ```
        code_block_pattern = r'```(?:json)?\s*(.*?)```'
        match = re.search(code_block_pattern, raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
        
        # 2. 직접 JSON 파싱 시도
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        # 3. 첫 { 부터 마지막 } 까지 추출 (앞뒤 텍스트 제거)
        first_brace = raw.find('{')
        last_brace = raw.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            candidate = raw[first_brace:last_brace + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse failed: {e}. Candidate: {candidate[:200]}")
        
        raise ValueError(f"No JSON found in response: {raw[:200]}")
    
    def _generate_call_id(self, agent_name: str, user_prompt: str) -> str:
        """호출 ID 생성 (타임스탬프 + 해시)."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.sha256(user_prompt.encode('utf-8')).hexdigest()[:8]
        return f"{ts}_{agent_name}_{prompt_hash}"
    
    def _save_record(self, record: LLMCallRecord):
        """호출 기록을 JSON 파일로 저장."""
        filename = f"{record.call_id}.json"
        path = self.log_dir / filename
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(asdict(record), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save record {filename}: {e}")


# =========================================================================
# Singleton helper
# =========================================================================

_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """기본 LLM 클라이언트 싱글톤. 에이전트에서 공유 사용."""
    global _default_client
    if _default_client is None:
        log_dir = os.getenv("LLM_LOG_DIR", "llm_logs")
        _default_client = LLMClient(log_dir=log_dir)
    return _default_client


def reset_default_client():
    """싱글톤 초기화 (테스트용)."""
    global _default_client
    _default_client = None
