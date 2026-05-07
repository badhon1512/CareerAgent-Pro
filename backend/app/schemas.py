from typing import Any

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    input: str = Field(
        min_length=2,
        max_length=1000,
        description="The agent instruction, for example: open https://example.com",
    )
    cv_text: str | None = Field(
        default=None,
        max_length=20000,
        description="Optional extracted CV text used for job match scoring.",
    )


class AgentToolCall(BaseModel):
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0
    tool_call_count: int = 0
    tool_call_time_ms: int = 0
    total_time_ms: int = 0
    model: str | None = None


class AgentRunResponse(BaseModel):
    result: str
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    trace: AgentTrace | None = None


class CVExtractResponse(BaseModel):
    filename: str
    content_type: str | None = None
    text: str
    character_count: int
