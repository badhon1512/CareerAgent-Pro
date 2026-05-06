from typing import Any

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    input: str = Field(
        min_length=2,
        max_length=1000,
        description="The agent instruction, for example: open https://example.com",
    )


class AgentToolCall(BaseModel):
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    result: str
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
