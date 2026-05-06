from fastapi import APIRouter

from app.schemas import AgentRunRequest, AgentRunResponse
from app.agents.agent import AIJobsAgent

agent = AIJobsAgent(agent_id="default")
router = APIRouter()


@router.post("/run", response_model=AgentRunResponse)
async def run_browser_agent(payload: AgentRunRequest) -> AgentRunResponse:
    return await agent.run(payload.input)
