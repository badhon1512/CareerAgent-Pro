from anyio import to_thread
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.state import AgentState
from app.core.config import get_settings
from app.schemas import AgentRunResponse, AgentToolCall
from app.tools.playwright_tools import get_playwright_tools
from app.tools.job_api import check_job_api_config, search_adzuna_jobs, search_google_jobs

SYSTEM_PROMPT = """You are ApplyPilot AI, a job-search browser agent.

You help users search for jobs and inspect job pages.

Use tools when useful:
- Use check_job_api_config if job API credentials seem missing or broken.
- Prefer search_adzuna_jobs for job search requests.
- Use search_google_jobs when broader coverage is useful or the user asks for Google/LinkedIn/Indeed/company-site style coverage.
- If a tool returns an error, do not call the same tool again with the same input.
- If search_google_jobs returns a 401 or invalid key error, report the key issue and stop using that tool.
- If a user asks for links only, return only job links, one per line.
- Use navigate_browser to open search engines, job portals, and job pages.
- Use extract_text after navigation to read visible page content.
- Use click_element when you need to open a specific result or control.

Keep responses short and practical. Return useful links and extracted details when tools provide them.
Ask one concise follow-up only when the user's request is too vague to act on.

Response Format:
[{job_link: "URL of the job posting if found, else null", extracted_details: "Key details about the job from the page, or null if not applicable"}]
"""


class AIJobsAgent:
    def __init__(self, agent_id: str = "default") -> None:
        settings = get_settings()
        self.agent_id = agent_id
        self.tools = [check_job_api_config, search_adzuna_jobs, search_google_jobs]
        self.agent = None

        if not settings.openai_api_key or not settings.openai_api_key.strip():
            self.missing_api_key = True
            return

        self.missing_api_key = False
        self.llm_model = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self.checkpointer = InMemorySaver()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._run_llm)
        graph.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_llm,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=self.checkpointer)

    def _ensure_agent(self) -> None:
        if self.agent is not None:
            return
        

        self.tools.extend(get_playwright_tools())
        self.llm = self.llm_model.bind_tools(self.tools)
        self.agent = self._build_graph()

    def _run_llm(self, state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _route_after_llm(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    async def run(self, user_input: str, thread_id: str = "default") -> AgentRunResponse:
        return await to_thread.run_sync(self.run_sync, user_input, thread_id)

    def run_sync(self, user_input: str, thread_id: str = "default") -> AgentRunResponse:
        if self.missing_api_key or self.agent is None:
            if self.missing_api_key:
                return AgentRunResponse(
                    result=(
                        "Agent is ready, but OPENAI_API_KEY is not configured. Add it to "
                        "backend/.env, then restart the server."
                    ),
                    tool_calls=[],
                )

        try:
            self._ensure_agent()
        except Exception as exc:
            error_message = repr(exc)
            return AgentRunResponse(
                result=(
                    "Agent is ready, but Playwright could not start the browser toolkit. "
                    "Check Chromium installation and browser launch permissions."
                ),
                tool_calls=[
                    AgentToolCall(
                        name="playwright_toolkit",
                        input={"input": user_input},
                        output={"error": error_message},
                    )
                ],
            )

        try:
            state = self.agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 6,
                },
            )
        except Exception as exc:
            error_message = repr(exc)
            return AgentRunResponse(
                result=(
                    "The LangGraph agent is wired, but the model or browser tool call "
                    "failed during execution. Check OPENAI_API_KEY, OPENAI_MODEL, "
                    "network access, and Playwright browser installation."
                ),
                tool_calls=[
                    AgentToolCall(
                        name="langgraph_agent",
                        input={"input": user_input},
                        output={"error": error_message},
                    )
                ],
            )

        return self._to_response(state)

    def _to_response(self, state: AgentState) -> AgentRunResponse:
        messages = state["messages"]
        final_message = messages[-1]
        result = getattr(final_message, "content", "") or "Done."

        tool_calls: list[AgentToolCall] = []
        tool_call_indexes: dict[str, int] = {}
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    call_id = call.get("id")
                    tool_call_indexes[call_id] = len(tool_calls)
                    tool_calls.append(
                        AgentToolCall(
                            name=call["name"],
                            input=dict(call.get("args") or {}),
                            output={},
                        )
                    )
            elif isinstance(message, ToolMessage):
                index = tool_call_indexes.get(message.tool_call_id)
                if index is not None:
                    tool_calls[index].output = {"content": message.content}

        return AgentRunResponse(result=str(result), tool_calls=tool_calls)
