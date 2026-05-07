import re
import time
from typing import Any
from uuid import UUID

from anyio import to_thread
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.state import AgentState
from app.core.config import get_settings
from app.schemas import AgentRunResponse, AgentToolCall, AgentTrace
from app.tools.playwright_tools import get_playwright_tools
from app.tools.job_api import check_job_api_config, search_adzuna_jobs, search_google_jobs

SYSTEM_PROMPT = """You are CareerAgent Pro, a job-search browser agent.

You help users search for jobs and inspect job pages.

Use tools when useful:
- Use check_job_api_config if job API credentials seem missing or broken.
- For job-search requests, use all available job discovery sources in a balanced order: first search_adzuna_jobs, then search_google_jobs when configured/valid, then Playwright portal tools for StepStone, Indeed, XING, and job detail verification.
- Always try search_adzuna_jobs for job searches unless it already failed in this run. Sometime It might not give response for a country based query, in that case try it with differnt cities multiple times.
- Use search_google_jobs when broader coverage is useful or the user asks for Google/LinkedIn/Indeed/company-site style coverage. If it returns a 401 or invalid key error, stop using it for this run.
- If a tool returns an error, do not call the same tool again with the same input.
- If a site or portal fails, blocks access, times out, or returns unusable results, do not keep using that same site for the same search. Move to another source.
- If search_google_jobs returns a 401 or invalid key error, report the key issue and stop using that tool.
- If a user asks for links only, return only job links, one per line.
- If CV context is provided, do not return links only. Return scored job objects.
- Never return portal homepages, generic search pages, category pages, or company landing pages as job results. Only return direct job posting URLs for individual roles.
- If you only found a portal/search page, continue searching or say no direct job links were found for that source.
- For every job-search request, combine API tools and Playwright/browser tools. Do not rely only on browser tools when API tools are available.
- Use API tools for structured initial discovery and Playwright tools to expand coverage, inspect portal pages, collect direct links, and read job descriptions.
- LangGraph has access to the full LangChain Playwright browser toolkit, not just one browser tool. Use the browser tools as a real browser: navigate pages, extract text, extract hyperlinks, inspect elements, click controls, move back, and open job detail pages when needed.
- Use search_portal_jobs for StepStone, Indeed, and XING job discovery. This is the preferred high-level browser helper because it searches portal result pages and filters direct job-posting URLs.
- When using search_portal_jobs, request multiple result pages with max_pages and enough max_detail_pages to inspect job descriptions. Use the returned job description text for ranking, match reasons, and cover letters.
- Use navigate_browser to open search engines, job portals, and job pages.
- Do not just navigate to portal homepages like stepstone.de, indeed.com, or xing.com and stop. A homepage visit is not a job search.
- For portal searches, actively use Playwright browser tools on StepStone, Indeed, and XING when relevant. Search each portal directly with search_portal_jobs, extract job links, then move on.
- Use extract_hyperlinks or get_elements to discover job cards/links when a page has search results.
- Use extract_text after navigation or clicking to read visible page content and job descriptions.
- Use click_element when you need to open a specific result, next-page button, cookie banner, filter, or control.
- Use previous_webpage/current_webpage style navigation tools when useful to recover context.
- For job searches, always search entry-level/junior/new graduate variants first, then mid-level, then senior variants only after those. Make enough tool calls to collect a useful set of direct job postings.
- Do not repeat the same tool call with the same portal, query, and location. If results are weak, change one thing: portal, location, seniority, query wording, or source.
- Do not repeat the same API query if it returns no results. Try a broader query, a different seniority, a different location, or switch tools.
- If search_portal_jobs returns no jobs for one portal/query, do not call the same portal/query again. Try a different portal or query variant.
- Prefer result pages and links that clearly include a job title, company, and role-specific URL. Discard duplicate links and generic portal URLs.
- Prefer jobs that include extracted description text. If a job only has a URL and no description, score it cautiously and mark the missing job description as unclear.
- Aim for at least 20 distinct job links when the user asks for a job list. More is better when the links are relevant, unless tools fail or fewer relevant jobs are available.

Keep responses short and practical. Return useful links and extracted details when tools provide them.
Ask one concise follow-up only when the user's request is too vague to act on.

Response Format:
Return valid JSON only. Do not wrap the answer in markdown or code fences. Only keep the valid job links not the portal homepages or search pages. 
The final answer must be one JSON object with this shape:
{
  "summary": "One short sentence that says how many useful job links were found and how they were scored or filtered.",
  "jobs": [
    {
      "job_link": "URL of the job posting",
      "title": "job title if known",
      "company": "company if known",
      "match_score": 0-10,
      "matched_cv_evidence": ["short reasons based on CV"],
      "missing_or_unclear": ["important missing or unclear requirements"],
      "extracted_details": "short job summary",
      "cover_letter": "A short tailored cover letter, 120-180 words, based on the CV and job details."
    }
  ]
}

When CV context is provided, score each position from 0 to 10 based on overlap between the CV and the job title/description/requirements. Be honest: 10 means excellent fit, 5 means partial fit, 0 means no clear fit.
If job descriptions are short or incomplete, still provide a provisional score and put uncertainty in missing_or_unclear.
When CV context is provided, write a short cover letter for each returned job using only facts found in the CV plus the job title/company/requirements. If the candidate's first name is clearly present in the CV, use that first name in the greeting or opening. Never add false information, never invent experience, projects, tools, degrees, locations, dates, achievements, or eligibility. If something is not in the CV, do not claim it.
"""


class AgentTraceCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self._tool_starts: dict[UUID, float] = {}
        self.tool_call_time_ms = 0

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_starts[run_id] = time.perf_counter()

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        started_at = self._tool_starts.pop(run_id, None)
        if started_at is not None:
            self.tool_call_time_ms += round((time.perf_counter() - started_at) * 1000)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.on_tool_end(error, run_id=run_id, **kwargs)


class AIJobsAgent:
    def __init__(self, agent_id: str = "default") -> None:
        settings = get_settings()
        self.agent_id = agent_id
        self.tools = [check_job_api_config, search_adzuna_jobs, search_google_jobs]
        self.agent = None
        self.settings = settings

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
        graph.add_node("after_tools", self._after_tools)
        graph.add_node("finalize", self._finalize_response)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_llm,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "after_tools")
        graph.add_conditional_edges(
            "after_tools",
            self._route_after_tools,
            {"agent": "agent", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
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

    def _after_tools(self, state: AgentState) -> AgentState:
        return {"search_rounds": state.get("search_rounds", 0) + 1}

    def _finalize_response(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(
                content=(
                    f"{SYSTEM_PROMPT}\n\n"
                    "You must now produce the final answer. Do not call tools. "
                    "Use the tool results already available in the conversation."
                )
            ),
            *state["messages"],
        ]
        response = self.llm_model.invoke(messages)
        return {"messages": [response]}

    def _route_after_llm(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    def _route_after_tools(self, state: AgentState) -> str:
        search_rounds = state.get("search_rounds", 0)
        if search_rounds >= 5:
            return "finalize"

        links = self._count_distinct_links(state["messages"])
        if links >= 20:
            return "finalize"

        return "agent"

    async def run(
        self,
        user_input: str,
        thread_id: str = "default",
        cv_text: str | None = None,
    ) -> AgentRunResponse:
        return await to_thread.run_sync(self.run_sync, user_input, thread_id, cv_text)

    def run_sync(
        self,
        user_input: str,
        thread_id: str = "default",
        cv_text: str | None = None,
    ) -> AgentRunResponse:
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
            prompt = self._build_user_prompt(user_input, cv_text)
            trace_callback = AgentTraceCallback()
            started_at = time.perf_counter()
            state = self.agent.invoke(
                {"messages": [HumanMessage(content=prompt)], "search_rounds": 0},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 22,
                    "callbacks": [trace_callback],
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

        total_time_ms = round((time.perf_counter() - started_at) * 1000)
        return self._to_response(state, total_time_ms, trace_callback.tool_call_time_ms)

    def _build_user_prompt(self, user_input: str, cv_text: str | None) -> str:
        if not cv_text:
            return user_input

        return (
            f"{user_input}\n\n"
            "CV CONTEXT FOR MATCH SCORING:\n"
            f"{cv_text[:18000]}\n\n"
            "For this job search, use the API tools first for structured discovery: search_adzuna_jobs, then search_google_jobs if configured and valid. "
            "For this job search, use Playwright/browser tools to collect more jobs from portals. "
            "Prefer search_portal_jobs over generic navigate_browser for StepStone, Indeed, and XING. "
            "You also have the full Playwright toolkit through LangGraph, including navigation, hyperlink extraction, text extraction, element inspection, clicking, and page history tools. "
            "Do not only open portal homepages; search direct job result pages and collect direct job posting URLs. "
            "Use max_pages to inspect multiple result pages and max_detail_pages to open job pages for descriptions. "
            "Search entry-level, junior, trainee, and new-graduate variants first before senior variants. "
            "Return only direct job posting URLs, not portal homepages or generic search result pages. "
            "If one portal/query does not produce direct job links, switch portal or query instead of repeating it. "
            "Use the CV context to score every returned job from 0 to 10. "
            "For each job, explain matched CV evidence and missing or unclear requirements. "
            "For each job, write a short tailored cover_letter using only facts from the CV plus the job title/company/requirements. "
            "If the candidate first name is clearly present in the CV, use it in the greeting or opening. "
            "Never add false information or invent experience, projects, tools, degrees, dates, achievements, or eligibility. "
            "Return one JSON object only with summary and jobs. Do not use markdown code fences."
        )

    def _to_response(
        self,
        state: AgentState,
        total_time_ms: int = 0,
        tool_call_time_ms: int = 0,
    ) -> AgentRunResponse:
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

        trace = self._build_trace(messages, tool_calls, total_time_ms, tool_call_time_ms)
        return AgentRunResponse(result=str(result), tool_calls=tool_calls, trace=trace)

    def _count_distinct_links(self, messages: list[Any]) -> int:
        links: set[str] = set()
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            content = str(message.content)
            for match in re.findall(r"https?://[^\s\"'<>]+", content):
                link = match.rstrip(".,);]")
                if self._looks_like_direct_job_link(link):
                    links.add(link)
        return len(links)

    def _looks_like_direct_job_link(self, link: str) -> bool:
        lowered = link.lower()
        if any(blocked in lowered for blocked in ["/jobs/search", "/jobs?q=", "xing.com/jobs/search"]):
            return False

        return any(
            marker in lowered
            for marker in [
                "stepstone.de/stellenangebote--",
                "stepstone.de/job/",
                "stepstone.de/jobs/stellenangebote-",
                "indeed.com/viewjob",
                "de.indeed.com/viewjob",
                "indeed.com/rc/clk",
                "de.indeed.com/rc/clk",
                "xing.com/jobs/",
            ]
        )

    def _build_trace(
        self,
        messages: list[Any],
        tool_calls: list[AgentToolCall],
        total_time_ms: int,
        tool_call_time_ms: int,
    ) -> AgentTrace:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        for message in messages:
            if not isinstance(message, AIMessage):
                continue

            usage = getattr(message, "usage_metadata", None) or {}
            if usage:
                input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens += int(usage.get("output_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)
                continue

            token_usage = getattr(message, "response_metadata", {}).get("token_usage", {})
            input_tokens += int(token_usage.get("prompt_tokens") or 0)
            output_tokens += int(token_usage.get("completion_tokens") or 0)
            total_tokens += int(token_usage.get("total_tokens") or 0)

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        cost = (
            (input_tokens / 1_000_000) * self.settings.trace_input_cost_per_1m_tokens
            + (output_tokens / 1_000_000) * self.settings.trace_output_cost_per_1m_tokens
        )

        return AgentTrace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
            tool_call_count=len(tool_calls),
            tool_call_time_ms=tool_call_time_ms,
            total_time_ms=total_time_ms,
            model=self.settings.openai_model,
        )
