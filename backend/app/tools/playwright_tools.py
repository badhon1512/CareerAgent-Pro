from concurrent.futures import Future
from queue import Queue
from threading import Thread
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.sync_api import Browser, Playwright, sync_playwright


class PlaywrightToolkitWorker:
    def __init__(self) -> None:
        self._jobs: Queue[tuple[Callable[[], Any] | None, Future[Any] | None]] = Queue()
        self._thread = Thread(target=self._run, name="playwright-toolkit", daemon=True)
        self._started = False
        self._tools_by_name: dict[str, Any] = {}
        self._browser: Browser | None = None
        self._playwright: Playwright | None = None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()
        self.call(lambda: True)

    def call(self, func: Callable[[], Any]) -> Any:
        future: Future[Any] = Future()
        self._jobs.put((func, future))
        return future.result()

    def invoke_tool(self, tool_name: str, tool_input: Any) -> Any:
        return self.call(lambda: self._tools_by_name[tool_name].invoke(tool_input))

    def close(self) -> None:
        if not self._started:
            return
        future: Future[Any] = Future()
        self._jobs.put((None, future))
        future.result()
        self._started = False

    def _run(self) -> None:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=False,
            slow_mo=300,
            args=["--disable-gpu", "--no-sandbox"],
        )
        toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=self._browser)
        self._tools_by_name = {tool.name: tool for tool in toolkit.get_tools()}

        while True:
            func, future = self._jobs.get()
            if func is None:
                try:
                    self._browser.close()
                    self._playwright.stop()
                    future.set_result(True)
                except Exception as exc:
                    future.set_exception(exc)
                return

            try:
                future.set_result(func())
            except Exception as exc:
                future.set_exception(exc)


_worker = PlaywrightToolkitWorker()
_tools: list[StructuredTool] | None = None


def _make_proxy_tool(tool: Any) -> StructuredTool:
    def run_tool(tool_input: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            payload = kwargs
        elif tool_input is None:
            payload = {}
        else:
            payload = tool_input
        return _worker.invoke_tool(tool.name, payload)

    return StructuredTool.from_function(
        func=run_tool,
        name=tool.name,
        description=tool.description or f"Proxy for Playwright tool {tool.name}.",
        args_schema=tool.args_schema,
    )


def get_playwright_tools() -> list[StructuredTool]:
    """Return thread-safe proxy tools for LangChain's Playwright toolkit."""
    global _tools

    if _tools is not None:
        return _tools

    _worker.start()
    source_tools = _worker.call(lambda: list(_worker._tools_by_name.values()))
    _tools = [_make_proxy_tool(tool) for tool in source_tools]
    return _tools


def close_playwright_browser() -> None:
    global _tools
    _worker.close()
    _tools = None
