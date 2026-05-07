from concurrent.futures import Future
from queue import Queue
from threading import Thread
from typing import Any, Callable
from urllib.parse import quote_plus, urlencode, urljoin, urlparse, parse_qsl, urlunparse

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


def _portal_search_url(portal: str, query: str, location: str) -> str:
    normalized = portal.strip().lower()
    q = quote_plus(query.strip())
    loc = quote_plus(location.strip())

    if normalized == "stepstone":
        return f"https://www.stepstone.de/jobs/{q}/in-{loc}"
    if normalized == "indeed":
        return f"https://de.indeed.com/jobs?q={q}&l={loc}"
    if normalized == "xing":
        return f"https://www.xing.com/jobs/search?keywords={q}&location={loc}"

    raise ValueError("portal must be one of: stepstone, indeed, xing")


def _portal_page_url(portal: str, search_url: str, page_number: int) -> str:
    if page_number <= 1:
        return search_url

    normalized = portal.strip().lower()
    parsed = urlparse(search_url)
    query_params = dict(parse_qsl(parsed.query))

    if normalized == "indeed":
        query_params["start"] = str((page_number - 1) * 10)
        return urlunparse(parsed._replace(query=urlencode(query_params)))

    if normalized == "xing":
        query_params["page"] = str(page_number)
        return urlunparse(parsed._replace(query=urlencode(query_params)))

    if normalized == "stepstone":
        query_params["page"] = str(page_number)
        return urlunparse(parsed._replace(query=urlencode(query_params)))

    return search_url


def _looks_like_job_url(url: str) -> bool:
    lowered = url.lower()
    blocked_parts = [
        "/jobs/search",
        "/jobs?q=",
        "/jobs/",
        "/stellenangebote?",
        "/karriere",
        "/career",
        "/companies",
        "/company/",
        "/login",
        "/signin",
        "/register",
    ]

    if lowered.rstrip("/") in {
        "https://www.stepstone.de",
        "https://www.indeed.com",
        "https://de.indeed.com",
        "https://www.xing.com",
    }:
        return False

    if any(part in lowered for part in blocked_parts):
        if not any(marker in lowered for marker in ["/viewjob", "/rc/clk", "/job/", ".html", "/jobs/stellenangebote-"]):
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


def _clean_text(text: str, max_chars: int = 4000) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:max_chars]


def _read_job_page(page: Any, url: str) -> dict[str, str | int | None]:
    response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(1500)

    title = ""
    try:
        title = page.title()
    except Exception:
        title = ""

    visible_text = ""
    try:
        visible_text = page.locator("body").inner_text(timeout=8000)
    except Exception:
        visible_text = ""

    return {
        "url": url,
        "status_code": response.status if response else None,
        "page_title": title[:180],
        "description": _clean_text(visible_text),
    }


def _search_portal_jobs_sync(
    portal: str,
    query: str,
    location: str = "Germany",
    max_results: int = 20,
    max_pages: int = 3,
    max_detail_pages: int = 8,
) -> dict[str, Any]:
    max_results = min(max(max_results, 1), 50)
    max_pages = min(max(max_pages, 1), 8)
    max_detail_pages = min(max(max_detail_pages, 0), max_results)
    search_url = _portal_search_url(portal, query, location)
    browser = _worker._browser
    if browser is None:
        raise RuntimeError("Playwright browser is not running")

    page = browser.new_page()
    try:
        jobs: list[dict[str, str]] = []
        seen: set[str] = set()
        searched_pages: list[dict[str, Any]] = []

        for page_number in range(1, max_pages + 1):
            page_url = _portal_page_url(portal, search_url, page_number)
            response = page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
            status_code = response.status if response else None
            page.wait_for_timeout(2500)

            anchors = page.locator("a[href]").evaluate_all(
                """elements => elements.map((element) => ({
                    href: element.href,
                    text: (element.innerText || element.textContent || "").trim()
                }))"""
            )

            page_job_count = 0
            for anchor in anchors:
                href = str(anchor.get("href") or "").strip()
                if not href:
                    continue
                absolute_url = urljoin(page_url, href)
                if absolute_url in seen or not _looks_like_job_url(absolute_url):
                    continue
                seen.add(absolute_url)
                jobs.append(
                    {
                        "source": portal,
                        "title": str(anchor.get("text") or "").strip()[:180],
                        "url": absolute_url,
                        "result_page_url": page_url,
                    }
                )
                page_job_count += 1
                if len(jobs) >= max_results:
                    break

            searched_pages.append(
                {
                    "page": page_number,
                    "url": page_url,
                    "status_code": status_code,
                    "job_urls_found": page_job_count,
                }
            )
            if len(jobs) >= max_results:
                break

        detail_page = browser.new_page()
        try:
            for job in jobs[:max_detail_pages]:
                detail = _read_job_page(detail_page, job["url"])
                if detail.get("page_title") and not job.get("title"):
                    job["title"] = str(detail["page_title"])
                job["job_page_status_code"] = detail["status_code"]
                job["description"] = str(detail.get("description") or "")
        finally:
            detail_page.close()

        return {
            "portal": portal,
            "query": query,
            "location": location,
            "search_url": search_url,
            "searched_pages": searched_pages,
            "jobs": jobs,
            "note": (
                "Only direct job-posting-like URLs are returned; home/search pages are filtered out. "
                "description is extracted by opening job pages for the first max_detail_pages jobs."
            ),
        }
    finally:
        page.close()


def search_portal_jobs(
    portal: str,
    query: str,
    location: str = "Germany",
    max_results: int = 20,
    max_pages: int = 3,
    max_detail_pages: int = 8,
) -> dict[str, Any]:
    """
    Search StepStone, Indeed, or XING with Playwright across result pages.

    Returns direct job posting links, the result page URL where each job was found,
    and description text from the first max_detail_pages job pages.

    Use this instead of navigating to portal homepages. Change portal/query/location when a
    search returns too few direct job links.
    """
    _worker.start()
    return _worker.call(
        lambda: _search_portal_jobs_sync(
            portal,
            query,
            location,
            max_results,
            max_pages,
            max_detail_pages,
        )
    )


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
    """Return the full LangChain Playwright toolkit plus focused job-search helpers."""
    global _tools

    if _tools is not None:
        return _tools

    _worker.start()
    source_tools = _worker.call(lambda: list(_worker._tools_by_name.values()))
    playwright_tools = [_make_proxy_tool(tool) for tool in source_tools]
    _tools = [
        StructuredTool.from_function(search_portal_jobs),
        *playwright_tools,
    ]
    return _tools


def close_playwright_browser() -> None:
    global _tools
    _worker.close()
    _tools = None
