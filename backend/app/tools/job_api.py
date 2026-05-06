import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def _key_status(name: str) -> dict:
    value = _env(name)
    return {
        "name": name,
        "configured": bool(value),
        "length": len(value) if value else 0,
    }


# ADZUNA JOB SEARCH TOOL

@tool
def search_adzuna_jobs(
    query: str,
    location: str = "Germany",
    page: int = 1,
    results_per_page: int = 20,
) -> list[dict]:
    """
    Search jobs using the Adzuna Jobs API.

    Use this tool for structured and reliable job discovery.
    Best for AI jobs, software jobs, engineering, remote jobs, etc.
    """

    adzuna_app_id = _env("ADZUNA_APP_ID")
    adzuna_app_key = _env("ADZUNA_APP_KEY")

    if not adzuna_app_id or not adzuna_app_key:
        return [{
            "error": "Missing ADZUNA_APP_ID or ADZUNA_APP_KEY in .env"
        }]

    results_per_page = min(max(results_per_page, 1), 50)
    page = max(page, 1)

    url = f"https://api.adzuna.com/v1/api/jobs/de/search/{page}"

    params = {
        "app_id": adzuna_app_id,
        "app_key": adzuna_app_key,
        "what": query,
        "where": location,
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }

    try:
        response = requests.get(url, params=params, timeout=20)

        if response.status_code != 200:
            return [{
                "error": "Adzuna API request failed",
                "status_code": response.status_code,
                "details": response.text[:1000],
            }]

        data = response.json()

        jobs = []

        for job in data.get("results", []):
            jobs.append({
                "source": "Adzuna",
                "title": job.get("title"),
                "company": job.get("company", {}).get("display_name"),
                "location": job.get("location", {}).get("display_name"),
                "description": job.get("description"),
                "url": job.get("redirect_url"),
                "created": job.get("created"),
                "category": job.get("category", {}).get("label"),
                "contract_type": job.get("contract_type"),
                "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"),
            })

        return jobs

    except requests.RequestException as e:
        return [{
            "error": "Network error while calling Adzuna API",
            "details": str(e),
        }]


# GOOGLE JOBS TOOL (SERPAPI)

@tool
def search_google_jobs(
    query: str,
    location: str = "Germany",
    results_per_page: int = 10,
) -> list[dict]:
    """
    Search jobs using Google Jobs via SerpApi.

    Use this tool when broader job coverage is needed.
    Includes jobs from LinkedIn, Indeed, company sites, etc.
    """

    serpapi_key = _env("SERPAPI_KEY")

    if not serpapi_key:
        return [{
            "error": "Missing SERPAPI_KEY in .env"
        }]

    results_per_page = min(max(results_per_page, 1), 20)

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "api_key": serpapi_key,
    }

    try:
        response = requests.get(url, params=params, timeout=20)

        if response.status_code != 200:
            error = "SerpApi request failed"
            if response.status_code == 401:
                error = "SerpApi API key is invalid or unauthorized"
            return [{
                "error": error,
                "status_code": response.status_code,
                "details": response.text[:1000],
            }]

        data = response.json()

        jobs = []

        for job in data.get("jobs_results", [])[:results_per_page]:
            jobs.append({
                "source": "Google Jobs",
                "title": job.get("title"),
                "company": job.get("company_name"),
                "location": job.get("location"),
                "description": job.get("description"),
                "job_id": job.get("job_id"),
                "via": job.get("via"),
                "thumbnail": job.get("thumbnail"),
                "detected_extensions": job.get("detected_extensions"),
                "apply_options": job.get("apply_options"),
            })

        return jobs

    except requests.RequestException as e:
        return [{
            "error": "Network error while calling SerpApi",
            "details": str(e),
        }]


@tool
def check_job_api_config() -> list[dict]:
    """
    Check whether job-search API environment variables are loaded.

    This does not reveal secret values.
    """
    return [
        _key_status("ADZUNA_APP_ID"),
        _key_status("ADZUNA_APP_KEY"),
        _key_status("SERPAPI_KEY"),
    ]
