from app.core.config import get_settings
from app.models import CVProfile, JobMatch, JobSearchRequest, SavedJob
from app.repository import repository
from app.schemas import JobSearchCreate, JobSearchResult
from app.tools.job_sources import search_demo_jobs
from app.tools.matching import score_job_against_profile


def run_job_search(payload: JobSearchCreate, profile: CVProfile) -> JobSearchResult:
    settings = get_settings()
    search = JobSearchRequest(**payload.model_dump())

    postings = search_demo_jobs(payload.query, payload.location)
    matches: list[JobMatch] = [
        score_job_against_profile(job=posting, profile=profile) for posting in postings
    ]
    matches.sort(key=lambda match: match.score, reverse=True)

    saved_jobs = []
    for match in matches:
        if match.score < settings.match_score_threshold:
            continue
        saved_jobs.append(
            repository.add_saved_job(
                SavedJob(
                    cv_profile_id=profile.id,
                    job=match.job,
                    match_score=match.score,
                    matched_skills=match.matched_skills,
                    missing_skills=match.missing_skills,
                    reason=match.reason,
                )
            )
        )

    return JobSearchResult(search_id=search.id, matches=matches, saved_jobs=saved_jobs)
