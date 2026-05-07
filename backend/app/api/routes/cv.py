from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import CVExtractResponse
from app.services.cv_extraction import extract_cv_text

router = APIRouter()


@router.post("/extract", response_model=CVExtractResponse)
async def extract_cv(file: UploadFile = File(...)) -> CVExtractResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded CV file is empty")

    try:
        text = extract_cv_text(file.filename or "cv", file.content_type, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    text = text[:20000]
    return CVExtractResponse(
        filename=file.filename or "cv",
        content_type=file.content_type,
        text=text,
        character_count=len(text),
    )
