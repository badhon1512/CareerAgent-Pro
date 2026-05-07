from fastapi import APIRouter

from app.api.routes import agent, cv, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(cv.router, prefix="/cv", tags=["cv"])
