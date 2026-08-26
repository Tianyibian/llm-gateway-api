from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.conversation_routes import router as conversation_router
from app.api.routes import router
from app.core.config import get_settings
from app.db.session import engine
from app.services.factory import LLMServiceFactory

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Database-backed, stateful chat and stateless reasoning endpoints "
        "with streaming OpenAI and Ollama adapters."
    ),
    lifespan=lifespan,
)
app.include_router(router)
app.include_router(conversation_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "configured_provider": settings.llm_provider,
        "provider": LLMServiceFactory(settings).resolve_provider(),
        "orchestrator": settings.llm_orchestrator,
    }
