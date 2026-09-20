from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.conversation_routes import router as conversation_router
from app.api.routes import router
from app.api.graphrag_routes import router as graphrag_router
from app.core.config import get_settings
from app.db.session import engine
from app.services.factory import LLMServiceFactory
from app.services.snowflake_analytics import dispose_snowflake_engines

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await engine.dispose()
    dispose_snowflake_engines()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "A streaming LLM gateway with stateful conversations, LangGraph routing, "
        "pgvector Knowledge Base retrieval, and grounded answers through OpenAI "
        "or Ollama."
    ),
    lifespan=lifespan,
)
app.include_router(router)
app.include_router(conversation_router)
app.include_router(graphrag_router)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def assistant_ui() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "configured_provider": settings.llm_provider,
        "provider": LLMServiceFactory(settings).resolve_provider(),
        "orchestrator": settings.llm_orchestrator,
        "analytics_backend": LLMServiceFactory(settings).resolve_analytics_backend(),
    }
