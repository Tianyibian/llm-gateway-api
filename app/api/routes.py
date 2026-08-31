from __future__ import annotations

from collections.abc import AsyncIterator
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.api.conversation_routes import get_conversation_service
from app.models.schemas import (
    AssistantRequest,
    ChatRequest,
    ClassificationRequest,
    ConversationChatRequest,
    Message,
    QueryClassification,
)
from app.services.base import LLMService, ServiceType
from app.services.assistant_service import AssistantGraphService
from app.services.conversation_service import ConversationService
from app.services.errors import ConversationNotFoundError, LLMConfigurationError
from app.services.factory import LLMServiceFactory
from app.services.query_classifier import QueryClassifier
from app.services.knowledge_service import KnowledgeRetriever
from app.services.streaming import encode_sse
from app.services.vision_service import OpenAIVisionService
from app.core.config import Settings, get_settings

router = APIRouter(prefix="/api", tags=["llm"])
logger = logging.getLogger(__name__)


def get_llm_factory() -> LLMServiceFactory:
    return LLMServiceFactory()


def get_query_classifier(
    factory: LLMServiceFactory = Depends(get_llm_factory),
) -> QueryClassifier:
    try:
        return factory.create_classifier()
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def get_assistant_service(
    factory: LLMServiceFactory = Depends(get_llm_factory),
) -> AssistantGraphService:
    try:
        return factory.create_assistant()
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def get_knowledge_retriever(
    factory: LLMServiceFactory = Depends(get_llm_factory),
) -> KnowledgeRetriever:
    try:
        return factory.create_knowledge_retriever()
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


async def _stream(service: LLMService, request: ChatRequest) -> AsyncIterator[str]:
    yield encode_sse(
        "metadata",
        {
            "provider": service.provider,
            "model": service.model,
            "service": service.service_type.value,
        },
    )
    try:
        async for delta in service.stream(request.messages):
            yield encode_sse("delta", {"content": delta})
        yield encode_sse("done", "[DONE]")
    except Exception as exc:  # The HTTP headers have already been sent for a stream.
        logger.exception("LLM stream failed")
        yield encode_sse(
            "error",
            {
                "type": type(exc).__name__,
                "message": "LLM request failed. Check the server log and configuration.",
            },
        )


async def _stream_assistant(
    service: AssistantGraphService,
    request: AssistantRequest,
    *,
    history: list[tuple[str, str]],
    conversation_id: str,
    conversation_service: ConversationService,
    delete_if_empty_on_failure: bool,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> AsyncIterator[str]:
    yield encode_sse(
        "metadata",
        {
            "provider": service.provider,
            "model": service.model,
            "service": "assistant",
            "conversation_id": conversation_id,
        },
    )
    assistant_chunks: list[str] = []
    turn_persisted = False
    try:
        async for event_name, payload in service.stream(
            request.query,
            history=history,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        ):
            if event_name == "delta":
                assistant_chunks.append(payload.get("content", ""))
            yield encode_sse(event_name, payload)

        assistant_content = "".join(assistant_chunks)
        if not assistant_content.strip():
            raise RuntimeError("The assistant returned an empty response.")
        await conversation_service.save_turn(
            request.user_id,
            conversation_id,
            request.query,
            assistant_content,
        )
        turn_persisted = True
        yield encode_sse("done", "[DONE]")
    except Exception as exc:
        logger.exception("Assistant graph stream failed")
        yield encode_sse(
            "error",
            {
                "type": type(exc).__name__,
                "message": "Assistant request failed. Check the server log and configuration.",
            },
        )
    finally:
        if delete_if_empty_on_failure and not turn_persisted:
            try:
                await conversation_service.delete_conversation_if_empty(
                    request.user_id,
                    conversation_id,
                )
            except Exception:
                logger.exception("Failed to remove an empty assistant conversation")


async def _stream_conversation(
    service: LLMService,
    messages: list[Message],
    request: ConversationChatRequest,
    conversation_id: str,
    conversation_service: ConversationService,
    delete_if_empty_on_failure: bool,
) -> AsyncIterator[str]:
    turn_persisted = False
    yield encode_sse(
        "metadata",
        {
            "provider": service.provider,
            "model": service.model,
            "service": service.service_type.value,
            "conversation_id": conversation_id,
        },
    )
    assistant_chunks: list[str] = []
    try:
        async for delta in service.stream(messages):
            assistant_chunks.append(delta)
            yield encode_sse("delta", {"content": delta})

        assistant_content = "".join(assistant_chunks)
        if not assistant_content.strip():
            raise RuntimeError("The LLM returned an empty response.")

        await conversation_service.save_turn(
            request.user_id,
            conversation_id,
            request.current_user_message.content,
            assistant_content,
        )
        turn_persisted = True
        yield encode_sse("done", "[DONE]")
    except Exception as exc:
        logger.exception("Stateful LLM stream failed")
        yield encode_sse(
            "error",
            {
                "type": type(exc).__name__,
                "message": "LLM request or conversation persistence failed.",
            },
        )
    finally:
        if delete_if_empty_on_failure and not turn_persisted:
            try:
                await conversation_service.delete_conversation_if_empty(
                    request.user_id,
                    conversation_id,
                )
            except Exception:
                logger.exception("Failed to remove an empty auto-created conversation")


def _create_streaming_response(
    request: ChatRequest,
    factory: LLMServiceFactory,
    service_type: ServiceType,
) -> StreamingResponse:
    try:
        service = factory.create(service_type)
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return StreamingResponse(
        _stream(service, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.post(
    "/chat",
    response_class=StreamingResponse,
    summary="Stream a conversational response",
)
async def chat(
    request: ConversationChatRequest,
    factory: LLMServiceFactory = Depends(get_llm_factory),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> StreamingResponse:
    try:
        service = factory.create(ServiceType.CHAT)
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if request.conversation_id is None:
        title = " ".join(request.current_user_message.content.split())[:80]
        conversation = await conversation_service.create_conversation(
            request.user_id,
            title or None,
        )
        conversation_id = conversation.id
        history: list[Message] = []
        delete_if_empty_on_failure = True
    else:
        conversation_id = str(request.conversation_id)
        try:
            stored_messages = await conversation_service.get_conversation_messages(
                request.user_id,
                conversation_id,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        history = [
            Message(role=message.role, content=message.content)
            for message in stored_messages
        ]
        delete_if_empty_on_failure = False

    llm_messages = [*history, *request.messages]
    return StreamingResponse(
        _stream_conversation(
            service,
            llm_messages,
            request,
            conversation_id,
            conversation_service,
            delete_if_empty_on_failure,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/reason",
    response_class=StreamingResponse,
    summary="Stream a reasoning result",
)
async def reason(
    request: ChatRequest,
    factory: LLMServiceFactory = Depends(get_llm_factory),
) -> StreamingResponse:
    return _create_streaming_response(request, factory, ServiceType.REASON)


@router.post(
    "/classify",
    response_model=QueryClassification,
    summary="Classify a query for downstream search routing",
)
async def classify_query(
    request: ClassificationRequest,
    classifier: QueryClassifier = Depends(get_query_classifier),
) -> QueryClassification:
    try:
        return await classifier.classify(request.query)
    except Exception as exc:
        logger.exception("Query classification failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Query classification failed. Check the server log and model configuration.",
        ) from exc


@router.post(
    "/assistant",
    response_class=StreamingResponse,
    summary="Run the stateful multimodal LangGraph assistant",
)
async def assistant(
    http_request: Request,
    service: AssistantGraphService = Depends(get_assistant_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    content_type = http_request.headers.get("content-type", "").lower()

    try:
        if content_type.startswith("application/json"):
            request = AssistantRequest.model_validate(await http_request.json())
        elif content_type.startswith("multipart/form-data"):
            form = await http_request.form()
            request = AssistantRequest.model_validate(
                {
                    "query": form.get("query"),
                    "user_id": form.get("user_id"),
                    "conversation_id": form.get("conversation_id") or None,
                }
            )
            image = form.get("image")
            if image is not None:
                if not hasattr(image, "read"):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail="image must be a file upload",
                    )
                image_bytes = await image.read(settings.vision_max_image_bytes + 1)
                await image.close()
                if len(image_bytes) > settings.vision_max_image_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            "Image exceeds the "
                            f"{settings.vision_max_image_bytes // (1024 * 1024)} MB limit."
                        ),
                    )
                try:
                    image_mime_type = OpenAIVisionService.validate_image(image_bytes)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail=str(exc),
                    ) from exc
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Use application/json or multipart/form-data.",
            )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(include_context=False),
        ) from exc

    if request.conversation_id is None:
        title = " ".join(request.query.split())[:80]
        conversation = await conversation_service.create_conversation(
            request.user_id,
            title or None,
        )
        conversation_id = str(conversation.id)
        history: list[tuple[str, str]] = []
        delete_if_empty_on_failure = True
    else:
        conversation_id = str(request.conversation_id)
        try:
            stored_messages = await conversation_service.get_conversation_messages(
                request.user_id,
                conversation_id,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        history = [(message.role, message.content) for message in stored_messages]
        delete_if_empty_on_failure = False

    return StreamingResponse(
        _stream_assistant(
            service,
            request,
            history=history,
            conversation_id=conversation_id,
            conversation_service=conversation_service,
            delete_if_empty_on_failure=delete_if_empty_on_failure,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/knowledge/status",
    summary="Show the current Knowledge Base RAG index status",
)
async def knowledge_status(
    retriever: KnowledgeRetriever = Depends(get_knowledge_retriever),
) -> dict[str, object]:
    try:
        status_result = await retriever.status()
    except Exception as exc:
        logger.exception("Knowledge Base status failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge Base schema is unavailable. Run Alembic migrations.",
        ) from exc
    return {
        "documents": status_result.documents,
        "chunks": status_result.chunks,
        "public_documents": status_result.public_documents,
        "embedding_provider": status_result.embedding_provider,
        "embedding_model": status_result.embedding_model,
        "ready": status_result.chunks > 0,
    }
