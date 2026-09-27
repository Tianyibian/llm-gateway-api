"""Bound multipart bodies and close temporary uploads before returning parsed data."""
from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile
import json

from app.models.schemas import AssistantRequest
from app.services.file_query import MAX_FILE_BYTES, parse_upload
from app.services.vision_service import OpenAIVisionService


async def parse_assistant_form(request: Request, settings):
    limit = max(settings.vision_max_image_bytes, MAX_FILE_BYTES) + 128 * 1024
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise HTTPException(413, "Upload request is too large.")
        body.extend(chunk)

    async def receive():
        return {"type": "http.request", "body": bytes(body), "more_body": False}

    bounded = Request(request.scope, receive=receive)
    async with bounded.form(max_files=2, max_fields=5) as form:
        allowed = {"query", "user_id", "conversation_id", "graphrag_search_mode", "policy_filters", "image", "file"}
        if any(k not in allowed or len(form.getlist(k)) != 1 for k in form):
            raise HTTPException(422, "Unknown or duplicate upload fields.")
        if "image" in form and "file" in form:
            raise HTTPException(422, "Attach one image or one document, not both.")
        if any(not isinstance(form.get(k), str) for k in ("query", "user_id") if k in form):
            raise HTTPException(422, "query and user_id must be text fields.")
        if "conversation_id" in form and not isinstance(form["conversation_id"], str):
            raise HTTPException(422, "conversation_id must be a text field.")
        if "graphrag_search_mode" in form and not isinstance(form["graphrag_search_mode"], str):
            raise HTTPException(422, "graphrag_search_mode must be a text field.")
        filters = None
        if "policy_filters" in form:
            if not isinstance(form["policy_filters"], str):
                raise HTTPException(422, "policy_filters must be a JSON text field.")
            try:
                filters = json.loads(form["policy_filters"])
            except (ValueError, RecursionError):
                raise HTTPException(422, "policy_filters must contain valid JSON.") from None
        parsed = AssistantRequest.model_validate({
            "query": form.get("query"), "user_id": form.get("user_id"),
            "conversation_id": form.get("conversation_id") or None,
            "graphrag_search_mode": form.get("graphrag_search_mode", "local"),
            "policy_filters": filters,
        })
        image_bytes = image_mime_type = document = None
        field = "file" if "file" in form else "image"
        upload = form.get(field)
        if upload is not None:
            if not isinstance(upload, UploadFile):
                raise HTTPException(422, f"{field} must be a file upload")
            max_bytes = MAX_FILE_BYTES if field == "file" else settings.vision_max_image_bytes
            data = await upload.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise HTTPException(413, f"{field} exceeds the {max_bytes // (1024 * 1024)} MB limit.")
            try:
                if field == "file":
                    document = await parse_upload(data, upload.filename or "")
                else:
                    image_mime_type = OpenAIVisionService.validate_image(data)
                    image_bytes = data
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None
        return parsed, image_bytes, image_mime_type, document
