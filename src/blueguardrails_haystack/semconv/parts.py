# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Convert individual Haystack/provider content parts to OTel GenAI parts."""

import base64
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from haystack.dataclasses import ByteStream, FileContent, ImageContent
from haystack.dataclasses.chat_message import (
    ChatMessageContentT,
    ReasoningContent,
    TextContent,
    ToolCall,
    ToolCallResult,
)

_DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/json",
    "application/xml",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_URL_KIND_MODALITIES = {
    "image-url": "image",
    "audio-url": "audio",
    "video-url": "video",
}


def _mime_to_modality(mime_type: str | None) -> str:
    """Map a MIME type to a GenAI content modality."""
    if not mime_type:
        return "file"
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("text/") or mime_type in _DOCUMENT_MIME_TYPES:
        return "document"
    return "file"


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and (parsed.netloc or parsed.scheme in {"file", "gs", "s3"}))


def _part(kind: str, modality: str, **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": kind, "modality": modality, **values}
    if values.get("mime_type") is None:
        result.pop("mime_type", None)
    return result


def blob_part(content: str, mime_type: str | None, modality: str | None = None) -> dict[str, Any]:
    """Create a GenAI blob part for inline data."""
    return _part("blob", modality or _mime_to_modality(mime_type), content=content, mime_type=mime_type)


def uri_part(uri: str, mime_type: str | None = None, modality: str | None = None) -> dict[str, Any]:
    """Create a GenAI URI part for external content."""
    return _part("uri", modality or _mime_to_modality(mime_type), uri=uri, mime_type=mime_type)


def file_part(file_id: str, mime_type: str | None = None, modality: str | None = None) -> dict[str, Any]:
    """Create a GenAI file part for provider-hosted content."""
    return _part("file", modality or _mime_to_modality(mime_type), file_id=file_id, mime_type=mime_type)


def image_content_to_semconv(part: ImageContent) -> dict[str, Any]:
    result = blob_part(part.base64_image, part.mime_type, modality="image")
    if part.detail:
        result["detail"] = part.detail
    return result


def file_content_to_semconv(part: FileContent) -> dict[str, Any]:
    result = blob_part(part.base64_data, part.mime_type)
    if part.filename:
        result["filename"] = part.filename
    return result


def chat_content_part_to_semconv(part: ChatMessageContentT) -> dict[str, Any] | None:
    """Convert a Haystack chat content part to a GenAI part."""
    if isinstance(part, TextContent):
        return {"type": "text", "content": part.text}
    if isinstance(part, ToolCall):
        result: dict[str, Any] = {"type": "tool_call", "name": part.tool_name}
        if part.id:
            result["id"] = part.id
        if part.arguments:
            result["arguments"] = part.arguments
        return result
    if isinstance(part, ToolCallResult):
        result = {"type": "tool_call_response", "response": part.result}
        if part.origin and part.origin.id:
            result["id"] = part.origin.id
        return result
    if isinstance(part, ReasoningContent):
        return {"type": "reasoning", "content": part.reasoning_text}
    if isinstance(part, ImageContent):
        return image_content_to_semconv(part)
    if isinstance(part, FileContent):
        return file_content_to_semconv(part)
    return None


def _media_type(part: Any) -> Any:
    return getattr(part, "media_type", None) or getattr(part, "mime_type", None)


def plain_part_to_semconv(part: Any) -> dict[str, Any] | None:
    """Convert a non-chat generator part to a GenAI part."""
    if isinstance(part, str):
        return {"type": "text", "content": part}
    if isinstance(part, ByteStream):
        return blob_part(base64.b64encode(part.data).decode("ascii"), part.mime_type)
    if isinstance(part, ImageContent):
        return image_content_to_semconv(part)
    if isinstance(part, FileContent):
        return file_content_to_semconv(part)

    url = getattr(part, "url", None) or getattr(part, "uri", None)
    if isinstance(url, str):
        return uri_part(url, _media_type(part), modality=_URL_KIND_MODALITIES.get(getattr(part, "kind", "")))

    file_id = getattr(part, "file_id", None)
    if isinstance(file_id, str):
        return file_part(file_id, _media_type(part))

    base64_content = getattr(part, "base64", None)
    if isinstance(base64_content, str):
        return blob_part(base64_content, _media_type(part))

    # Handle Google Part objects without importing optional dependencies.
    text = getattr(part, "text", None)
    if isinstance(text, str) and text:
        return {"type": "text", "content": text}

    inline_data = getattr(part, "inline_data", None)
    data = getattr(inline_data, "data", None)
    if data:
        content = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else str(data)
        return blob_part(content, getattr(inline_data, "mime_type", None))

    return None


def plain_parts_to_semconv(parts: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert an iterable of non-chat parts, dropping unsupported values."""
    return [converted for part in parts if (converted := plain_part_to_semconv(part))]


def generated_image_to_semconv(image: Any) -> dict[str, Any] | None:
    """Convert a generated image value to a GenAI part."""
    if isinstance(image, ImageContent):
        return image_content_to_semconv(image)
    if isinstance(image, ByteStream):
        return blob_part(base64.b64encode(image.data).decode("ascii"), image.mime_type, modality="image")
    if isinstance(image, str):
        if _is_url(image):
            return uri_part(image, modality="image")
        return blob_part(image, mime_type=None, modality="image")
    return plain_part_to_semconv(image)
