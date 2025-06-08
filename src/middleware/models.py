#!/usr/bin/env python3
"""
Pydantic Models for OpenMed OpenWebUI-Compatible Backend

This module contains all the Pydantic models used for request/response validation,
serialization, and API documentation. The models are designed to be compatible
with OpenAI's API specification while adding support for file uploads and
additional features.

Models included:
- Request models (ChatCompletionRequest, CompletionRequest, etc.)
- Response models (ChatCompletionResponse, CompletionResponse, etc.)
- File handling models (FileUploadResponse, ProcessedFileInfo, etc.)
- Utility models (HealthResponse, ErrorResponse, etc.)
"""

from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ContentItem(BaseModel):
    """Individual content item within a message"""
    type: str = Field(..., description="Content type (text, image_url)")
    text: Optional[str] = Field(None, description="Text content")
    image_url: Optional[Dict[str, str]] = Field(None, description="Image URL object with 'url' key")


class Message(BaseModel):
    """A single message in a conversation"""
    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: Union[str, List[ContentItem]] = Field(..., description="The content of the message (string or array of content items)")
    name: Optional[str] = Field(None, description="The name of the author of this message")
    
    @field_validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant', 'function']:
            raise ValueError('Role must be one of: system, user, assistant, function')
        return v


class FileReference(BaseModel):
    """Reference to an uploaded file"""
    type: str = Field(..., description="Type of file reference (file, collection)")
    id: str = Field(..., description="Unique identifier for the file or collection")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions (OpenAI compatible)"""
    model: str = Field(..., description="The model to use for completion")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate", ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    n: Optional[int] = Field(1, description="Number of completions to generate", ge=1, le=5)
    stream: Optional[bool] = Field(False, description="Whether to stream partial results")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty", ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias dictionary")
    user: Optional[str] = Field(None, description="User identifier")
    
    # File support (OpenWebUI extension)
    files: Optional[List[FileReference]] = Field(None, description="List of file references")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "openmed-chat-v1",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how can you help me?"}
                ],
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": False
            }
        }


class CompletionRequest(BaseModel):
    """Request model for text completions (OpenAI compatible)"""
    model: str = Field(..., description="The model to use for completion")
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for")
    max_tokens: Optional[int] = Field(16, description="Maximum number of tokens to generate", ge=1, le=4096)
    temperature: Optional[float] = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    n: Optional[int] = Field(1, description="Number of completions to generate", ge=1, le=5)
    stream: Optional[bool] = Field(False, description="Whether to stream partial results")
    logprobs: Optional[int] = Field(None, description="Include log probabilities", ge=0, le=5)
    echo: Optional[bool] = Field(False, description="Echo back the prompt")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty", ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(1, description="Number of completions to generate server-side", ge=1, le=20)
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias dictionary")
    user: Optional[str] = Field(None, description="User identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "openmed-completion-v1",
                "prompt": "Once upon a time",
                "max_tokens": 100,
                "temperature": 0.8,
                "stream": False
            }
        }


class Choice(BaseModel):
    """A single choice in a completion response"""
    index: int = Field(..., description="Index of the choice")
    message: Optional[Dict[str, Any]] = Field(None, description="Message object (for chat completions)")
    text: Optional[str] = Field(None, description="Generated text (for text completions)")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities")


class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions (OpenAI compatible)"""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "openmed-chat-v1",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21
                }
            }
        }


class CompletionResponse(BaseModel):
    """Response model for text completions (OpenAI compatible)"""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "cmpl-abc123",
                "object": "text_completion",
                "created": 1677652288,
                "model": "openmed-completion-v1",
                "choices": [{
                    "text": "This is a sample completion.",
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12
                }
            }
        }


class Model(BaseModel):
    """Model information"""
    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type")
    created: Optional[int] = Field(None, description="Unix timestamp of creation")
    owned_by: str = Field(..., description="Organization that owns the model")
    permission: Optional[List[Dict[str, Any]]] = Field(None, description="Model permissions")
    root: Optional[str] = Field(None, description="Root model")
    parent: Optional[str] = Field(None, description="Parent model")
    
    # Custom fields for OpenMed
    description: Optional[str] = Field(None, description="Model description")
    capabilities: Optional[List[str]] = Field(None, description="Model capabilities")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "openmed-chat-v1",
                "object": "model",
                "owned_by": "openmed",
                "description": "OpenMed Chat Model for conversational AI",
                "capabilities": ["chat", "completion", "file-processing"]
            }
        }


class ModelListResponse(BaseModel):
    """Response model for listing models (OpenAI compatible)"""
    object: str = Field("list", description="Object type")
    data: List[Model] = Field(..., description="List of available models")
    
    class Config:
        schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "id": "openmed-chat-v1",
                        "object": "model",
                        "owned_by": "openmed",
                        "description": "OpenMed Chat Model for conversational AI"
                    }
                ]
            }
        }


class ProcessedFileInfo(BaseModel):
    """Information about a processed file"""
    id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    safe_filename: str = Field(..., description="Safe filename used for storage")
    file_path: str = Field(..., description="Path to the stored file")
    size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type of the file")
    upload_time: datetime = Field(..., description="Time when file was uploaded")
    processed: bool = Field(True, description="Whether the file has been processed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional file metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "file_abc123",
                "filename": "document.pdf",
                "safe_filename": "a1b2c3d4_document.pdf",
                "file_path": "/uploads/a1b2c3d4_document.pdf",
                "size": 102400,
                "mime_type": "application/pdf",
                "upload_time": "2024-01-15T10:30:00",
                "processed": True
            }
        }


class FileUploadResponse(BaseModel):
    """Response model for file uploads"""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type of the file")
    upload_time: datetime = Field(..., description="Time when file was uploaded")
    message: str = Field(..., description="Success message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="File metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "file_id": "file_abc123",
                "filename": "document.pdf",
                "size": 102400,
                "mime_type": "application/pdf",
                "upload_time": "2024-01-15T10:30:00",
                "message": "File uploaded successfully",
                "metadata": {"category": "document", "language": "en"}
            }
        }


class HealthResponse(BaseModel):
    """Response model for health checks"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "version": "1.0.0",
                "uptime_seconds": 3600
            }
        }


class ErrorDetail(BaseModel):
    """Error detail information"""
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    code: Optional[Union[str, int]] = Field(None, description="Error code")
    param: Optional[str] = Field(None, description="Parameter that caused the error")


class ErrorResponse(BaseModel):
    """Response model for errors (OpenAI compatible)"""
    error: ErrorDetail = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "message": "Invalid request format",
                    "type": "invalid_request_error",
                    "code": 400
                }
            }
        }


class StreamingChoice(BaseModel):
    """A single choice in a streaming response"""
    index: int = Field(..., description="Index of the choice")
    delta: Dict[str, Any] = Field(..., description="Delta object containing partial content")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing")


class StreamingResponse(BaseModel):
    """Response model for streaming completions"""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(..., description="Object type (chat.completion.chunk or text_completion)")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[StreamingChoice] = Field(..., description="List of streaming choices")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "openmed-chat-v1",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None
                }]
            }
        }


class EmbeddingRequest(BaseModel):
    """Request model for embeddings (OpenAI compatible)"""
    model: str = Field(..., description="The model to use for embeddings")
    input: Union[str, List[str]] = Field(..., description="Input text(s) to embed")
    user: Optional[str] = Field(None, description="User identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "openmed-embedding-v1",
                "input": "The quick brown fox jumps over the lazy dog."
            }
        }


class Embedding(BaseModel):
    """A single embedding result"""
    object: str = Field("embedding", description="Object type")
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., description="Index of the embedding")


class EmbeddingResponse(BaseModel):
    """Response model for embeddings (OpenAI compatible)"""
    object: str = Field("list", description="Object type")
    data: List[Embedding] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embeddings")
    usage: Usage = Field(..., description="Token usage information")
    
    class Config:
        schema_extra = {
            "example": {
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }],
                "model": "openmed-embedding-v1",
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 0,
                    "total_tokens": 8
                }
            }
        }


# Utility functions for model validation and creation
def create_chat_completion_response(
    completion_id: str,
    model: str,
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> ChatCompletionResponse:
    """Helper function to create a chat completion response"""
    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=int(datetime.now(UTC).timestamp()),
        model=model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    )


def create_completion_response(
    completion_id: str,
    model: str,
    text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> CompletionResponse:
    """Helper function to create a text completion response"""
    return CompletionResponse(
        id=completion_id,
        object="text_completion",
        created=int(datetime.now(UTC).timestamp()),
        model=model,
        choices=[{
            "text": text,
            "index": 0,
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    )


def validate_model_id(model_id: str, available_models: List[str]) -> bool:
    """Validate that a model ID is available"""
    return model_id in available_models


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage"""
    import re
    # Remove or replace unsafe characters
    safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Limit length
    if len(safe_filename) > 100:
        name, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, '')
        safe_filename = name[:90] + ('.' + ext if ext else '')
    return safe_filename
