"""
OpenAI-Compatible FastAPI Backend for Open WebUI

This backend provides OpenAI-compatible API endpoints that can be used with Open WebUI.
It implements the required endpoints for models, chat completions, and completions.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import time
import json
import asyncio
from datetime import datetime

# Pydantic models for request/response validation
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author", example="user")
    content: str = Field(..., description="The content of the message", example="Hello, how are you?")

    class Config:
        schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?"
            }
        }

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use", example="openmed-chat-v1")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation so far")
    max_tokens: Optional[int] = Field(150, description="The maximum number of tokens to generate", example=150)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature to use", example=0.7, ge=0.0, le=2.0)
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress", example=False)
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter", example=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty", example=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty", example=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating")

    class Config:
        schema_extra = {
            "example": {
                "model": "openmed-chat-v1",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": False
            }
        }

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use", example="openmed-completion-v1")
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for", example="Once upon a time")
    max_tokens: Optional[int] = Field(150, description="The maximum number of tokens to generate", example=150)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature to use", example=0.7, ge=0.0, le=2.0)
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress", example=False)
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter", example=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty", example=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty", example=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating")

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

class ModelInfo(BaseModel):
    id: str = Field(..., description="The model identifier", example="openmed-chat-v1")
    object: str = Field("model", description="The object type", example="model")
    created: int = Field(..., description="The Unix timestamp when the model was created", example=1640995200)
    owned_by: str = Field("openmed", description="The organization that owns the model", example="openmed")

    class Config:
        schema_extra = {
            "example": {
                "id": "openmed-chat-v1",
                "object": "model",
                "created": 1640995200,
                "owned_by": "openmed"
            }
        }

class ModelsResponse(BaseModel):
    object: str = Field("list", description="The object type", example="list")
    data: List[ModelInfo] = Field(..., description="List of available models")

    class Config:
        schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "id": "openmed-chat-v1",
                        "object": "model",
                        "created": 1640995200,
                        "owned_by": "openmed"
                    }
                ]
            }
        }

class ChatChoice(BaseModel):
    index: int = Field(..., description="The index of the choice", example=0)
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: str = Field("stop", description="The reason the model stopped generating", example="stop")

class CompletionChoice(BaseModel):
    index: int = Field(..., description="The index of the choice", example=0)
    text: str = Field(..., description="The generated text", example=" Hello World!")
    finish_reason: str = Field("stop", description="The reason the model stopped generating", example="stop")

class Usage(BaseModel):
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt", example=10)
    completion_tokens: int = Field(..., description="Number of tokens in the completion", example=15)
    total_tokens: int = Field(..., description="Total number of tokens used", example=25)

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="A unique identifier for the chat completion", example="chatcmpl-123")
    object: str = Field("chat.completion", description="The object type", example="chat.completion")
    created: int = Field(..., description="The Unix timestamp when the completion was created", example=1640995200)
    model: str = Field(..., description="The model used for the completion", example="openmed-chat-v1")
    choices: List[ChatChoice] = Field(..., description="A list of chat completion choices")
    usage: Usage = Field(..., description="Usage statistics for the completion request")

    class Config:
        schema_extra = {
            "example": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1640995200,
                "model": "openmed-chat-v1",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you today?"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25
                }
            }
        }

class CompletionResponse(BaseModel):
    id: str = Field(..., description="A unique identifier for the completion", example="cmpl-123")
    object: str = Field("text_completion", description="The object type", example="text_completion")
    created: int = Field(..., description="The Unix timestamp when the completion was created", example=1640995200)
    model: str = Field(..., description="The model used for the completion", example="openmed-completion-v1")
    choices: List[CompletionChoice] = Field(..., description="A list of completion choices")
    usage: Usage = Field(..., description="Usage statistics for the completion request")

    class Config:
        schema_extra = {
            "example": {
                "id": "cmpl-123",
                "object": "text_completion",
                "created": 1640995200,
                "model": "openmed-completion-v1",
                "choices": [
                    {
                        "index": 0,
                        "text": " there was a brave knight.",
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 6,
                    "total_tokens": 10
                }
            }
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="The health status", example="healthy")
    timestamp: str = Field(..., description="The current timestamp", example="2023-12-01T12:00:00.000Z")

class RootResponse(BaseModel):
    message: str = Field(..., description="Welcome message", example="OpenMed OpenAI-Compatible Backend")
    status: str = Field(..., description="Server status", example="running")
    endpoints: List[str] = Field(..., description="Available API endpoints")

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="OpenMed OpenAI-Compatible Backend",
    description="""
    🏥 **OpenMed OpenAI-Compatible Backend**
    
    A FastAPI backend that provides OpenAI-compatible API endpoints for use with Open WebUI and other applications.
    
    ## Features
    
    - ✅ **OpenAI API Compatibility**: Full compatibility with OpenAI's API format
    - ✅ **Streaming Support**: Real-time streaming responses for better user experience
    - ✅ **CORS Enabled**: Cross-origin requests supported for web applications
    - ✅ **Request Validation**: Comprehensive input validation using Pydantic models
    - ✅ **Health Monitoring**: Health check endpoints for monitoring server status
    - ✅ **Auto-documentation**: Automatic API documentation with Swagger UI
    - ✅ **Hello World Responses**: Simple test responses for development and testing
    
    ## Quick Start
    
    1. **Health Check**: `GET /health` - Check if the server is running
    2. **List Models**: `GET /v1/models` - Get available models
    3. **Chat**: `POST /v1/chat/completions` - Start a chat conversation
    4. **Completion**: `POST /v1/completions` - Generate text completions
    
    ## Integration with Open WebUI
    
    Set the base URL in Open WebUI to: `http://localhost:8000/v1`
    """,
    version="1.0.0",
    contact={
        "name": "OpenMed Team",
        "url": "https://github.com/openmed",
        "email": "support@openmed.ai"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",  # Swagger UI endpoint
    redoc_url="/redoc",  # ReDoc endpoint
    openapi_url="/openapi.json"  # OpenAPI schema endpoint
)

# Enable CORS for Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",  # Open WebUI default URL
        "http://localhost:3000",  # Alternative frontend URL
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available models
AVAILABLE_MODELS = [
    {
        "id": "openmed-chat-v1",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "openmed"
    },
    {
        "id": "openmed-completion-v1",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "openmed"
    },
    {
        "id": "hello-world-model",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "openmed"
    }
]

def generate_response_id() -> str:
    """Generate a unique response ID"""
    return f"chatcmpl-{int(time.time())}"

def calculate_usage(prompt: str, response: str) -> Usage:
    """Calculate token usage (simplified implementation)"""
    prompt_tokens = len(prompt.split())
    completion_tokens = len(response.split())
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )

@app.get(
    "/",
    response_model=RootResponse,
    tags=["Server"],
    summary="Root endpoint",
    description="Get basic information about the API server and available endpoints."
)
async def root():
    """
    ## Root Endpoint
    
    Returns basic information about the OpenMed backend server including:
    - Server status
    - Available API endpoints
    - Welcome message
    """
    return RootResponse(
        message="OpenMed OpenAI-Compatible Backend",
        status="running",
        endpoints=["/v1/models", "/v1/chat/completions", "/v1/completions"]
    )

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Server"],
    summary="Health check",
    description="Check the health status of the API server.",
    responses={
        200: {
            "description": "Server is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2023-12-01T12:00:00.000Z"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    ## Health Check Endpoint
    
    Use this endpoint to verify that the server is running and responding correctly.
    Returns the current server status and timestamp.
    """
    return HealthResponse(
        status="healthy", 
        timestamp=datetime.now().isoformat()
    )

@app.get(
    "/v1/models",
    response_model=ModelsResponse,
    tags=["Models"],
    summary="List available models",
    description="Lists the currently available models, and provides basic information about each one such as the owner and availability.",
    responses={
        200: {
            "description": "Successfully retrieved list of models",
            "content": {
                "application/json": {
                    "example": {
                        "object": "list",
                        "data": [
                            {
                                "id": "openmed-chat-v1",
                                "object": "model",
                                "created": 1640995200,
                                "owned_by": "openmed"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def list_models():
    """
    ## List Models Endpoint
    
    Returns a list of all available models that can be used with the chat completions 
    and completions endpoints. Each model includes:
    
    - **id**: The model identifier
    - **object**: Always "model"
    - **created**: Unix timestamp when the model was created
    - **owned_by**: The organization that owns the model
    """
    return ModelsResponse(data=[ModelInfo(**model) for model in AVAILABLE_MODELS])

@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["Chat"],
    summary="Create chat completion",
    description="Creates a model response for the given chat conversation. Supports both streaming and non-streaming responses.",
    responses={
        200: {
            "description": "Successfully generated chat completion",
            "content": {
                "application/json": {
                    "example": {
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1640995200,
                        "model": "openmed-chat-v1",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Hello! How can I help you today?"
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 15,
                            "total_tokens": 25
                        }
                    }
                }
            }
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "model"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error"
        }
    }
)
async def chat_completions(request: ChatCompletionRequest):
    """
    ## Chat Completions Endpoint
    
    Creates a model response for the given chat conversation. This endpoint is compatible 
    with OpenAI's chat completions API.
    
    ### Features:
    - **Multi-turn conversations**: Supports system, user, and assistant messages
    - **Streaming**: Set `stream: true` for real-time response streaming  
    - **Temperature control**: Adjust randomness of responses (0.0 - 2.0)
    - **Token limits**: Control maximum response length with `max_tokens`
    
    ### Example Usage:
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "openmed-chat-v1",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }
    )
    ```
    """
    try:
        # Generate hello world response based on the last user message
        last_user_message = None
        for message in reversed(request.messages):
            if message.role == "user":
                last_user_message = message.content
                break
        
        # Create a hello world response
        response_content = f"Hello World! You said: '{last_user_message or 'nothing'}'. This is a simple response from the OpenMed backend."
        
        response_data = {
            "id": generate_response_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": calculate_usage(
                " ".join([msg.content for msg in request.messages]),
                response_content
            ).dict()
        }
        
        if request.stream:
            return StreamingResponse(
                generate_chat_stream(response_data),
                media_type="text/plain"
            )
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error processing chat completion: {str(e)}"
        )

@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    tags=["Completions"],
    summary="Create completion",
    description="Creates a completion for the provided prompt and parameters. Supports both streaming and non-streaming responses.",
    responses={
        200: {
            "description": "Successfully generated completion",
            "content": {
                "application/json": {
                    "example": {
                        "id": "cmpl-123",
                        "object": "text_completion",
                        "created": 1640995200,
                        "model": "openmed-completion-v1",
                        "choices": [
                            {
                                "index": 0,
                                "text": " there was a brave knight.",
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 6,
                            "total_tokens": 10
                        }
                    }
                }
            }
        },
        422: {
            "description": "Validation Error"
        },
        500: {
            "description": "Internal Server Error"
        }
    }
)
async def completions(request: CompletionRequest):
    """
    ## Text Completions Endpoint
    
    Creates a completion for the provided prompt and parameters. This endpoint is 
    compatible with OpenAI's completions API.
    
    ### Features:
    - **Single or multiple prompts**: Accepts string or array of strings
    - **Streaming**: Set `stream: true` for real-time response streaming
    - **Temperature control**: Adjust randomness of responses (0.0 - 2.0)
    - **Stop sequences**: Specify sequences to stop generation
    
    ### Example Usage:
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "openmed-completion-v1",
            "prompt": "Once upon a time",
            "max_tokens": 50
        }
    )
    ```
    """
    try:
        # Handle both string and list prompts
        if isinstance(request.prompt, list):
            prompt_text = " ".join(request.prompt)
        else:
            prompt_text = request.prompt
            
        # Generate hello world completion
        response_text = f" Hello World! Your prompt was: '{prompt_text}'. This is a simple completion from the OpenMed backend."
        
        response_data = {
            "id": generate_response_id(),
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": response_text,
                    "finish_reason": "stop"
                }
            ],
            "usage": calculate_usage(prompt_text, response_text).dict()
        }
        
        if request.stream:
            return StreamingResponse(
                generate_completion_stream(response_data),
                media_type="text/plain"
            )
            
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error processing completion: {str(e)}"
        )

async def generate_chat_stream(response_data: Dict[str, Any]):
    """Generate streaming response for chat completions"""
    content = response_data["choices"][0]["message"]["content"]
    words = content.split()
    
    # Send data chunk by chunk
    for i, word in enumerate(words):
        chunk = {
            "id": response_data["id"],
            "object": "chat.completion.chunk",
            "created": response_data["created"],
            "model": response_data["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)  # Simulate processing time
    
    # Send final chunk
    final_chunk = {
        "id": response_data["id"],
        "object": "chat.completion.chunk",
        "created": response_data["created"],
        "model": response_data["model"],
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

async def generate_completion_stream(response_data: Dict[str, Any]):
    """Generate streaming response for completions"""
    text = response_data["choices"][0]["text"]
    words = text.split()
    
    # Send data chunk by chunk
    for i, word in enumerate(words):
        chunk = {
            "id": response_data["id"],
            "object": "text_completion",
            "created": response_data["created"],
            "model": response_data["model"],
            "choices": [
                {
                    "index": 0,
                    "text": word + " ",
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)  # Simulate processing time
    
    # Send final chunk
    final_chunk = {
        "id": response_data["id"],
        "object": "text_completion",
        "created": response_data["created"],
        "model": response_data["model"],
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 