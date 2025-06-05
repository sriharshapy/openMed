"""
Configuration settings for the OpenMed OpenAI-Compatible Backend
"""

import os
from typing import List, Dict, Any

# Server Configuration
SERVER_HOST = os.getenv("OPENMED_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("OPENMED_PORT", "8000"))
DEBUG_MODE = os.getenv("OPENMED_DEBUG", "true").lower() == "true"

# CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost:8080",    # Open WebUI default
    "http://localhost:3000",    # Alternative frontend
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000",
    "*"  # Allow all for development - remove in production
]

# Model Configuration
DEFAULT_MODELS: List[Dict[str, Any]] = [
    {
        "id": "openmed-chat-v1",
        "object": "model",
        "owned_by": "openmed",
        "description": "OpenMed Chat Model for conversational AI"
    },
    {
        "id": "openmed-completion-v1",
        "object": "model",
        "owned_by": "openmed",
        "description": "OpenMed Completion Model for text generation"
    },
    {
        "id": "hello-world-model",
        "object": "model",
        "owned_by": "openmed",
        "description": "Simple hello world model for testing"
    },
    {
        "id": "openmed-medical-v1",
        "object": "model",
        "owned_by": "openmed",
        "description": "Specialized medical AI model"
    }
]

# Default Response Configuration
DEFAULT_MAX_TOKENS = 150
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0

# API Configuration
API_TITLE = "OpenMed OpenAI-Compatible Backend"
API_DESCRIPTION = """
A FastAPI backend that provides OpenAI-compatible endpoints for Open WebUI.

This backend implements:
- `/v1/models` - List available models
- `/v1/chat/completions` - Chat completions with streaming support
- `/v1/completions` - Text completions with streaming support

Features:
- OpenAI API compatibility
- CORS enabled for web applications
- Streaming responses
- Comprehensive request/response validation
- Health monitoring endpoints
"""
API_VERSION = "1.0.0"

# Logging Configuration
LOG_LEVEL = os.getenv("OPENMED_LOG_LEVEL", "info")

# Hello World Responses
HELLO_WORLD_RESPONSES = [
    "Hello World! Welcome to the OpenMed backend.",
    "Greetings! This is a simple response from OpenMed.",
    "Hello! OpenMed backend is running successfully.",
    "Hi there! This is your friendly OpenMed assistant.",
    "Welcome! OpenMed is ready to help you."
]

# Error Messages
ERROR_MESSAGES = {
    "model_not_found": "The specified model was not found",
    "invalid_request": "Invalid request format",
    "processing_error": "Error processing your request",
    "server_error": "Internal server error occurred"
} 