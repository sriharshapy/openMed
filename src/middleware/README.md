# OpenMed OpenAI-Compatible Backend

A FastAPI backend that provides OpenAI-compatible API endpoints for use with Open WebUI and other applications that support the OpenAI API format.

## Features

- ✅ **OpenAI API Compatibility**: Implements `/v1/models`, `/v1/chat/completions`, and `/v1/completions` endpoints
- ✅ **Streaming Support**: Real-time streaming responses for better UX
- ✅ **CORS Enabled**: Cross-origin requests supported for web applications
- ✅ **Request Validation**: Comprehensive input validation using Pydantic models
- ✅ **Health Monitoring**: Health check endpoints for monitoring
- ✅ **Enhanced Swagger UI**: Interactive API documentation with examples and detailed descriptions
- ✅ **Response Models**: Fully typed response models with validation
- ✅ **Hello World Responses**: Simple test responses for development

## Quick Start

### 1. Install Dependencies

Make sure you have the required dependencies installed:

```bash
# From the project root directory
pip install -r requirements.txt
```

### 2. Run the Backend

```bash
# Navigate to the middleware directory
cd src/middleware

# Run using the startup script
python run_backend.py

# Or run directly
python openai_backend.py

# Or using uvicorn
uvicorn openai_backend:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Explore the API Documentation

Once running, you can explore the interactive API documentation:

- **🔥 Swagger UI**: http://localhost:8000/docs
- **📖 ReDoc**: http://localhost:8000/redoc  
- **📋 OpenAPI Schema**: http://localhost:8000/openapi.json

The Swagger UI includes:
- **Interactive Testing**: Try out all endpoints directly in the browser
- **Request Examples**: Pre-filled examples for easy testing
- **Response Models**: Complete response schemas with examples
- **Parameter Validation**: Real-time validation of request parameters
- **Authentication**: Support for API key authentication (if needed)

### 4. Test the Backend

Once running, you can test the endpoints:

```bash
# Check if the server is running
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openmed-chat-v1",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Test text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openmed-completion-v1",
    "prompt": "Hello world"
  }'
```

## 🔥 Swagger UI Features

### Interactive Documentation

The enhanced Swagger UI provides:

1. **📋 Organized by Tags**:
   - **Server**: Root and health endpoints
   - **Models**: Model listing functionality  
   - **Chat**: Chat completion endpoints
   - **Completions**: Text completion endpoints

2. **🎯 Try It Out Feature**:
   - Click "Try it out" on any endpoint
   - Modify the example request
   - Execute directly from the browser
   - See real responses

3. **📝 Detailed Examples**:
   - Pre-filled request examples for each endpoint
   - Multiple response scenarios (success, validation errors, server errors)
   - Complete request/response schemas

4. **🔍 Parameter Validation**:
   - Real-time validation of request parameters
   - Type checking and constraint validation
   - Clear error messages for invalid inputs

5. **📊 Response Models**:
   - Complete response schemas with examples
   - Field descriptions and data types
   - Nested model definitions

### Accessing Swagger UI

1. Start the backend server
2. Open your browser and go to: http://localhost:8000/docs
3. Explore the available endpoints organized by tags
4. Click on any endpoint to see detailed documentation
5. Use "Try it out" to test endpoints interactively

## Integration with Open WebUI

To connect this backend with Open WebUI:

### Method 1: Environment Variable

```bash
export OPENAI_API_BASE_URL="http://localhost:8000/v1"
open-webui serve
```

### Method 2: Open WebUI Settings

1. Start Open WebUI
2. Go to Settings → Connections
3. Add a new connection:
   - **Name**: OpenMed Backend
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: (leave empty or use any string)

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Swagger Tag |
|----------|--------|-------------|-------------|
| `/` | GET | Root endpoint with basic info | Server |
| `/health` | GET | Health check endpoint | Server |
| `/docs` | GET | **Swagger UI Documentation** | - |
| `/redoc` | GET | **ReDoc Documentation** | - |
| `/v1/models` | GET | List available models | Models |
| `/v1/chat/completions` | POST | Chat completions (OpenAI compatible) | Chat |
| `/v1/completions` | POST | Text completions (OpenAI compatible) | Completions |

### Documentation Endpoints

- **🔥 Interactive API Docs**: `http://localhost:8000/docs` (Swagger UI)
- **📖 Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)
- **📋 OpenAPI Schema**: `http://localhost:8000/openapi.json`

## Available Models

The backend provides several test models:

- `openmed-chat-v1` - Chat model for conversational AI
- `openmed-completion-v1` - Completion model for text generation  
- `hello-world-model` - Simple test model
- `openmed-medical-v1` - Specialized medical AI model

## Request Examples

### Chat Completion

```json
{
  "model": "openmed-chat-v1",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "stream": false
}
```

### Text Completion

```json
{
  "model": "openmed-completion-v1",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.8,
  "stream": false
}
```

### Streaming Requests

Add `"stream": true` to any completion request to enable streaming responses.

## Configuration

### Environment Variables

- `OPENMED_HOST` - Server host (default: "0.0.0.0")
- `OPENMED_PORT` - Server port (default: "8000")
- `OPENMED_DEBUG` - Debug mode (default: "true")
- `OPENMED_LOG_LEVEL` - Log level (default: "info")

### Custom Configuration

Edit `config.py` to customize:
- Available models
- Default response parameters
- CORS settings
- Error messages

## Development

### Project Structure

```
src/middleware/
├── openai_backend.py    # Main FastAPI application with Swagger UI
├── run_backend.py       # Startup script
├── config.py           # Configuration settings
├── test_backend.py     # Test script
└── README.md           # This file
```

### Swagger UI Customization

The Swagger UI is automatically generated from the FastAPI application with enhanced features:

```python
# Enhanced Pydantic models with Field descriptions
class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use", example="openmed-chat-v1")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    # ... more fields with descriptions and examples

# Detailed endpoint documentation
@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["Chat"],
    summary="Create chat completion",
    description="Creates a model response for the given chat conversation..."
)
```

### Adding Custom Models

1. Edit `config.py` and add your model to `DEFAULT_MODELS`
2. Implement custom logic in the completion endpoints
3. Update Swagger documentation if needed
4. Restart the server

### Adding Custom Responses

Modify the response generation logic in `openai_backend.py`:

```python
# In chat_completions function
response_content = "Your custom response logic here"

# In completions function  
response_text = "Your custom completion logic here"
```

## Testing with Swagger UI

### Manual Testing

1. Open http://localhost:8000/docs
2. Navigate to any endpoint (e.g., "Chat" → "POST /v1/chat/completions")
3. Click "Try it out"
4. Modify the example request or use the default
5. Click "Execute"
6. View the response

### Automated Testing

Use the provided test script:

```bash
cd src/middleware
python test_backend.py
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port using `OPENMED_PORT` environment variable
2. **CORS errors**: Check that your frontend URL is in `ALLOWED_ORIGINS` in `config.py`
3. **Connection refused**: Ensure the backend is running and accessible
4. **Swagger UI not loading**: Check that `/docs` endpoint is accessible

### Logs

The backend provides detailed logging. Check the console output for error messages and request logs.

### Testing

Use the `/health` endpoint to verify the backend is running:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

## API Documentation Features

### Swagger UI Benefits

- **🎯 Interactive Testing**: Test all endpoints without external tools
- **📋 Complete Documentation**: Auto-generated from code with examples
- **🔍 Schema Validation**: Real-time request/response validation
- **📊 Type Safety**: Fully typed models with Pydantic
- **🏷️ Organized Layout**: Endpoints grouped by functionality
- **💡 Examples**: Pre-filled examples for quick testing
- **🔗 OpenAPI Standard**: Standard OpenAPI 3.0+ specification

### Alternative Documentation

- **ReDoc**: More detailed, read-only documentation at `/redoc`
- **OpenAPI JSON**: Raw schema at `/openapi.json` for integration

## Contributing

1. Make your changes
2. Update Swagger documentation (Field descriptions, examples)
3. Test the endpoints using Swagger UI
4. Update documentation if needed
5. Submit your changes

## License

This backend is part of the OpenMed project. 