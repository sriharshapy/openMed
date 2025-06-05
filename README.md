# OpenMed Project

OpenMed is a medical AI project that combines a FastAPI backend middleware with OpenWebUI for an intuitive chat interface. The system provides OpenAI-compatible APIs for medical AI models.

## 🏗️ Project Structure

```
openMed/
├── src/
│   ├── middleware/          # FastAPI backend server
│   ├── open-webui/         # OpenWebUI components
│   ├── models/             # AI models
│   ├── utils/              # Utility functions
│   └── tests/              # Test files
├── weights/                # Model weights
├── venv/                   # Virtual environment
├── requirements.txt        # Python dependencies
├── run_openweb.bat        # OpenWebUI startup script
└── README.md              # This file
```

## 🚀 Installation Instructions

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Git

### 1. Virtual Environment Setup

#### On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Verify activation (you should see (venv) in your prompt)
where python
```

#### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in your prompt)
which python
```

### 2. Install Dependencies

With your virtual environment activated:

```bash
# Install all required packages
pip install -r requirements.txt

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note**: If you don't have a CUDA-compatible GPU, install the CPU version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🏃‍♂️ Running the Application

### Step 1: Start the FastAPI Backend

The FastAPI backend serves as the middleware that provides OpenAI-compatible APIs.

```bash
# Navigate to the middleware directory
cd src/middleware

# Run the backend server
python run_backend.py
```

**Console Output:** The backend will display comprehensive startup information including:
- Server details (host: 0.0.0.0, port: 8000)
- API documentation URLs
- Available endpoints
- OpenWebUI integration instructions

**Available Services:**
- **Main API Server**: http://localhost:8000
- **Swagger UI (Interactive API Docs)**: http://localhost:8000/docs
- **ReDoc (Alternative API Docs)**: http://localhost:8000/redoc
- **OpenAPI Specification**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health
- **Models Endpoint**: http://localhost:8000/v1/models
- **Chat Completions**: http://localhost:8000/v1/chat/completions
- **Embeddings**: http://localhost:8000/v1/embeddings

### Step 2: Start OpenWebUI

OpenWebUI provides the chat interface for interacting with the medical AI models.

#### Option A: Using the provided batch script (Windows)
```bash
# From the project root directory
.\run_openweb.bat
```

**Console Output:** The script will display:
- Environment configuration details
- Server startup information
- Interface access URL
- Backend connection requirements

#### Option B: Manual setup (Cross-platform)

**Set environment variables:**

Windows (PowerShell):
```powershell
$env:OPENAI_API_BASE_URL="http://localhost:8000/v1"
$env:OPENAI_API_KEY="your-api-key"
$env:OLLAMA_BASE_URL=""
$env:ENABLE_OLLAMA_API="false"
$env:ENABLE_OPENAI_API="true"
```

macOS/Linux (Bash):
```bash
export OPENAI_API_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your-api-key"
export OLLAMA_BASE_URL=""
export ENABLE_OLLAMA_API="false"
export ENABLE_OPENAI_API="true"
```

**Start OpenWebUI:**
```bash
open-webui serve --host 0.0.0.0 --port 5000
```

**OpenWebUI Default Configuration:**
- **Interface URL**: http://localhost:5000
- **Default Host**: 0.0.0.0 (accessible from all network interfaces)
- **Default Port**: 5000
- **Backend Integration**: Connects to http://localhost:8000/v1
- **API Key**: Can be any value (e.g., "your-api-key")

## 🔧 Configuration

### API Configuration

The system is configured to use:
- **Backend API**: http://localhost:8000/v1
- **Frontend Interface**: http://localhost:5000
- **API Key**: Set your API key in the environment variables
- **Swagger UI**: http://localhost:8000/docs (for API testing and exploration)

### Model Configuration

Models are configured in `src/middleware/config.py`. You can modify model parameters, add new models, or change model paths there.

## 📝 Usage

1. **Start the Backend**: Follow Step 1 above to start the FastAPI middleware
2. **Start OpenWebUI**: Follow Step 2 to start the web interface
3. **Access the Interface**: Open your browser to http://localhost:5000
4. **Explore APIs**: Visit http://localhost:8000/docs for interactive API documentation
5. **Start Chatting**: Begin interacting with the medical AI models through the chat interface

## 🛠️ API Documentation & Testing

### Swagger UI (Recommended for Development)
- **URL**: http://localhost:8000/docs
- **Features**: 
  - Interactive API testing
  - Request/response examples
  - Parameter documentation
  - Real-time API calls
  - Authentication testing

### ReDoc (Alternative Documentation)
- **URL**: http://localhost:8000/redoc
- **Features**:
  - Clean, readable documentation
  - Code examples in multiple languages
  - Comprehensive schema information

### OpenAPI Specification
- **URL**: http://localhost:8000/openapi.json
- **Usage**: Import into API testing tools like Postman, Insomnia, etc.

## 🐛 Troubleshooting

### Common Issues

**Virtual Environment Not Activating:**
- Make sure you're running the activation command from the project root
- On Windows, you might need to enable script execution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Backend Won't Start:**
- Check if port 8000 is already in use: `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (macOS/Linux)
- Ensure all dependencies are installed: `pip list`

**OpenWebUI Connection Issues:**
- Verify the backend is running on http://localhost:8000
- Check that environment variables are set correctly
- Ensure firewall isn't blocking the ports (8000 and 5000)
- Verify OpenWebUI can reach the backend: test http://localhost:8000/health

**Swagger UI Not Loading:**
- Ensure the backend server is running
- Check browser console for errors
- Try accessing http://localhost:8000/health first
- Clear browser cache if necessary

**CUDA/GPU Issues:**
- Install appropriate PyTorch version for your system
- Check CUDA compatibility: `nvidia-smi` (if you have NVIDIA GPU)

### Development Mode

To run in development mode with auto-reload:

```bash
# Backend with auto-reload (already enabled in run_backend.py)
cd src/middleware
uvicorn openai_backend:app --reload --host 0.0.0.0 --port 8000

# OpenWebUI with development settings
open-webui serve --host 0.0.0.0 --port 5000 --dev
```

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **OpenWebUI Documentation**: https://docs.openwebui.com/
- **Swagger UI Guide**: https://swagger.io/tools/swagger-ui/
- **Project-specific documentation**: Check `src/middleware/README.md` for detailed API information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly using Swagger UI at http://localhost:8000/docs
5. Submit a pull request

## 📄 License

[Add your license information here]

---

**Happy coding! 🚀** 