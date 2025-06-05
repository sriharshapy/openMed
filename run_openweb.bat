@echo off
echo ======================================================================
echo 🌐 Starting OpenMed OpenWebUI Interface
echo ======================================================================
echo.
echo 🔧 Environment Configuration:
echo    • OpenAI API Base URL: http://localhost:8000/v1
echo    • OpenAI API Key: your-api-key
echo    • Ollama API: Disabled
echo    • OpenAI API: Enabled
echo.
echo 📡 Setting environment variables...
set OPENAI_API_BASE_URL=http://localhost:8000/v1
set OPENAI_API_KEY=your-api-key
set OLLAMA_BASE_URL=
set ENABLE_OLLAMA_API=false
set ENABLE_OPENAI_API=true

echo.
echo 🚀 Starting OpenWebUI Server...
echo    • Host: 0.0.0.0 (accessible from all interfaces)
echo    • Port: 5000
echo    • Interface URL: http://localhost:5000
echo.
echo 💡 Make sure the FastAPI backend is running on http://localhost:8000
echo    Run: cd src/middleware && python run_backend.py
echo.
echo 🛑 Press Ctrl+C to stop OpenWebUI
echo ======================================================================
echo.

open-webui serve --host 0.0.0.0 --port 5000

echo.
echo ======================================================================
echo 🛑 OpenMed OpenWebUI Interface Stopped
echo ======================================================================