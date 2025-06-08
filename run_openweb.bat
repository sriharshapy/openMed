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
echo    • Data Directory: %CD%\open-webui
echo.
echo 📡 Setting environment variables...

REM Core API Configuration
set OPENAI_API_BASE_URL=http://localhost:8000/v1
set OPENAI_API_KEY=your-api-key
set OLLAMA_BASE_URL=
set ENABLE_OLLAMA_API=false
set ENABLE_OPENAI_API=true

REM Data Directory (CRITICAL for file uploads)
set DATA_DIR=%CD%\open-webui

REM File Upload Configuration
set RAG_FILE_MAX_SIZE=200
set RAG_FILE_MAX_COUNT=10
set UPLOAD_FILE_MAX_SIZE=200

REM Disable problematic features if needed
set PDF_EXTRACT_IMAGES=false
set RAG_EXTRACT_IMAGES=false

REM Timeout settings for large files
set WORKER_TIMEOUT=300
set REQUEST_TIMEOUT=300

REM Create data directory if it doesn't exist
if not exist "%DATA_DIR%" (
    echo 📁 Creating data directory: %DATA_DIR%
    mkdir "%DATA_DIR%"
    echo    ✅ Data directory created successfully
) else (
    echo 📁 Data directory exists: %DATA_DIR%
)

echo.
echo 🚀 Starting OpenWebUI Server...
echo    • Host: 0.0.0.0 (accessible from all interfaces)
echo    • Port: 3000
echo    • Interface URL: http://localhost:3000
echo    • Data Directory: %DATA_DIR%
echo    • Max File Size: %RAG_FILE_MAX_SIZE%MB
echo.
echo 💡 Make sure the FastAPI backend is running on http://localhost:8000
echo    Run: cd src/middleware && python run_backend.py
echo.
echo 🛑 Press Ctrl+C to stop OpenWebUI
echo ======================================================================
echo.

REM Verify data directory exists before starting
if not exist "%DATA_DIR%" (
    echo ❌ ERROR: Data directory could not be created!
    echo Please check permissions and try again.
    pause
    exit /b 1
)

echo 🎯 Final Environment Check:
echo    • DATA_DIR: %DATA_DIR%
echo    • RAG_FILE_MAX_SIZE: %RAG_FILE_MAX_SIZE%MB
echo    • OPENAI_API_BASE_URL: %OPENAI_API_BASE_URL%
echo.

open-webui serve --host 0.0.0.0 --port 3000

echo.
echo ======================================================================
echo 🛑 OpenMed OpenWebUI Interface Stopped
echo ======================================================================
pause