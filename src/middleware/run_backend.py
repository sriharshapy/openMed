#!/usr/bin/env python3
"""
Startup script for the OpenMed OpenAI-Compatible Backend

This script runs the FastAPI backend server with proper configuration.
"""

import uvicorn
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to run the backend server"""
    print("=" * 70)
    print("🚀 Starting OpenMed OpenAI-Compatible Backend Server")
    print("=" * 70)
    print()
    print("📡 Backend Server Details:")
    print("   • Server URL: http://localhost:8000")
    print("   • Host: 0.0.0.0 (accessible from all interfaces)")
    print("   • Port: 8000")
    print("   • Auto-reload: Enabled (development mode)")
    print()
    print("📚 API Documentation & Testing:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("   • OpenAPI JSON: http://localhost:8000/openapi.json")
    print()
    print("🔗 Available Endpoints:")
    print("   • Health Check: http://localhost:8000/health")
    print("   • Models List: http://localhost:8000/v1/models")
    print("   • Chat Completions: http://localhost:8000/v1/chat/completions")
    print("   • Embeddings: http://localhost:8000/v1/embeddings")
    print()
    print("💡 OpenWebUI Integration:")
    print("   To connect OpenWebUI to this backend, use these settings:")
    print("   • API Base URL: http://localhost:8000/v1")
    print("   • API Key: your-api-key (or any value)")
    print()
    print("🔧 Quick Start Commands:")
    print("   # Set environment and start OpenWebUI")
    print("   export OPENAI_API_BASE_URL='http://localhost:8000/v1'")
    print("   export OPENAI_API_KEY='your-api-key'")
    print("   open-webui serve --host 0.0.0.0 --port 5000")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    try:
        uvicorn.run(
            "openai_backend:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("🛑 OpenMed Backend Server Stopped Successfully")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Troubleshooting tips:")
        print("   • Check if port 8000 is already in use")
        print("   • Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   • Verify you're in the correct directory")
        sys.exit(1)

if __name__ == "__main__":
    main() 