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
    print("🚀 Starting OpenMed OpenAI-Compatible Backend...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📋 Models Endpoint: http://localhost:8000/v1/models")
    print("\n💡 To connect with Open WebUI, set:")
    print("   export OPENAI_API_BASE_URL='http://localhost:8000/v1'")
    print("   open-webui serve")
    print("\n🛑 Press Ctrl+C to stop the server\n")
    
    try:
        uvicorn.run(
            "openai_backend:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped.")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 