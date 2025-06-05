#!/usr/bin/env python3
"""
Swagger UI Demonstration Script for OpenMed Backend

This script demonstrates how to use the enhanced Swagger UI features
and provides examples of interacting with the API programmatically.
"""

import webbrowser
import time
import requests
import json
from typing import Dict, Any

def open_swagger_ui():
    """Open the Swagger UI in the default web browser"""
    swagger_url = "http://localhost:8000/docs"
    print("🔥 Opening Swagger UI in your browser...")
    print(f"📡 URL: {swagger_url}")
    
    try:
        webbrowser.open(swagger_url)
        print("✅ Swagger UI opened successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print(f"💡 Please manually open: {swagger_url}")
        return False

def check_server_running():
    """Check if the backend server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running!")
            return True
        else:
            print(f"❌ Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the backend server.")
        print("💡 Make sure to run: python run_backend.py")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

def demonstrate_api_calls():
    """Demonstrate API calls that can be tested in Swagger UI"""
    print("\n📋 API Examples (try these in Swagger UI):")
    print("=" * 60)
    
    examples = [
        {
            "title": "1. Health Check",
            "method": "GET",
            "endpoint": "/health",
            "description": "Check if the server is healthy",
            "curl": "curl http://localhost:8000/health"
        },
        {
            "title": "2. List Models",
            "method": "GET", 
            "endpoint": "/v1/models",
            "description": "Get all available models",
            "curl": "curl http://localhost:8000/v1/models"
        },
        {
            "title": "3. Chat Completion",
            "method": "POST",
            "endpoint": "/v1/chat/completions",
            "description": "Generate a chat response",
            "payload": {
                "model": "openmed-chat-v1",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            },
            "curl": """curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "openmed-chat-v1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'"""
        },
        {
            "title": "4. Text Completion",
            "method": "POST",
            "endpoint": "/v1/completions", 
            "description": "Generate text completion",
            "payload": {
                "model": "openmed-completion-v1",
                "prompt": "Once upon a time",
                "max_tokens": 30,
                "temperature": 0.8
            },
            "curl": """curl -X POST http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "openmed-completion-v1", 
    "prompt": "Once upon a time",
    "max_tokens": 30
  }'"""
        },
        {
            "title": "5. Streaming Chat",
            "method": "POST",
            "endpoint": "/v1/chat/completions",
            "description": "Generate streaming chat response",
            "payload": {
                "model": "openmed-chat-v1",
                "messages": [
                    {"role": "user", "content": "Tell me a story"}
                ],
                "stream": True,
                "max_tokens": 50
            },
            "curl": """curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "openmed-chat-v1",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true,
    "max_tokens": 50
  }'"""
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}")
        print(f"Method: {example['method']}")
        print(f"Endpoint: {example['endpoint']}")
        print(f"Description: {example['description']}")
        
        if 'payload' in example:
            print("Payload:")
            print(json.dumps(example['payload'], indent=2))
        
        print("cURL Command:")
        print(example['curl'])
        print("-" * 40)

def swagger_ui_guide():
    """Provide a guide for using Swagger UI"""
    print("\n🎯 Swagger UI Usage Guide:")
    print("=" * 60)
    
    steps = [
        "1. 🌐 Open http://localhost:8000/docs in your browser",
        "2. 📋 Browse endpoints organized by tags (Server, Models, Chat, Completions)",
        "3. 🔍 Click on any endpoint to see detailed documentation",
        "4. 📝 Review the request/response schemas and examples",
        "5. 🎯 Click 'Try it out' to test the endpoint interactively",
        "6. ✏️  Modify the example request or use the default values",
        "7. ▶️  Click 'Execute' to send the request",
        "8. 📊 View the response body, headers, and status code",
        "9. 📋 Copy the curl command for use in other applications",
        "10. 🔄 Repeat for other endpoints to explore the API"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print("\n💡 Tips:")
    print("   • Use the 'Models' endpoint first to see available models")
    print("   • Try both streaming and non-streaming requests")
    print("   • Experiment with different temperature values (0.0 - 2.0)")
    print("   • Check the response schemas for integration guidance")

def test_api_programmatically():
    """Test the API programmatically to show working examples"""
    print("\n🧪 Testing API Programmatically:")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"   ✅ Health: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test models endpoint
    try:
        print("\n2. Testing models endpoint...")
        response = requests.get(f"{base_url}/v1/models")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {len(data.get('data', []))} models")
            for model in data.get('data', [])[:2]:  # Show first 2 models
                print(f"      - {model.get('id')}")
        else:
            print(f"   ❌ Models request failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test chat completion
    try:
        print("\n3. Testing chat completion...")
        payload = {
            "model": "openmed-chat-v1",
            "messages": [{"role": "user", "content": "Hello from Python!"}],
            "max_tokens": 30
        }
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"   ✅ Chat response: {content[:50]}...")
        else:
            print(f"   ❌ Chat completion failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Main demonstration function"""
    print("🚀 OpenMed Backend Swagger UI Demonstration")
    print("=" * 60)
    
    # Check if server is running
    if not check_server_running():
        print("\n❌ Backend server is not running!")
        print("💡 Please start the server first:")
        print("   cd src/middleware")
        print("   python run_backend.py")
        return
    
    print("\n🎉 Backend server is ready!")
    
    # Show Swagger UI guide
    swagger_ui_guide()
    
    # Show API examples
    demonstrate_api_calls()
    
    # Test API programmatically
    test_api_programmatically()
    
    # Open Swagger UI
    print("\n" + "=" * 60)
    open_swagger_ui()
    
    print("\n📖 Additional Documentation:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("   • OpenAPI Schema: http://localhost:8000/openapi.json")
    
    print("\n🎯 Next Steps:")
    print("   1. Explore the Swagger UI in your browser")
    print("   2. Try the 'Try it out' feature on each endpoint")
    print("   3. Test different parameter values")
    print("   4. Copy curl commands for your applications")
    print("   5. Use the API in your own projects")

if __name__ == "__main__":
    main() 