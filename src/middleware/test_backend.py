#!/usr/bin/env python3
"""
Test script for the OpenMed OpenAI-Compatible Backend

This script tests all the main endpoints to ensure they work correctly.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models_endpoint():
    """Test the models listing endpoint"""
    print("\n🔍 Testing models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models endpoint passed: Found {len(data.get('data', []))} models")
            for model in data.get('data', []):
                print(f"   - {model.get('id')}: {model.get('owned_by')}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_chat_completion():
    """Test the chat completions endpoint"""
    print("\n🔍 Testing chat completions endpoint...")
    try:
        payload = {
            "model": "openmed-chat-v1",
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            message = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ Chat completion passed")
            print(f"   Response: {message[:100]}...")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return False

def test_text_completion():
    """Test the text completions endpoint"""
    print("\n🔍 Testing text completions endpoint...")
    try:
        payload = {
            "model": "openmed-completion-v1",
            "prompt": "Hello, world",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{BASE_URL}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            text = data.get('choices', [{}])[0].get('text', '')
            print(f"✅ Text completion passed")
            print(f"   Response: {text[:100]}...")
            return True
        else:
            print(f"❌ Text completion failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Text completion error: {e}")
        return False

def test_streaming_chat():
    """Test streaming chat completions"""
    print("\n🔍 Testing streaming chat completions...")
    try:
        payload = {
            "model": "openmed-chat-v1",
            "messages": [
                {"role": "user", "content": "Tell me a short story"}
            ],
            "stream": True,
            "max_tokens": 30
        }
        
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming chat started successfully")
            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        chunk_count += 1
                        if chunk_count <= 3:  # Show first 3 chunks
                            print(f"   Chunk {chunk_count}: {line[:50]}...")
                        if '[DONE]' in line:
                            break
            print(f"✅ Streaming completed with {chunk_count} chunks")
            return True
        else:
            print(f"❌ Streaming chat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Streaming chat error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint passed: {data.get('message')}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting OpenMed Backend Tests...")
    print(f"📡 Testing server at: {BASE_URL}")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        test_health_endpoint,
        test_root_endpoint,
        test_models_endpoint,
        test_chat_completion,
        test_text_completion,
        test_streaming_chat
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 