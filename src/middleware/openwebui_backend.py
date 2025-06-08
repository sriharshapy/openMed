#!/usr/bin/env python3
"""
OpenMed OpenWebUI-Compatible Backend

A comprehensive FastAPI backend that provides OpenAI-compatible API endpoints
with file upload support for use with Open WebUI and other applications.

This module focuses on API routing and request handling, with business logic
abstracted into separate service modules for better maintainability.

Features:
- OpenAI API compatibility (/v1/models, /v1/chat/completions, /v1/completions)
- File upload and processing support
- Streaming responses
- CORS enabled
- Comprehensive validation
- Modular architecture with abstracted services
- Agentic workflow for medical intent classification
- Integration with ResNet50 pneumonia detection model
"""

import json
import logging
import os
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from config import (
    ALLOWED_ORIGINS,
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DEFAULT_MODELS,
    LOG_LEVEL,
)
from models import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    FileUploadResponse,
    HealthResponse,
    ModelListResponse,
    ProcessedFileInfo,
)
from api_utils import (
    ImageProcessor, 
    MessageProcessor, 
    ResponseGenerator, 
    StreamingUtils, 
    ResponseBuilder,
    APIResponseUtils
)
from services import (
    MedicalAnalysisService,
    WorkflowService,
    FileService,
    ResponseService
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = "open-webui/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create GradCAM directory
GRADCAM_DIR = "src/gradcam_images"
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Global file storage for demonstration
uploaded_files: Dict[str, ProcessedFileInfo] = {}

# Removed session tracking - now only analyzing current request images


@app.get("/", tags=["Server"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "OpenMed OpenWebUI-Compatible Backend",
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "file_upload": "/v1/files/upload",
            "gradcam_list": "/v1/gradcam",
            "gradcam_image": "/v1/gradcam/{filename}",
            "medical_analyze": "/v1/medical/analyze",
            "gradcam_test": "/v1/medical/gradcam-test",
            "stream_demo": "/v1/stream-demo",
            "documentation": "/docs",
            "openwebui": {
                "chat_completions": "/api/chat/completions",
                "file_upload": "/api/v1/files/",
                "file_list": "/api/v1/files/"
            }
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Server"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
        version=API_VERSION,
        uptime_seconds=0,  # Simplified for demo
    )


@app.get("/v1/models", response_model=ModelListResponse, tags=["Models"])
async def list_models():
    """List available models (OpenAI compatible)"""
    return ModelListResponse(
        object="list",
        data=DEFAULT_MODELS,
    )


@app.post("/v1/files/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(file: UploadFile = File(...), metadata: Optional[str] = Form(None)):
    """Upload a file for processing
    
    This endpoint allows file uploads that can be referenced in chat completions.
    Supported file types include text documents, PDFs, images, and more.
    """
    try:
        logger.info(f"🚀 FILE UPLOAD REQUEST RECEIVED")
        logger.info(f"   📄 Filename: {file.filename}")
        logger.info(f"   📦 Content-Type: {file.content_type}")
        logger.info(f"   📏 Size: {file.size} bytes" if file.size else "   📏 Size: Unknown")
        
        # Process metadata if provided
        file_metadata = {}
        if metadata:
            try:
                file_metadata = json.loads(metadata)
                logger.info(f"   📋 Metadata: {file_metadata}")
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Invalid metadata JSON: {metadata}")
        
        # Save file using FileService
        logger.info("💾 Starting file save process...")
        file_info = await FileService.save_uploaded_file(file, UPLOAD_DIR)
        
        # Store in global storage
        uploaded_files[file_info.id] = file_info
        logger.info(f"🗃️ File stored in memory with ID: {file_info.id}")
        logger.info(f"🎉 FILE UPLOAD COMPLETED SUCCESSFULLY!")
        
        return FileUploadResponse(
            file_id=file_info.id,
            filename=file_info.filename,
            size=file_info.size,
            mime_type=file_info.mime_type,
            upload_time=file_info.upload_time,
            message="File uploaded successfully",
            metadata=file_metadata,
        )
        
    except Exception as e:
        logger.error(f"💥 ERROR IN FILE UPLOAD ENDPOINT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", tags=["Chat"])
async def chat_completions(request: ChatCompletionRequest):
    """Create a chat completion (OpenAI compatible) with agentic workflow"""
    try:
        logger.info(f"🚀 Processing chat completion request for model: {request.model}")
        
        # Process base64 images in message content first
        saved_image_ids = await MessageProcessor.extract_and_save_images(
            request.messages, uploaded_files, UPLOAD_DIR
        )
        
        if saved_image_ids:
            logger.info(f"📸 Processed {len(saved_image_ids)} base64 images from message content")
        
        # Extract content and context
        user_message = MessageProcessor.extract_user_message(request.messages) 
        file_context = MessageProcessor.get_file_context(request, uploaded_files)
        
        # INTELLIGENT APPROACH: Only process images when there's a clear medical intent
        # First check if the user is actually asking for medical analysis
        
        # Quick pre-analysis of user intent to determine if we should process any images at all
        medical_analysis_keywords = ["analyze", "check", "detect", "examine", "scan", "tumor", "pneumonia", "tb", "tuberculosis"]
        diagnosis_summary_keywords = ["diagnosis", "full diagnosis", "overall diagnosis", "summary", "comprehensive", "based on our chat", "what did you find", "results summary"]
        latest_user_message = user_message.lower()
        
        # Check if the current message has medical intent for NEW analysis
        has_medical_intent = any(keyword in latest_user_message for keyword in medical_analysis_keywords)
        
        # Check if user is asking for a summary/diagnosis of previous results
        wants_diagnosis_summary = any(keyword in latest_user_message for keyword in diagnosis_summary_keywords)
        
        images_to_analyze = []
        
        if has_medical_intent:
            logger.info(f"🔬 Medical intent detected in message: '{user_message[:100]}...'")
            
            # Only include NEW base64 images from the CURRENT message
            current_message_images = [saved_image_ids.copy()[-1]] if saved_image_ids else []
            
            # For file references, only include if they're part of a medical request  
            latest_message_file_ids = []
            if hasattr(request, 'files') and request.files:
                for file_ref in request.files:
                    file_id = file_ref.id if hasattr(file_ref, 'id') else file_ref.get('id', '')
                    clean_file_id = file_id.replace("file-", "") if file_id.startswith("file-") else file_id
                    
                    if clean_file_id in uploaded_files:
                        latest_message_file_ids.append(clean_file_id)
                    elif file_id in uploaded_files:
                        latest_message_file_ids.append(file_id)
            
            # Combine images ONLY if there's medical intent
            images_to_analyze = current_message_images + latest_message_file_ids
            
            logger.info(f"🎯 MEDICAL ANALYSIS MODE")
            logger.info(f"📸 Base64 images from current message: {len(current_message_images)} - {current_message_images}")
            logger.info(f"📎 File references: {len(latest_message_file_ids)} - {latest_message_file_ids}")
            logger.info(f"🔬 Total images to analyze: {len(images_to_analyze)} - {images_to_analyze}")
        elif wants_diagnosis_summary:
            logger.info(f"📋 DIAGNOSIS SUMMARY REQUEST detected: '{user_message[:100]}...'")
            logger.info(f"📝 TEXT-ONLY ANALYSIS - No images required for diagnosis summary")
            logger.info(f"🔍 Will analyze conversation history and provide comprehensive diagnosis")
            # Don't analyze new images, but pass the conversation context for summary
        else:
            logger.info(f"💬 GENERAL CONVERSATION MODE - No medical intent detected")
            logger.info(f"🚫 Skipping all image processing for general questions")
            logger.info(f"📝 User message: '{user_message[:100]}...'")
        
        # Process agentic workflow using WorkflowService - handle both analysis and summary requests
        workflow_result = await WorkflowService.process_agentic_workflow(
            user_message, images_to_analyze, uploaded_files, wants_diagnosis_summary, request.messages
        )
        
        # Generate response using ResponseService
        response_content = await ResponseService.generate_intelligent_response(
            user_message, workflow_result, file_context, request.model
        )
        
        # Handle streaming
        if request.stream:
            completion_id = ResponseGenerator.create_completion_id()
            # Get base URL for image serving (more robust extraction)
            from fastapi import Request as FastAPIRequest
            try:
                # Try to get base URL from request context
                if hasattr(request, 'scope') and request.scope.get('headers'):
                    headers = dict(request.scope['headers'])
                    host = headers.get(b'host', b'localhost:8000').decode()
                    x_forwarded_proto = headers.get(b'x-forwarded-proto', b'http').decode()
                    scheme = "https" if x_forwarded_proto == "https" else "http"
                    base_url = f"{scheme}://{host}"
                else:
                    base_url = "http://localhost:8000"
            except Exception:
                base_url = "http://localhost:8000"
            
            return StreamingResponse(
                StreamingUtils.generate_optimized_medical_stream(
                    workflow_result, user_message, file_context, request.model, completion_id, base_url
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Non-streaming response
        completion_id = ResponseGenerator.create_completion_id()
        prompt_tokens = ResponseGenerator.calculate_tokens(user_message)
        completion_tokens = ResponseGenerator.calculate_tokens(response_content)
        
        logger.info(f"✅ Chat completion successful - Response length: {len(response_content)} chars")
        
        return ResponseBuilder.build_chat_completion_response(
            completion_id, request.model, response_content, prompt_tokens, completion_tokens
        )
        
    except Exception as e:
        logger.error(f"❌ Error in chat completion: {str(e)}")
        logger.error(f"💥 Request details: model={getattr(request, 'model', 'unknown')}, messages_count={len(getattr(request, 'messages', []))}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/completions", tags=["Chat", "OpenWebUI"])
async def openwebui_chat_completions(request: ChatCompletionRequest):
    """OpenWebUI-specific chat completions endpoint (same as /v1/chat/completions)"""
    logger.info(f"🌐 OpenWebUI chat completion request received for model: {request.model}")
    return await chat_completions(request)


@app.post("/v1/completions", tags=["Completions"])
async def completions(request: CompletionRequest):
    """Create a text completion (OpenAI compatible)"""
    try:
        # Extract prompt
        prompt = request.prompt if isinstance(request.prompt, str) else " ".join(request.prompt)
        
        # Generate response content using ResponseService
        response_content = ResponseService.generate_hello_world_response(prompt)
        
        # Handle streaming
        if request.stream:
            return StreamingResponse(
                StreamingUtils.generate_completion_stream(response_content, request.model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        
        # Non-streaming response
        completion_id = ResponseGenerator.create_completion_id("cmpl")
        prompt_tokens = ResponseGenerator.calculate_tokens(prompt)
        completion_tokens = ResponseGenerator.calculate_tokens(response_content)
        
        return ResponseBuilder.build_completion_response(
            completion_id, request.model, response_content, prompt_tokens, completion_tokens
        )
        
    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files", tags=["Files"])
async def list_uploaded_files():
    """List all uploaded files"""
    logger.info(f"📋 Listing uploaded files - Total: {len(uploaded_files)}")
    return FileService.get_file_list_info(uploaded_files, UPLOAD_DIR)


@app.post("/v1/session/clear", tags=["Session"])
async def clear_session_tracker():
    """Clear uploaded files cache (legacy endpoint - session tracking removed)"""
    global uploaded_files
    file_count = len(uploaded_files)
    uploaded_files.clear()
    logger.info(f"🧹 Cleared {file_count} uploaded file(s) from cache")
    return {"message": f"Cleared {file_count} uploaded file(s) from cache", "status": "success"}


@app.post("/api/v1/files/", response_model=Dict[str, Any], tags=["Files", "OpenWebUI"])
async def openwebui_upload_file(file: UploadFile = File(...)):
    """OpenWebUI-specific file upload endpoint"""
    try:
        logger.info(f"🌐 OpenWebUI file upload request: {file.filename}")
        
        # Use the FileService for upload logic
        file_info = await FileService.save_uploaded_file(file, UPLOAD_DIR)
        uploaded_files[file_info.id] = file_info
        
        # Return OpenWebUI-compatible response using APIResponseUtils
        return APIResponseUtils.format_openwebui_file_response(file_info)
        
    except Exception as e:
        logger.error(f"❌ Error in OpenWebUI file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/files/", tags=["Files", "OpenWebUI"])
async def openwebui_list_files():
    """OpenWebUI-specific file list endpoint"""
    logger.info(f"🌐 OpenWebUI file list request")
    
    files_list = []
    for file_id, file_info in uploaded_files.items():
        files_list.append(APIResponseUtils.format_openwebui_file_response(file_info))
    
    return {
        "object": "list",
        "data": files_list
    }


@app.post("/v1/medical/analyze", tags=["Medical"])
async def medical_analyze_endpoint(request: Dict[str, Any]):
    """Endpoint to test medical analysis directly"""
    prompt = request.get('prompt', '')
    file_ids = request.get('file_ids', [])
    
    try:
        # Run the agentic workflow using WorkflowService
        workflow_result = await WorkflowService.process_agentic_workflow(
            prompt, file_ids, uploaded_files
        )
        
        return APIResponseUtils.create_success_response({
            "workflow_result": workflow_result,
            "prompt": prompt,
            "file_ids": file_ids
        }, "Medical analysis completed")
        
    except Exception as e:
        logger.error(f"❌ Error in medical analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/medical/gradcam-test", tags=["Medical", "Testing"])
async def gradcam_test_endpoint(request: Dict[str, Any]):
    """Test endpoint for GradCAM generation"""
    file_id = request.get('file_id', '')
    
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = uploaded_files[file_id]
        if not hasattr(file_info, 'file_path') or not os.path.exists(file_info.file_path):
            raise HTTPException(status_code=404, detail="File not accessible")
        
        # Test GradCAM generation directly
        from services import MedicalAnalysisService
        result = await MedicalAnalysisService.analyze_image_for_pneumonia(
            file_info.file_path, generate_gradcam=True
        )
        
        return APIResponseUtils.create_success_response({
            "analysis_result": result,
            "file_info": {
                "id": file_info.id,
                "filename": file_info.filename,
                "path": file_info.file_path
            }
        }, "GradCAM test completed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in GradCAM test endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/stream-demo", tags=["Testing", "Streaming"])
async def stream_demo():
    """Demo endpoint showing mixed text and image streaming"""
    import asyncio
    
    async def generate_demo_stream():
        # Example: Stream text and image chunks
        texts = [
            "🎯 **Medical AI Analysis Demo**\n\n",
            "This is a demonstration of real-time streaming with both text and images.\n\n",
            "📸 **Sample Analysis Result:**\n",
            "- Prediction: **Normal**\n",
            "- Confidence: **92.5%**\n\n",
            "🎨 **GradCAM Visualization:**\n"
        ]
        
        # Example: 1x1 red pixel PNG for demo
        demo_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBAp6kAAAAAElFTkSuQmCC"
        
        for text in texts:
            chunk = json.dumps({"type": "text", "content": text}) + "\n"
            yield chunk.encode('utf-8')
            await asyncio.sleep(0.5)
        
        # Stream an image chunk
        image_chunk = json.dumps({
            "type": "image_url", 
            "url": f"data:image/png;base64,{demo_image_base64}"
        }) + "\n"
        yield image_chunk.encode('utf-8')
        await asyncio.sleep(0.5)
        
        # Final text
        final_text = json.dumps({
            "type": "text", 
            "content": "\n*Demo GradCAM visualization*\n\n✅ **Demo completed successfully!**"
        }) + "\n"
        yield final_text.encode('utf-8')
    
    return StreamingResponse(generate_demo_stream(), media_type="application/json")


@app.get("/v1/files/{file_id}", tags=["Files"])
async def get_file_info(file_id: str):
    """Get information about an uploaded file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    file_exists = os.path.exists(file_info.file_path) if hasattr(file_info, 'file_path') else False
    
    return {
        "id": file_info.id,
        "filename": file_info.filename,
        "safe_filename": getattr(file_info, 'safe_filename', None),
        "size": file_info.size,
        "mime_type": file_info.mime_type,
        "upload_time": file_info.upload_time,
        "processed": file_info.processed,
        "file_exists": file_exists,
        "file_path": getattr(file_info, 'file_path', None)
    }


@app.get("/v1/gradcam/{filename}", tags=["Medical", "Images"])
async def get_gradcam_image(filename: str):
    """Serve GradCAM visualization images"""
    try:
        # Construct the full path to the GradCAM image
        gradcam_path = os.path.join(GRADCAM_DIR, filename)
        
        # Check if file exists and is within the GradCAM directory (security check)
        if not os.path.exists(gradcam_path):
            raise HTTPException(status_code=404, detail="GradCAM image not found")
        
        # Ensure the path is within the GradCAM directory (prevent directory traversal)
        gradcam_abs_path = os.path.abspath(gradcam_path)
        gradcam_dir_abs = os.path.abspath(GRADCAM_DIR)
        
        if not gradcam_abs_path.startswith(gradcam_dir_abs):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Return the image file with proper caching headers
        return FileResponse(
            path=gradcam_abs_path,
            media_type="image/png",
            filename=filename,
            headers={
                "Cache-Control": "public, max-age=3600",  # Enable caching for 1 hour
                "ETag": f'"{filename}"',
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error serving GradCAM image {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving GradCAM image")


@app.get("/v1/gradcam", tags=["Medical", "Images"])
async def list_gradcam_images():
    """List available GradCAM images"""
    try:
        gradcam_files = []
        
        if os.path.exists(GRADCAM_DIR):
            for filename in os.listdir(GRADCAM_DIR):
                file_path = os.path.join(GRADCAM_DIR, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    stat = os.stat(file_path)
                    gradcam_files.append({
                        "filename": filename,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
                        "url": f"/v1/gradcam/{filename}"
                    })
        
        return {
            "gradcam_images": gradcam_files,
            "total_count": len(gradcam_files),
            "gradcam_directory": os.path.abspath(GRADCAM_DIR)
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing GradCAM images: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing GradCAM images")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler"""
    return ErrorResponse(
        error={
            "message": exc.detail,
            "type": "http_error",
            "code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return ErrorResponse(
        error={
            "message": "Internal server error",
            "type": "server_error",
            "code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "openwebui_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=LOG_LEVEL.lower(),
    )
