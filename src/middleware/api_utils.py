#!/usr/bin/env python3
"""
API Utilities for OpenMed OpenWebUI-Compatible Backend

This module contains utility classes for processing requests, generating responses,
handling streaming, and building API responses. These utilities are designed to
be reusable across different API endpoints.

Classes:
- MessageProcessor: Handles message content extraction and processing
- ResponseGenerator: Generates responses and provides helper functions
- StreamingUtils: Handles streaming response generation
- ResponseBuilder: Builds standardized API responses
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import uuid
from datetime import datetime, UTC
from typing import Any, AsyncGenerator, Dict, List, Union, Tuple, Optional

from config import HELLO_WORLD_RESPONSES
from models import ChatCompletionResponse, CompletionResponse, ProcessedFileInfo

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Utility class for processing base64 images"""
    
    @staticmethod
    def is_base64_image(url: str) -> bool:
        """Check if URL is a base64 encoded image"""
        return url.startswith("data:image/") and "base64," in url
    
    @staticmethod
    def extract_base64_info(url: str) -> Tuple[str, str, bytes]:
        """Extract MIME type, format, and image data from base64 URL"""
        try:
            # Format: data:image/jpeg;base64,/9j/4AAQSkZJRgAB...
            header, base64_data = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]  # image/jpeg
            image_format = mime_type.split("/")[1]  # jpeg
            image_data = base64.b64decode(base64_data)
            return mime_type, image_format, image_data
        except Exception as e:
            logger.error(f"Error extracting base64 image info: {str(e)}")
            raise ValueError(f"Invalid base64 image URL: {str(e)}")
    
    @staticmethod
    async def save_base64_image(url: str, upload_dir: str = "open-webui/uploads") -> ProcessedFileInfo:
        """Save base64 image to file and return ProcessedFileInfo"""
        try:
            # Extract image data
            mime_type, image_format, image_data = ImageProcessor.extract_base64_info(url)
            
            # Generate filename
            filename = f"base64_image_{uuid.uuid4().hex[:8]}.{image_format}"
            file_path = os.path.join(upload_dir, filename)
            absolute_path = os.path.abspath(file_path)
            
            # Ensure upload directory exists
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save image data
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            # Verify file was saved
            if not os.path.exists(file_path):
                raise Exception(f"Failed to save base64 image to {absolute_path}")
            
            logger.info(f"✅ BASE64 IMAGE SAVED:")
            logger.info(f"   📁 Filename: {filename}")
            logger.info(f"   📍 Path: {absolute_path}")
            logger.info(f"   📏 Size: {len(image_data)} bytes")
            logger.info(f"   🎭 MIME type: {mime_type}")
            
            # Create file info
            file_info = ProcessedFileInfo(
                id=uuid.uuid4().hex,
                filename=filename,
                safe_filename=filename,
                file_path=absolute_path,
                size=len(image_data),
                mime_type=mime_type,
                upload_time=datetime.now(UTC),
                processed=True,
            )
            
            return file_info
            
        except Exception as e:
            logger.error(f"❌ ERROR SAVING BASE64 IMAGE: {str(e)}")
            raise


class MessageProcessor:
    """Utility class for processing messages and extracting content"""
    
    @staticmethod
    def extract_user_message(messages: List[Any]) -> str:
        """Extract the LATEST user message content from messages list only"""
        # Extract only the most recent user message to avoid historical GradCAM references
        user_messages = []
        for message in messages:
            if message.role == "user":
                user_messages.append(MessageProcessor._extract_content(message.content))
        
        # Return only the latest user message to prevent historical GradCAM leakage
        return user_messages[-1] if user_messages else ""
    
    @staticmethod
    def _extract_content(content: Union[str, List[Any]]) -> str:
        """Extract text content from message content (string or list)"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if hasattr(item, 'type') and item.type == "text" and hasattr(item, 'text'):
                    text_parts.append(item.text)
                elif hasattr(item, 'type') and item.type == "image_url":
                    # Add placeholder for image content
                    text_parts.append("[Image attached]")
            return " ".join(text_parts)
        return ""
    
    @staticmethod
    async def extract_and_save_images(messages: List[Any], uploaded_files: Dict[str, Any], upload_dir: str = "open-webui/uploads") -> List[str]:
        """Extract base64 images from messages, save them, and return file IDs"""
        saved_file_ids = []
        
        for message in messages:
            if message.role == "user" and isinstance(message.content, list):
                for item in message.content:
                    if (hasattr(item, 'type') and item.type == "image_url" and 
                        hasattr(item, 'image_url') and item.image_url and 
                        'url' in item.image_url):
                        
                        url = item.image_url['url']
                        if ImageProcessor.is_base64_image(url):
                            try:
                                logger.info(f"📸 Processing base64 image in message content")
                                file_info = await ImageProcessor.save_base64_image(url, upload_dir)
                                uploaded_files[file_info.id] = file_info
                                saved_file_ids.append(file_info.id)
                                logger.info(f"💾 Saved base64 image with ID: {file_info.id}")
                            except Exception as e:
                                logger.error(f"❌ Failed to process base64 image: {str(e)}")
        
        return saved_file_ids
    
    @staticmethod
    def get_file_context(request: Any, uploaded_files: Dict[str, Any]) -> str:
        """Extract file context from request"""
        file_context = ""
        
        # Check for files in request.files (file references)
        if hasattr(request, 'files') and request.files:
            for file_ref in request.files:
                file_id = file_ref.id if hasattr(file_ref, 'id') else file_ref.get('id', '')
                
                # Handle OpenWebUI file ID format (file-xxxxx)
                clean_file_id = file_id.replace("file-", "") if file_id.startswith("file-") else file_id
                
                if clean_file_id in uploaded_files:
                    file_info = uploaded_files[clean_file_id]
                    file_context += f"\n[File attached: {file_info.filename} ({file_info.mime_type})]"
                elif file_id in uploaded_files:
                    file_info = uploaded_files[file_id]
                    file_context += f"\n[File attached: {file_info.filename} ({file_info.mime_type})]"
        
        # Check for images in message content (base64 images that were processed)
        if hasattr(request, 'messages'):
            for message in request.messages:
                if message.role == "user" and isinstance(message.content, list):
                    for item in message.content:
                        if (hasattr(item, 'type') and item.type == "image_url" and 
                            hasattr(item, 'image_url') and item.image_url and 
                            'url' in item.image_url):
                            url = item.image_url['url']
                            if ImageProcessor.is_base64_image(url):
                                try:
                                    mime_type, image_format, _ = ImageProcessor.extract_base64_info(url)
                                    file_context += f"\n[Base64 image attached: {image_format.upper()} image ({mime_type})]"
                                except:
                                    file_context += "\n[Base64 image attached: Unknown format]"
        
        return file_context


class ResponseGenerator:
    """Response generation utility class"""
    
    @staticmethod
    def generate_hello_world_response(user_message: str = "") -> str:
        """Generate a hello world response"""
        import random
        base_response = random.choice(HELLO_WORLD_RESPONSES)
        
        if user_message:
            return f"{base_response}\n\nYou said: '{user_message}'\n\nThis is a simple hello world response from the OpenMed backend. In a real implementation, this would be processed by an AI model."
        
        return f"{base_response}\n\nThis is a simple hello world response from the OpenMed backend."
    
    @staticmethod
    def create_completion_id(prefix: str = "chatcmpl") -> str:
        """Create a unique completion ID"""
        return f"{prefix}-{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def calculate_tokens(text: str) -> int:
        """Simple token calculation (word count approximation)"""
        return len(text.split()) if text else 0


class StreamingUtils:
    """Utilities for streaming responses"""
    
    @staticmethod
    async def generate_chat_stream(content: str, model: str, completion_id: str) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        words = content.split()
        
        for i, word in enumerate(words):
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now(UTC).timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " " if i < len(words) - 1 else word},
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0.05)
        
        yield "data: [DONE]\n\n"
    
    @staticmethod
    async def generate_mixed_content_stream(
        workflow_result: Dict[str, Any], 
        user_message: str, 
        file_context: str, 
        model: str, 
        completion_id: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with both text and images"""
        from services import ResponseService
        
        # Start with initial analysis message
        initial_message = "🔬 Starting medical analysis..."
        chunk_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now(UTC).timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": initial_message},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(0.5)
        
        # Stream the main response content
        response_content = await ResponseService.generate_intelligent_response(
            user_message, workflow_result, file_context, model
        )
        
        # Split response into chunks and stream them
        lines = response_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(UTC).timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": line + "\n"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.1)
        
        # GradCAM images are already included inline in the response content
        # No need to add them separately to prevent duplication
        
        # Final chunk
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now(UTC).timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    @staticmethod
    async def generate_realtime_analysis_stream(
        user_message: str,
        all_image_ids: List[str],
        uploaded_files: Dict[str, Any],
        model: str,
        completion_id: str
    ) -> AsyncGenerator[str, None]:
        """Generate real-time streaming analysis with GradCAM as it's processed"""
        from services import WorkflowService, ResponseService, MedicalAnalysisService
        
        try:
            # Stream initial message
            yield StreamingUtils._create_text_chunk(
                completion_id, model, "🧠 Starting intelligent medical analysis...\n\n"
            )
            await asyncio.sleep(0.5)
            
            # Step 1: Intent Analysis
            yield StreamingUtils._create_text_chunk(
                completion_id, model, "🎯 **Step 1: Analyzing intent...**\n"
            )
            await asyncio.sleep(0.3)
            
            intent_analysis = WorkflowService._analyze_intent(user_message)
            intent_text = f"- Intent: {intent_analysis.get('intent_type', 'unknown')}\n"
            intent_text += f"- Confidence: {intent_analysis.get('confidence', 0):.1%}\n"
            intent_text += f"- Reasoning: {intent_analysis.get('reasoning', 'N/A')}\n\n"
            
            yield StreamingUtils._create_text_chunk(completion_id, model, intent_text)
            await asyncio.sleep(0.5)
            
            # Check if medical analysis is needed
            if (intent_analysis.get('wants_pneumonia_check') and 
                intent_analysis.get('confidence', 0) > 0.5 and 
                all_image_ids):
                
                yield StreamingUtils._create_text_chunk(
                    completion_id, model, "🔬 **Step 2: Medical image analysis...**\n"
                )
                await asyncio.sleep(0.3)
                
                # Check service availability
                service_available = await MedicalAnalysisService.check_resnet50_service()
                if not service_available:
                    yield StreamingUtils._create_text_chunk(
                        completion_id, model, "⚠️ Medical analysis service unavailable\n\n"
                    )
                else:
                    analysis_results = []
                    
                    for i, file_id in enumerate(all_image_ids, 1):
                        if file_id in uploaded_files:
                            file_info = uploaded_files[file_id]
                            if hasattr(file_info, 'file_path') and os.path.exists(file_info.file_path):
                                yield StreamingUtils._create_text_chunk(
                                    completion_id, model, f"📸 **Analyzing image {i}: {file_info.filename}**\n"
                                )
                                await asyncio.sleep(0.2)
                                
                                # Perform analysis with GradCAM
                                analysis = await MedicalAnalysisService.analyze_image_for_pneumonia(
                                    file_info.file_path, generate_gradcam=True
                                )
                                analysis['filename'] = file_info.filename
                                analysis['file_id'] = file_id
                                analysis_results.append(analysis)
                                
                                if analysis.get('success'):
                                    # Stream analysis result
                                    result_text = f"- Prediction: **{analysis.get('prediction_label', 'Unknown')}**\n"
                                    result_text += f"- Confidence: **{analysis.get('confidence', 0):.1%}**\n"
                                    
                                    probabilities = analysis.get('probabilities', {})
                                    if probabilities:
                                        result_text += "- Probabilities:\n"
                                        for class_name, prob in probabilities.items():
                                            result_text += f"  - {class_name}: {prob:.1%}\n"
                                    
                                    yield StreamingUtils._create_text_chunk(completion_id, model, result_text)
                                    await asyncio.sleep(0.3)
                                    
                                    # Stream GradCAM if available
                                    if analysis.get('gradcam_available') and analysis.get('gradcam_filename'):
                                        yield StreamingUtils._create_text_chunk(
                                            completion_id, model, f"\n🎨 **GradCAM Visualization for {file_info.filename}:**\n"
                                        )
                                        await asyncio.sleep(0.2)
                                        
                                        # Stream the image using URL endpoint
                                        gradcam_filename = analysis.get('gradcam_filename')
                                        if gradcam_filename:
                                            # Use URL endpoint for optimal performance
                                            image_url = f"/v1/gradcam/{gradcam_filename}"
                                            image_chunk = {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(datetime.now(UTC).timestamp()),
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "content": f"![GradCAM for {file_info.filename}]({image_url})\n"
                                                    },
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(image_chunk)}\n\n"
                                        await asyncio.sleep(0.5)
                                        
                                        yield StreamingUtils._create_text_chunk(
                                            completion_id, model, "*Heat map showing areas of focus for the AI decision*\n\n"
                                        )
                                        await asyncio.sleep(0.3)
                                    
                                    # Add medical recommendations
                                    if analysis.get('prediction_label') == "Pneumonia":
                                        yield StreamingUtils._create_text_chunk(
                                            completion_id, model, "⚠️ **Pneumonia detected** - Please consult with a healthcare professional immediately.\n\n"
                                        )
                                    else:
                                        yield StreamingUtils._create_text_chunk(
                                            completion_id, model, "✅ No signs of pneumonia detected in this image.\n\n"
                                        )
                                    await asyncio.sleep(0.3)
                                else:
                                    yield StreamingUtils._create_text_chunk(
                                        completion_id, model, f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}\n\n"
                                    )
                                    await asyncio.sleep(0.2)
                    
                    # Add summary if multiple images
                    if len(analysis_results) > 1:
                        successful_analyses = [a for a in analysis_results if a.get('success')]
                        pneumonia_detected = [a for a in successful_analyses if a.get('prediction') == 1]
                        
                        if successful_analyses:
                            yield StreamingUtils._create_text_chunk(
                                completion_id, model, "📊 **Summary:**\n"
                            )
                            summary_text = f"- Images analyzed: {len(successful_analyses)}\n"
                            summary_text += f"- Pneumonia detected in: {len(pneumonia_detected)} image(s)\n"
                            
                            if pneumonia_detected:
                                summary_text += "- 🚨 **Immediate medical attention recommended**\n"
                            
                            yield StreamingUtils._create_text_chunk(completion_id, model, summary_text + "\n")
                            await asyncio.sleep(0.3)
            else:
                yield StreamingUtils._create_text_chunk(
                    completion_id, model, "💬 No medical analysis requested or no images provided.\n\n"
                )
            
            # Add disclaimers
            yield StreamingUtils._create_text_chunk(
                completion_id, model, "⚠️ **Important Medical Disclaimer:**\n"
            )
            yield StreamingUtils._create_text_chunk(
                completion_id, model, "This AI analysis is for educational purposes only and should not replace professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.\n\n"
            )
            await asyncio.sleep(0.3)
            
            # Add model info
            yield StreamingUtils._create_text_chunk(
                completion_id, model, f"🤖 **System Info:** Analysis powered by {model} with ResNet50 medical imaging AI\n"
            )
            
        except Exception as e:
            logger.error(f"❌ Error in real-time analysis stream: {str(e)}")
            yield StreamingUtils._create_text_chunk(
                completion_id, model, f"❌ **Error:** {str(e)}\n"
            )
        
        # Final chunk
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now(UTC).timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    @staticmethod
    def _create_text_chunk(completion_id: str, model: str, content: str) -> str:
        """Helper to create a text chunk for streaming"""
        chunk_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now(UTC).timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        }
        return f"data: {json.dumps(chunk_data)}\n\n"
    
    @staticmethod
    async def generate_optimized_medical_stream(
        workflow_result: Dict[str, Any], 
        user_message: str, 
        file_context: str, 
        model: str, 
        completion_id: str,
        base_url: str = ""
    ) -> AsyncGenerator[str, None]:
        """
        Generate optimized streaming response - GradCAM images are included inline
        """
        from services import ResponseService
        
        # Stream the response content which already includes inline GradCAM images
        response_content = await ResponseService.generate_intelligent_response(
            user_message, workflow_result, file_context, model
        )
        
        # Split response into chunks and stream them
        lines = response_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(UTC).timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": line + "\n"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.05)  # Faster text streaming
        
        # GradCAM images are already included inline in the response content
        # No need to add them separately to prevent duplication
        
        # Final chunk
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now(UTC).timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    @staticmethod
    async def generate_completion_stream(content: str, model: str) -> AsyncGenerator[str, None]:
        """Generate streaming text completion response"""
        words = content.split()
        
        for i, word in enumerate(words):
            chunk_data = {
                "id": ResponseGenerator.create_completion_id("cmpl"),
                "object": "text_completion",
                "created": int(datetime.now(UTC).timestamp()),
                "model": model,
                "choices": [{
                    "text": word + " " if i < len(words) - 1 else word,
                    "index": 0,
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0.05)
        
        yield "data: [DONE]\n\n"


class ResponseBuilder:
    """Utility class for building API responses"""
    
    @staticmethod
    def build_chat_completion_response(
        completion_id: str,
        model: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> ChatCompletionResponse:
        """Build a chat completion response"""
        return ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=int(datetime.now(UTC).timestamp()),
            model=model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
    
    @staticmethod
    def build_completion_response(
        completion_id: str,
        model: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> CompletionResponse:
        """Build a text completion response"""
        return CompletionResponse(
            id=completion_id,
            object="text_completion",
            created=int(datetime.now(UTC).timestamp()),
            model=model,
            choices=[{
                "text": content,
                "index": 0,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        ) 


class FileUtils:
    """Utility functions for file processing and validation"""
    
    @staticmethod
    def is_image_file(mime_type: str) -> bool:
        """Check if the file is an image based on MIME type"""
        return mime_type.startswith('image/')
    
    @staticmethod
    def is_supported_image_format(mime_type: str) -> bool:
        """Check if the image format is supported for medical analysis"""
        supported_formats = [
            'image/jpeg', 'image/jpg', 'image/png', 
            'image/bmp', 'image/tiff', 'image/webp'
        ]
        return mime_type.lower() in supported_formats
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension from filename"""
        return os.path.splitext(filename.lower())[1]
    
    @staticmethod
    def validate_file_for_medical_analysis(file_info: ProcessedFileInfo) -> Dict[str, Any]:
        """Validate if a file can be used for medical analysis"""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        # Check if file exists
        if not os.path.exists(file_info.file_path):
            validation_result["errors"].append("File not found on disk")
            return validation_result
        
        # Check if it's an image
        if not FileUtils.is_image_file(file_info.mime_type):
            validation_result["errors"].append(f"File is not an image (MIME type: {file_info.mime_type})")
            return validation_result
        
        # Check if format is supported
        if not FileUtils.is_supported_image_format(file_info.mime_type):
            validation_result["warnings"].append(f"Image format may not be optimal for analysis: {file_info.mime_type}")
        
        # Check file size (warn if too small or too large)
        if file_info.size < 1024:  # Less than 1KB
            validation_result["warnings"].append("Image file is very small, may affect analysis quality")
        elif file_info.size > 50 * 1024 * 1024:  # Larger than 50MB
            validation_result["warnings"].append("Image file is very large, processing may be slow")
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        return validation_result


class APIResponseUtils:
    """Utilities for handling API responses and formatting"""
    
    @staticmethod
    def create_error_response(message: str, error_type: str = "api_error", code: int = 500) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "error": {
                "message": message,
                "type": error_type,
                "code": code
            }
        }
    
    @staticmethod
    def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Create standardized success response"""
        return {
            "success": True,
            "message": message,
            "data": data
        }
    
    @staticmethod
    def format_openwebui_file_response(file_info: ProcessedFileInfo) -> Dict[str, Any]:
        """Format file info for OpenWebUI compatibility"""
        return {
            "id": f"file-{file_info.id}",
            "object": "file",
            "bytes": file_info.size,
            "created_at": int(file_info.upload_time.timestamp()),
            "filename": file_info.filename,
            "purpose": "vision"
        }
    
    @staticmethod
    def create_health_response(status: str, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create health check response"""
        response = {
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime": "unknown"  # Could be calculated if needed
        }
        
        if additional_info:
            response.update(additional_info)
        
        return response