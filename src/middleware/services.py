#!/usr/bin/env python3
"""
Core Services for OpenMed OpenWebUI-Compatible Backend

This module contains the core business logic services abstracted from the main
backend implementation. Services are designed to be reusable, testable, and
maintainable with clear separation of concerns.

Services included:
- MedicalAnalysisService: Multi-disease medical image analysis using ResNet50
- WorkflowService: Agentic workflow processing for multiple diseases
- FileService: File management operations
- ResponseService: Intelligent response generation
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import uuid
import aiohttp
import aiofiles
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Union

from models import ProcessedFileInfo
from config import HELLO_WORLD_RESPONSES

logger = logging.getLogger(__name__)

# ResNet50 Full Network API configuration
RESNET50_FULL_URL = "http://localhost:6010"

# GradCAM configuration
GRADCAM_OUTPUT_DIR = "src/gradcam_images"

# Disease configuration
SUPPORTED_DISEASES = {
    'pneumonia': {
        'endpoint': '/predict/pneumonia',
        'image_types': ['chest_xray', 'chest_radiograph'],
        'classes': ['Normal', 'Pneumonia']
    },
    'brain_tumor': {
        'endpoint': '/predict/brain_tumor', 
        'image_types': ['brain_mri', 'cranial_scan'],
        'classes': ['Glioma', 'Meningioma', 'Tumor']
    },
    'tb': {
        'endpoint': '/predict/tb',
        'image_types': ['chest_xray', 'chest_radiograph'], 
        'classes': ['Normal', 'TB']
    }
}


class MedicalAnalysisService:
    """Service for multi-disease medical image analysis using ResNet50 model pipeline"""
    
    @staticmethod
    async def analyze_image_for_disease(
        image_path: str, 
        disease_type: str, 
        generate_gradcam: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze an image for a specific disease using the ResNet50 Full Network
        
        Args:
            image_path: Path to the image file to analyze
            disease_type: Type of disease ('pneumonia', 'brain_tumor', 'tb')
            generate_gradcam: Whether to generate GradCAM visualization
            
        Returns:
            Dict with analysis results including prediction, confidence, and GradCAM path
        """
        if disease_type not in SUPPORTED_DISEASES:
            return {
                "success": False,
                "error": f"Unsupported disease type: {disease_type}",
                "message": f"Disease type '{disease_type}' is not supported. Available: {list(SUPPORTED_DISEASES.keys())}"
            }
            
        try:
            # Read and encode the image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            async with aiohttp.ClientSession() as session:
                logger.info(f"🔬 Analyzing image for {disease_type} with ResNet50 Full Network...")
                
                # Call disease-specific API endpoint
                prediction_result = await MedicalAnalysisService._predict_with_disease_network(
                    session, image_base64, disease_type, generate_gradcam, image_path
                )
                
                return prediction_result
                
        except Exception as e:
            logger.error(f"❌ Error in medical analysis for {disease_type}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "disease_type": disease_type,
                "message": f"An error occurred during {disease_type} analysis"
            }
    
    @staticmethod
    async def analyze_image_for_pneumonia(image_path: str, generate_gradcam: bool = True) -> Dict[str, Any]:
        """
        Legacy method for pneumonia analysis - now routes to multi-disease method
        
        Args:
            image_path: Path to the image file to analyze
            generate_gradcam: Whether to generate GradCAM visualization
            
        Returns:
            Dict with analysis results including prediction, confidence, and GradCAM path
        """
        return await MedicalAnalysisService.analyze_image_for_disease(
            image_path, 'pneumonia', generate_gradcam
        )
    
    @staticmethod
    async def _predict_with_disease_network(
        session: aiohttp.ClientSession, 
        image_base64: str, 
        disease_type: str,
        generate_gradcam: bool, 
        image_path: str
    ) -> Dict[str, Any]:
        """Make prediction using the disease-specific ResNet50 Full Network API"""
        disease_config = SUPPORTED_DISEASES[disease_type]
        endpoint = disease_config['endpoint']
        
        prediction_payload = {
            "image_base64": image_base64,
            "generate_gradcam": generate_gradcam,
            "target_class": None
        }
        
        try:
            async with session.post(
                f"{RESNET50_FULL_URL}{endpoint}",
                json=prediction_payload,
                timeout=aiohttp.ClientTimeout(total=60)  # Longer timeout for GradCAM
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ {disease_type} network prediction failed {response.status}: {error_text}")
                    return {
                        "success": False,
                        "error": f"Prediction error: {response.status}",
                        "disease_type": disease_type,
                        "message": f"Failed to get {disease_type} prediction from ResNet50 Full Network"
                    }
                
                result = await response.json()
                
                if not result.get('success'):
                    logger.error(f"❌ {disease_type} prediction was not successful")
                    return {
                        "success": False,
                        "error": "Prediction failed",
                        "disease_type": disease_type,
                        "message": result.get('message', f'Unknown {disease_type} prediction error')
                    }
                
                # Process the result
                prediction = result.get('predicted_class')
                confidence = result.get('confidence')
                prediction_label = result.get('prediction_label')
                probabilities = result.get('probabilities', [])
                class_names = result.get('class_names', disease_config['classes'])
                gradcam_heatmap = result.get('gradcam_heatmap')
                
                logger.info(f"✅ {disease_type} prediction successful: {prediction_label} (confidence: {confidence:.3f})")
                
                # Build response with disease-specific probabilities
                probabilities_dict = {}
                for i, class_name in enumerate(class_names):
                    probabilities_dict[class_name] = float(probabilities[i]) if i < len(probabilities) else 0.0
                
                response_data = {
                    "success": True,
                    "disease_type": disease_type,
                    "prediction": int(prediction),
                    "confidence": float(confidence),
                    "prediction_label": prediction_label,
                    "class_names": class_names,
                    "probabilities": probabilities_dict,
                    "message": f"{disease_type.title()} analysis complete. Prediction: {prediction_label} (Confidence: {confidence:.3f})"
                }
                
                # Handle GradCAM if available
                if generate_gradcam and gradcam_heatmap:
                    logger.info("🎨 Processing GradCAM heatmap...")
                    gradcam_result = await MedicalAnalysisService._process_gradcam_heatmap(
                        gradcam_heatmap, image_path, disease_type
                    )
                    
                    if gradcam_result.get('success'):
                        response_data.update({
                            "gradcam_available": True,
                            "gradcam_path": gradcam_result.get("gradcam_path"),
                            "gradcam_filename": gradcam_result.get("gradcam_filename")
                        })
                        logger.info(f"📁 GradCAM saved to: {gradcam_result.get('gradcam_path')}")
                    else:
                        response_data.update({
                            "gradcam_available": False,
                            "gradcam_error": gradcam_result.get("message", "GradCAM processing failed")
                        })
                        logger.warning(f"⚠️ GradCAM processing failed: {gradcam_result.get('message')}")
                elif generate_gradcam:
                    response_data.update({
                        "gradcam_available": False,
                        "gradcam_error": "No GradCAM heatmap received from network"
                    })
                    logger.warning("⚠️ No GradCAM heatmap received despite being requested")
                
                return response_data
                
        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error in {disease_type} network prediction: {str(e)}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "disease_type": disease_type,
                "message": f"Unable to connect to {disease_type} ResNet50 Full Network service"
            }
    
    @staticmethod
    async def _process_gradcam_heatmap(gradcam_heatmap: List, original_image_path: str, disease_type: str = "unknown") -> Dict[str, Any]:
        """Process GradCAM heatmap received from the full network and create visualization"""
        try:
            # Ensure GradCAM output directory exists
            os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)
            
            logger.info(f"🎨 Creating GradCAM visualization from {disease_type} network heatmap...")
            gradcam_path = await MedicalAnalysisService._create_gradcam_visualization(
                gradcam_heatmap, original_image_path, disease_type
            )
            
            gradcam_filename = os.path.basename(gradcam_path)
            
            logger.info(f"✅ GradCAM visualization created successfully: {gradcam_path}")
            
            return {
                "success": True,
                "gradcam_path": os.path.abspath(gradcam_path),
                "gradcam_filename": gradcam_filename,
                "disease_type": disease_type,
                "message": f"GradCAM visualization created from {disease_type} network heatmap"
            }
                
        except Exception as e:
            logger.error(f"❌ Error processing GradCAM heatmap for {disease_type}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "disease_type": disease_type,
                "message": f"Failed to process GradCAM heatmap for {disease_type}"
            }
    
    @staticmethod
    async def _create_gradcam_visualization(gradcam_heatmap: List, original_image_path: str, disease_type: str = "unknown") -> str:
        """Create GradCAM visualization overlay from heatmap data"""
        try:
            import numpy as np
            import cv2
            from PIL import Image
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            # Convert heatmap to numpy array
            heatmap_array = np.array(gradcam_heatmap)
            
            # Load original image
            original_image = cv2.imread(original_image_path)
            if original_image is None:
                raise Exception(f"Could not load original image: {original_image_path}")
            
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Resize heatmap to match original image dimensions
            img_height, img_width = original_rgb.shape[:2]
            heatmap_resized = cv2.resize(heatmap_array, (img_width, img_height))
            
            # Normalize heatmap
            heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            
            # Apply colormap
            colormap = cm.get_cmap('jet')
            heatmap_colored = colormap(heatmap_normalized)
            heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
            
            # Create overlay
            alpha = 0.5
            overlay = cv2.addWeighted(original_rgb, 1-alpha, heatmap_colored, alpha, 0)
            
            # Generate output filename with disease type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_basename = os.path.splitext(os.path.basename(original_image_path))[0]
            gradcam_filename = f"gradcam_{disease_type}_{original_basename}_{timestamp}.png"
            gradcam_path = os.path.join(GRADCAM_OUTPUT_DIR, gradcam_filename)
            
            # Save the visualization
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_rgb)
            plt.title('Original Image')
            plt.axis('off')
            
            # Heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap_normalized, cmap='jet')
            plt.title(f'{disease_type.title()} GradCAM Heatmap')
            plt.axis('off')
            plt.colorbar()
            
            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title(f'{disease_type.title()} GradCAM Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return gradcam_path
            
        except Exception as e:
            logger.error(f"❌ Error creating GradCAM visualization for {disease_type}: {str(e)}")
            raise
    
    @staticmethod
    async def check_resnet50_service() -> bool:
        """Check if the ResNet50 Full Network service is available"""
        try:
            async with aiohttp.ClientSession() as session:
                logger.info("🔍 Checking ResNet50 Full Network service health...")
                return await MedicalAnalysisService._check_service_health(
                    session, f"{RESNET50_FULL_URL}/health", "ResNet50 Full Network"
                )
        except Exception as e:
            logger.error(f"❌ Error checking ResNet50 service: {str(e)}")
            return False
    
    @staticmethod
    async def _check_service_health(session: aiohttp.ClientSession, url: str, service_name: str) -> bool:
        """Check health of a specific service"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.info(f"✅ {service_name} service is healthy")
                    return True
                else:
                    logger.warning(f"⚠️ {service_name} service returned status {response.status}")
                    return False
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to {service_name}: {str(e)}")
            return False


class WorkflowService:
    """Service for processing agentic workflows for multiple diseases"""
    
    @staticmethod
    async def process_agentic_workflow(
        user_message: str, 
        uploaded_images: List[str], 
        uploaded_files: Dict[str, ProcessedFileInfo],
        wants_diagnosis_summary: bool = False,
        conversation_messages: List = None
    ) -> Dict[str, Any]:
        """
        Process the agentic workflow for multi-disease medical intent classification and analysis
        LATEST MESSAGE FOCUS: Only analyze images from the current user request
        
        Args:
            user_message: User's text message (from latest user turn)
            uploaded_images: List of file IDs for uploaded images (current request only)
            uploaded_files: Dictionary of uploaded file information
            
        Returns:
            Dict with workflow results including intent analysis and medical analysis if applicable
        """
        logger.info(f"🧠 Starting LATEST MESSAGE ONLY multi-disease workflow")
        logger.info(f"💬 User message: '{user_message[:100]}...'")
        logger.info(f"📸 Current request images: {len(uploaded_images)} - {uploaded_images}")
        logger.info(f"📋 Diagnosis summary request: {wants_diagnosis_summary}")
        
        # Handle diagnosis summary requests - TEXT-ONLY ANALYSIS
        if wants_diagnosis_summary:
            logger.info(f"📋 Processing TEXT-ONLY diagnosis summary request")
            logger.info(f"📝 No images needed - analyzing conversation history for comprehensive diagnosis")
            return {
                "intent_analysis": {
                    "wants_medical_analysis": False,
                    "wants_diagnosis_summary": True,
                    "disease_type": "comprehensive_summary",
                    "confidence": 0.95,
                    "reasoning": "User requested a full diagnosis or comprehensive summary - analyzing conversation history without requiring new images",
                    "analysis_type": "text_only"
                },
                "medical_analysis": None,
                "images_analyzed": [],
                "workflow_status": "text_only_diagnosis_summary",
                "conversation_messages": conversation_messages
            }
        
        # Clear any residual state - focus only on current request
        if not uploaded_images:
            logger.info(f"🚫 No images in current request - skipping medical analysis")
            return {
                "intent_analysis": {"wants_medical_analysis": False, "message": "No images provided in current request"},
                "medical_analysis": None,
                "images_analyzed": [],
                "workflow_status": "no_images_current_request"
            }
        
        # Step 1: Analyze intent using the updated multi-disease agent
        intent_analysis = WorkflowService._analyze_intent(user_message)
        wants_analysis = intent_analysis.get('wants_medical_analysis', False)
        disease_type = intent_analysis.get('disease_type')
        confidence = intent_analysis.get('confidence', 0.0)
        
        logger.info(f"🎯 Intent analysis: wants_medical_analysis={wants_analysis}, disease_type={disease_type}, confidence={confidence}")
        
        workflow_result = {
            "intent_analysis": intent_analysis,
            "medical_analysis": None,
            "images_analyzed": [],
            "workflow_status": "completed"
        }
        
        # Step 2: ONLY perform analysis if medical analysis is requested for CURRENT images
        if (wants_analysis and confidence > 0.5 and uploaded_images and disease_type):
            logger.info(f"🔬 Performing {disease_type} analysis on {len(uploaded_images)} current request images")
            
            workflow_result["medical_analysis"] = await WorkflowService._perform_medical_analysis(
                uploaded_images, uploaded_files, disease_type
            )
            
            # Extract successfully analyzed images
            if isinstance(workflow_result["medical_analysis"], list):
                workflow_result["images_analyzed"] = [
                    r['filename'] for r in workflow_result["medical_analysis"] if r.get('success')
                ]
        elif wants_analysis and confidence > 0.5 and uploaded_images and not disease_type:
            # If medical analysis is wanted but disease type is unclear, try pneumonia as default
            logger.info("🔄 Disease type unclear, defaulting to pneumonia analysis for current request images")
            workflow_result["medical_analysis"] = await WorkflowService._perform_medical_analysis(
                uploaded_images, uploaded_files, 'pneumonia'
            )
            
            if isinstance(workflow_result["medical_analysis"], list):
                workflow_result["images_analyzed"] = [
                    r['filename'] for r in workflow_result["medical_analysis"] if r.get('success')
                ]
        else:
            logger.info(f"⏭️ Skipping medical analysis: wants_analysis={wants_analysis}, confidence={confidence}, images={len(uploaded_images)}, disease_type={disease_type}")
        
        logger.info(f"✅ Workflow completed for CURRENT REQUEST - analyzed {len(workflow_result.get('images_analyzed', []))} images")
        return workflow_result
    
    @staticmethod
    def _analyze_intent(user_message: str) -> Dict[str, Any]:
        """Analyze user intent for multi-disease detection"""
        # Import updated agent for multi-disease medical intent analysis
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))
        
        try:
            from agent import analyze_medical_intent
            result = analyze_medical_intent(user_message)
            
            # Convert to format expected by middleware for backwards compatibility
            if 'wants_medical_analysis' in result:
                # Add legacy pneumonia check field for backwards compatibility
                result['wants_pneumonia_check'] = (
                    result.get('wants_medical_analysis', False) and 
                    result.get('disease_type') == 'pneumonia'
                )
            
            return result
        except ImportError:
            logger.warning("⚠️ Updated agent module not available, using fallback intent analysis")
            return WorkflowService._fallback_intent_analysis(user_message)
    
    @staticmethod
    def _fallback_intent_analysis(user_message: str) -> Dict[str, Any]:
        """Fallback intent analysis when agent is not available"""
        message_lower = user_message.lower()
        
        # Check for disease-specific keywords
        pneumonia_keywords = ['pneumonia', 'lung infection', 'chest x-ray', 'chest xray', 'pulmonary']
        brain_tumor_keywords = ['brain tumor', 'brain cancer', 'glioma', 'meningioma', 'brain mri', 'cranial']
        tb_keywords = ['tuberculosis', 'tb', 'pulmonary tb', 'mycobacterium']
        
        disease_type = None
        intent_type = "other"
        wants_medical_analysis = False
        
        if any(keyword in message_lower for keyword in pneumonia_keywords):
            disease_type = 'pneumonia'
            intent_type = 'pneumonia_check'
            wants_medical_analysis = True
        elif any(keyword in message_lower for keyword in brain_tumor_keywords):
            disease_type = 'brain_tumor'
            intent_type = 'brain_tumor_check'
            wants_medical_analysis = True
        elif any(keyword in message_lower for keyword in tb_keywords):
            disease_type = 'tb'
            intent_type = 'tb_check'
            wants_medical_analysis = True
        elif any(keyword in message_lower for keyword in ['medical', 'diagnose', 'analyze', 'scan', 'image']):
            intent_type = 'medical_imaging'
            wants_medical_analysis = True
        
        confidence = 0.8 if wants_medical_analysis else 0.2
        
        return {
            "wants_medical_analysis": wants_medical_analysis,
            "wants_pneumonia_check": disease_type == 'pneumonia',  # Legacy compatibility
            "disease_type": disease_type,
            "confidence": confidence,
            "intent_type": intent_type,
            "reasoning": f"Fallback analysis detected: {disease_type or 'no medical intent'}"
        }
    
    @staticmethod
    async def _perform_medical_analysis(
        uploaded_images: List[str], 
        uploaded_files: Dict[str, ProcessedFileInfo],
        disease_type: str
    ) -> Union[List[Dict], Dict[str, Any]]:
        """Perform medical analysis on uploaded images for a specific disease"""
        logger.info(f"🔬 Performing {disease_type} analysis on {len(uploaded_images)} image(s)")
        
        # Check if ResNet50 service is available
        service_available = await MedicalAnalysisService.check_resnet50_service()
        if not service_available:
            logger.warning("⚠️ ResNet50 service not available, skipping medical analysis")
            return {
                "success": False,
                "error": "Medical analysis service unavailable",
                "disease_type": disease_type,
                "message": f"The {disease_type} detection service is currently unavailable. Please try again later."
            }
        
        # Analyze images
        analysis_results = []
        for file_id in uploaded_images:
            if file_id in uploaded_files:
                file_info = uploaded_files[file_id]
                if hasattr(file_info, 'file_path') and os.path.exists(file_info.file_path):
                    logger.info(f"📸 Analyzing image for {disease_type}: {file_info.filename}")
                    analysis = await MedicalAnalysisService.analyze_image_for_disease(
                        file_info.file_path, disease_type
                    )
                    analysis['filename'] = file_info.filename
                    analysis['file_id'] = file_id
                    analysis_results.append(analysis)
                    
                    if analysis['success']:
                        logger.info(f"✅ {disease_type} analysis complete for {file_info.filename}: {analysis['prediction_label']} ({analysis['confidence']:.3f})")
                    else:
                        logger.error(f"❌ {disease_type} analysis failed for {file_info.filename}: {analysis.get('error', 'Unknown error')}")
                else:
                    logger.warning(f"⚠️ File not found: {file_id}")
        
        return analysis_results


class FileService:
    """Service for file management operations"""
    
    @staticmethod
    def get_safe_filename(filename: str) -> str:
        """Generate a safe filename with UUID prefix"""
        name, ext = os.path.splitext(filename)
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '.', '_')).rstrip()
        return f"{uuid.uuid4().hex[:8]}_{safe_name}{ext}"
    
    @staticmethod
    def get_mime_type(filename: str) -> str:
        """Get MIME type from filename"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"
    
    @staticmethod
    async def save_uploaded_file(file, upload_dir: str = "open-webui/uploads") -> ProcessedFileInfo:
        """Save uploaded file and return processed info"""
        try:
            # Log file reception
            original_filename = file.filename or "unnamed_file"
            logger.info(f"📁 FILE RECEIVED: '{original_filename}' (Size: {file.size} bytes, Content-Type: {file.content_type})")
            
            # Generate safe filename and full path
            safe_filename = FileService.get_safe_filename(original_filename)
            file_path = os.path.join(upload_dir, safe_filename)
            absolute_path = os.path.abspath(file_path)
            
            # Ensure upload directory exists
            os.makedirs(upload_dir, exist_ok=True)
            logger.info(f"📂 Upload directory: {os.path.abspath(upload_dir)}")
            
            # Read file content
            content = await file.read()
            actual_size = len(content)
            logger.info(f"📊 File content read: {actual_size} bytes")
            
            # Save file to disk
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Verify file was saved
            if os.path.exists(file_path):
                saved_size = os.path.getsize(file_path)
                logger.info(f"✅ FILE SAVED SUCCESSFULLY:")
                logger.info(f"   📁 Original filename: {original_filename}")
                logger.info(f"   🔒 Safe filename: {safe_filename}")
                logger.info(f"   📍 Saved location: {absolute_path}")
                logger.info(f"   📏 File size: {saved_size} bytes")
            else:
                logger.error(f"❌ FILE NOT FOUND after saving: {absolute_path}")
                raise Exception(f"File was not saved to {absolute_path}")
            
            # Reset file pointer for potential future reading
            await file.seek(0)
            
            # Create file info
            file_info = ProcessedFileInfo(
                id=uuid.uuid4().hex,
                filename=original_filename,
                safe_filename=safe_filename,
                file_path=absolute_path,
                size=actual_size,
                mime_type=FileService.get_mime_type(original_filename),
                upload_time=datetime.now(UTC),
                processed=True,
            )
            
            return file_info
            
        except Exception as e:
            logger.error(f"❌ ERROR SAVING FILE '{file.filename}': {str(e)}")
            raise Exception(f"Error saving file: {str(e)}")
    
    @staticmethod
    def get_file_list_info(uploaded_files: Dict[str, ProcessedFileInfo], upload_dir: str) -> Dict[str, Any]:
        """Get comprehensive file list information"""
        files_list = []
        for file_id, file_info in uploaded_files.items():
            file_exists = os.path.exists(file_info.file_path) if hasattr(file_info, 'file_path') else False
            files_list.append({
                "id": file_info.id,
                "filename": file_info.filename,
                "safe_filename": file_info.safe_filename,
                "size": file_info.size,
                "mime_type": file_info.mime_type,
                "upload_time": file_info.upload_time,
                "processed": file_info.processed,
                "file_exists": file_exists,
                "file_path": getattr(file_info, 'file_path', None)
            })
        
        # Also list physical files in upload directory
        physical_files = []
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    physical_files.append({
                        "filename": filename,
                        "size": os.path.getsize(file_path),
                        "path": os.path.abspath(file_path)
                    })
        
        return {
            "uploaded_files": files_list,
            "physical_files": physical_files,
            "upload_directory": os.path.abspath(upload_dir)
        }


class ResponseService:
    """Service for generating intelligent responses"""
    
    @staticmethod
    async def generate_intelligent_response(
        user_message: str, 
        workflow_result: Dict[str, Any], 
        file_context: str, 
        model: str
    ) -> str:
        """
        Generate an intelligent response based on the agentic workflow results
        
        Args:
            user_message: Original user message
            workflow_result: Results from the agentic workflow
            file_context: File context information
            model: Model name requested
            
        Returns:
            Formatted response string
        """
        intent_analysis = workflow_result.get('intent_analysis', {})
        medical_analysis = workflow_result.get('medical_analysis')
        images_analyzed = workflow_result.get('images_analyzed', [])
        wants_medical_analysis = intent_analysis.get('wants_medical_analysis', False)
        
        # Base response
        response_parts = []
        
        # Check if this is a diagnosis summary request
        wants_diagnosis_summary = intent_analysis.get('wants_diagnosis_summary', False)
        conversation_messages = workflow_result.get('conversation_messages', [])
        
        # Check if this is a general advice/help request after medical analysis
        user_message_lower = user_message.lower()
        asking_for_advice = any(phrase in user_message_lower for phrase in [
            "what should i do", "advice", "next steps", "what now", "help me", 
            "recommend", "suggestion", "guidance", "what to do"
        ])
        
        if wants_diagnosis_summary:
            # Generate comprehensive diagnosis summary from conversation history - TEXT ONLY
            response_parts.append("📋 **Comprehensive Medical Analysis Summary** (Text-Only Analysis)")
            response_parts.append("*Based on conversation history - no new images required*")
            
            # Analyze conversation history for medical findings
            pneumonia_found = False
            brain_tumor_found = False
            tb_found = False
            findings = []
            images_analyzed_count = 0
            
            logger.info(f"📋 Analyzing conversation history with {len(conversation_messages) if conversation_messages else 0} messages")
            
            if conversation_messages:
                for msg in conversation_messages:
                    # Handle both dict and Pydantic Message objects
                    if hasattr(msg, 'role'):
                        # Pydantic Message object
                        msg_role = msg.role
                        msg_content = str(msg.content) if hasattr(msg, 'content') else ''
                    else:
                        # Dictionary format
                        msg_role = msg.get('role', '')
                        msg_content = str(msg.get('content', ''))
                    
                    if msg_role == 'assistant':
                        content = msg_content
                        content_lower = content.lower()
                        
                        # Count images analyzed
                        if 'medical analysis results' in content_lower:
                            images_analyzed_count += content_lower.count('prediction:')
                        
                        # Look for specific disease findings with confidence levels
                        if 'pneumonia detected' in content_lower or 'prediction: pneumonia' in content_lower:
                            pneumonia_found = True
                            # Extract confidence if available
                            if 'confidence:' in content:
                                confidence_match = content.split('confidence:')[1].split('%')[0].strip()
                                findings.append(f"**Pneumonia**: Detected with {confidence_match}% confidence in chest X-ray analysis")
                            else:
                                findings.append("**Pneumonia**: Detected with high confidence in chest X-ray analysis")
                                
                        if 'tumor detected' in content_lower or 'prediction: tumor' in content_lower or 'prediction: glioma' in content_lower:
                            brain_tumor_found = True
                            if 'confidence:' in content:
                                confidence_match = content.split('confidence:')[1].split('%')[0].strip()
                                findings.append(f"**Brain Tumor**: Detected with {confidence_match}% confidence in brain imaging analysis")
                            else:
                                findings.append("**Brain Tumor**: Detected in brain imaging analysis")
                                
                        if 'tb detected' in content_lower or 'prediction: tb' in content_lower:
                            tb_found = True
                            if 'confidence:' in content:
                                confidence_match = content.split('confidence:')[1].split('%')[0].strip()
                                findings.append(f"**Tuberculosis (TB)**: Detected with {confidence_match}% confidence in chest imaging analysis")
                            else:
                                findings.append("**Tuberculosis (TB)**: Detected in chest imaging analysis")
            
            if findings:
                response_parts.append(f"\n📊 **Analysis Overview:**")
                response_parts.append(f"• **Total images analyzed**: {images_analyzed_count}")
                response_parts.append(f"• **Analysis method**: AI-powered medical imaging with GradCAM visualization")
                response_parts.append(f"• **Diseases screened**: Pneumonia, Brain Tumor, Tuberculosis")
                
                response_parts.append(f"\n🏥 **Key Findings from Our Analysis Session:**")
                for finding in findings:
                    response_parts.append(f"• {finding}")
                
                response_parts.append(f"\n⚠️ **Medical Summary:**")
                if pneumonia_found:
                    response_parts.append("• **Pneumonia detected** - Respiratory infection requiring medical attention")
                if brain_tumor_found:
                    response_parts.append("• **Brain tumor detected** - Neurological condition requiring immediate specialist consultation")
                if tb_found:
                    response_parts.append("• **Tuberculosis detected** - Infectious disease requiring immediate treatment")
                
                response_parts.append(f"\n🚨 **Critical Action Required:**")
                response_parts.append("1. **Contact your healthcare provider IMMEDIATELY**")
                response_parts.append("2. **Bring all analysis results** to your medical appointment")
                response_parts.append("3. **Seek emergency care** if experiencing severe symptoms")
                
                response_parts.append(f"\n📞 **Emergency Contact:**")
                response_parts.append("If experiencing difficulty breathing, severe chest pain, seizures, or severe headaches, **call emergency services immediately**.")
                
                response_parts.append(f"\n🤖 **Analysis Note:**")
                response_parts.append("This diagnosis summary was generated through text-only analysis of our conversation history. No new images were processed for this summary.")
            else:
                response_parts.append(f"\n📝 **No specific medical findings** were detected in our conversation history.")
                response_parts.append("If you've uploaded medical images, please specify what type of analysis you'd like (pneumonia, brain tumor, or TB detection).")
                response_parts.append(f"\n💡 **How to get a diagnosis:** Upload medical images and ask me to 'analyze for pneumonia', 'check for brain tumor', or 'detect TB'.")
            
        elif asking_for_advice and not wants_medical_analysis:
            # Generate helpful general advice
            response_parts.append("Based on your medical analysis results, here are my recommendations:")
            
            response_parts.append("\n🏥 **Immediate Actions:**")
            response_parts.append("1. **Contact your healthcare provider** - Share these AI analysis results with your doctor")
            response_parts.append("2. **Keep original images** - Your doctor may want to review the original scans")
            response_parts.append("3. **Note symptoms** - Document any symptoms you're experiencing")
            
            response_parts.append("\n📋 **When visiting your doctor:**")
            response_parts.append("• Bring printed copies of the analysis results")
            response_parts.append("• Mention any symptoms (fever, cough, headaches, etc.)")
            response_parts.append("• Provide your complete medical history")
            response_parts.append("• Ask about follow-up tests if recommended")
            
            response_parts.append("\n⚕️ **Medical Follow-up:**")
            response_parts.append("• **Pneumonia detected**: Seek medical attention within 24 hours")
            response_parts.append("• **Brain tumor detected**: Contact a neurologist or your primary care physician")
            response_parts.append("• **TB detected**: Contact your doctor immediately for treatment options")
            
            response_parts.append("\n🤖 **About AI Analysis:**")
            response_parts.append("• This AI analysis is a screening tool, not a definitive diagnosis")
            response_parts.append("• Professional medical evaluation is always required")
            response_parts.append("• AI can help identify potential areas of concern for further investigation")
            
            response_parts.append("\n📞 **Emergency Situations:**")
            response_parts.append("If you experience severe symptoms like difficulty breathing, chest pain, severe headaches, or seizures, **seek emergency medical care immediately**.")
            
        else:
            # Standard medical analysis workflow
            # Add greeting and intent acknowledgment
            if intent_analysis.get('wants_medical_analysis'):
                disease_type = intent_analysis.get('disease_type', 'medical condition')
                response_parts.append(f"I understand you'd like me to analyze medical images for {disease_type} detection.")
            else:
                response_parts.append("Hello! I'm your medical AI assistant.")
            
            # Add file context information
            if file_context:
                response_parts.append(f"\nFiles received:{file_context}")
            
            # Add medical analysis results if available
            if medical_analysis:
                ResponseService._add_medical_analysis_to_response(response_parts, medical_analysis)
            
            # Add intent analysis information (for debugging/transparency)
            ResponseService._add_intent_analysis_to_response(response_parts, intent_analysis)
            
            # Add medical disclaimers and recommendations
            ResponseService._add_medical_disclaimers(response_parts, intent_analysis, medical_analysis, images_analyzed)
        
        # Add model information
        response_parts.append(f"\n🤖 **System Info:** Analysis powered by {model} with ResNet50 medical imaging AI")
        
        return "\n".join(response_parts)
    
    @staticmethod
    def _extract_gradcam_images(workflow_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract GradCAM image information from workflow results"""
        gradcam_images = []
        medical_analysis = workflow_result.get('medical_analysis')
        
        if isinstance(medical_analysis, list):
            # Multiple analysis results
            for analysis in medical_analysis:
                if analysis.get('gradcam_available') and analysis.get('gradcam_filename'):
                    gradcam_images.append({
                        'filename': analysis.get('filename', 'unknown'),
                        'gradcam_path': analysis.get('gradcam_path'),
                        'gradcam_filename': analysis.get('gradcam_filename')
                    })
        elif isinstance(medical_analysis, dict):
            # Single analysis result
            if medical_analysis.get('gradcam_available') and medical_analysis.get('gradcam_filename'):
                gradcam_images.append({
                    'filename': 'analysis_result',
                    'gradcam_path': medical_analysis.get('gradcam_path'),
                    'gradcam_filename': medical_analysis.get('gradcam_filename')
                })
        
        return gradcam_images
    
    @staticmethod
    def _add_medical_analysis_to_response(response_parts: List[str], medical_analysis: Union[List, Dict]):
        """Add medical analysis results to response"""
        if isinstance(medical_analysis, list):
            ResponseService._add_multiple_analysis_results(response_parts, medical_analysis)
        elif isinstance(medical_analysis, dict):
            ResponseService._add_single_analysis_result(response_parts, medical_analysis)
    
    @staticmethod
    def _add_multiple_analysis_results(response_parts: List[str], medical_analysis: List[Dict]):
        """Add multiple image analysis results to response"""
        response_parts.append(f"\n🔬 **Medical Analysis Results:**")
        
        for i, analysis in enumerate(medical_analysis, 1):
            if analysis.get('success'):
                filename = analysis.get('filename', f'Image {i}')
                prediction = analysis.get('prediction_label', 'Unknown')
                confidence = analysis.get('confidence', 0)
                probabilities = analysis.get('probabilities', {})
                
                response_parts.append(f"\n**{filename}:**")
                response_parts.append(f"- Prediction: {prediction}")
                response_parts.append(f"- Confidence: {confidence:.1%}")
                
                # Add detailed probability breakdown if available
                if probabilities:
                    response_parts.append(f"- Probabilities:")
                    for class_name, prob in probabilities.items():
                        response_parts.append(f"  - {class_name}: {prob:.1%}")
                
                # Add GradCAM visualization if available
                if analysis.get('gradcam_available') and analysis.get('gradcam_filename'):
                    gradcam_filename = analysis.get('gradcam_filename')
                    gradcam_url = f"http://localhost:8000/v1/gradcam/{gradcam_filename}"
                    response_parts.append(f"- 🎨 **GradCAM Visualization:**")
                    response_parts.append(f"\n![GradCAM for {filename}]({gradcam_url})")
                    response_parts.append("*Heat map showing areas of interest for the AI decision*")
                elif analysis.get('gradcam_error'):
                    response_parts.append(f"- ⚠️ **GradCAM Note:** {analysis.get('gradcam_error')}")
                
                if prediction == "Pneumonia":
                    response_parts.append("- ⚠️ **Pneumonia detected** - Please consult with a healthcare professional for proper diagnosis and treatment.")
                else:
                    response_parts.append("- ✅ No signs of pneumonia detected in this image.")
            else:
                filename = analysis.get('filename', f'Image {i}')
                error = analysis.get('error', 'Unknown error')
                response_parts.append(f"\n**{filename}:** Analysis failed - {error}")
        
        # Add summary if multiple images were analyzed
        if len(medical_analysis) > 1:
            ResponseService._add_analysis_summary(response_parts, medical_analysis)
    
    @staticmethod
    def _add_single_analysis_result(response_parts: List[str], medical_analysis: Dict):
        """Add single analysis result to response"""
        if medical_analysis.get('success', False):
            prediction = medical_analysis.get('prediction_label', 'Unknown')
            confidence = medical_analysis.get('confidence', 0)
            probabilities = medical_analysis.get('probabilities', {})
            
            response_parts.append(f"\n🔬 **Medical Analysis Result:**")
            response_parts.append(f"- Prediction: {prediction}")
            response_parts.append(f"- Confidence: {confidence:.1%}")
            
            # Add detailed probability breakdown if available
            if probabilities:
                response_parts.append(f"- Probabilities:")
                for class_name, prob in probabilities.items():
                    response_parts.append(f"  - {class_name}: {prob:.1%}")
            
            # Add GradCAM visualization if available
            if medical_analysis.get('gradcam_available') and medical_analysis.get('gradcam_filename'):
                gradcam_filename = medical_analysis.get('gradcam_filename')
                gradcam_url = f"http://localhost:8000/v1/gradcam/{gradcam_filename}"
                response_parts.append(f"- 🎨 **GradCAM Visualization:**")
                response_parts.append(f"\n![GradCAM Analysis]({gradcam_url})")
                response_parts.append("*Heat map showing areas of interest for the AI decision*")
            elif medical_analysis.get('gradcam_error'):
                response_parts.append(f"- ⚠️ **GradCAM Note:** {medical_analysis.get('gradcam_error')}")
            
            if prediction == "Pneumonia":
                response_parts.append("- ⚠️ **Pneumonia detected** - Please consult with a healthcare professional immediately.")
            else:
                response_parts.append("- ✅ No signs of pneumonia detected.")
        else:
            error_message = medical_analysis.get('message', 'Analysis failed')
            response_parts.append(f"\n❌ **Medical Analysis Error:** {error_message}")
    
    @staticmethod
    def _add_analysis_summary(response_parts: List[str], medical_analysis: List[Dict]):
        """Add analysis summary for multiple images"""
        successful_analyses = [a for a in medical_analysis if a.get('success')]
        pneumonia_detected = [a for a in successful_analyses if a.get('prediction') == 1]
        
        if successful_analyses:
            response_parts.append(f"\n📊 **Summary:**")
            response_parts.append(f"- Images analyzed: {len(successful_analyses)}")
            response_parts.append(f"- Pneumonia detected in: {len(pneumonia_detected)} image(s)")
            
            if pneumonia_detected:
                response_parts.append("- 🚨 **Immediate medical attention recommended**")
    
    @staticmethod
    def _add_intent_analysis_to_response(response_parts: List[str], intent_analysis: Dict):
        """Add intent analysis information to response"""
        if intent_analysis:
            intent_type = intent_analysis.get('intent_type', 'unknown')
            confidence = intent_analysis.get('confidence', 0)
            reasoning = intent_analysis.get('reasoning', 'No reasoning provided')
            
            response_parts.append(f"\n🧠 **AI Analysis:**")
            response_parts.append(f"- Intent detected: {intent_type}")
            response_parts.append(f"- Confidence: {confidence:.1%}")
            response_parts.append(f"- Reasoning: {reasoning}")
    
    @staticmethod
    def _add_medical_disclaimers(response_parts: List[str], intent_analysis: Dict, medical_analysis: Any, images_analyzed: List[str]):
        """Add medical disclaimers and recommendations"""
        if intent_analysis.get('wants_pneumonia_check') or medical_analysis:
            response_parts.append("\n⚠️ **Important Medical Disclaimer:**")
            response_parts.append("This AI analysis is for educational purposes only and should not replace professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.")
        
        if images_analyzed:
            response_parts.append(f"\n📈 **Next Steps:**")
            response_parts.append("1. Share these results with your healthcare provider")
            response_parts.append("2. Provide complete medical history and symptoms")
            response_parts.append("3. Follow professional medical advice for treatment")
    
    @staticmethod
    def generate_hello_world_response(user_message: str = "") -> str:
        """Generate a hello world response"""
        import random
        base_response = random.choice(HELLO_WORLD_RESPONSES)
        if user_message:
            return f"{base_response}\n\nYou said: {user_message[:100]}{'...' if len(user_message) > 100 else ''}"
        return base_response
