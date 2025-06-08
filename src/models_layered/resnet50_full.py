#!/usr/bin/env python3
"""
ResNet50 Full Network with GradCAM - Multi-Disease Support
This module provides ResNet50 architecture that loads weights from checkpoints
and provides both classification and GradCAM visualization for multiple diseases:
- Pneumonia (2 classes: Normal, Pneumonia)
- Brain Tumor (3 classes: Glioma, Meningioma, Tumor) 
- Tuberculosis/TB (2 classes: Normal, TB)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
import io
from PIL import Image
import torchvision.transforms as transforms


def get_device():
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        return device
    else:
        return torch.device("cpu")


class ResNet50Full(nn.Module):
    """
    Complete ResNet50 network for medical image classification with GradCAM support.
    Supports multiple diseases: Pneumonia, Brain Tumor, TB.
    """
    
    def __init__(self, num_classes=2, pretrained_imagenet=True):
        """
        Initialize complete ResNet50 network.
        
        Args:
            num_classes (int): Number of output classes
            pretrained_imagenet (bool): Whether to load ImageNet pretrained weights initially
        """
        super(ResNet50Full, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained_imagenet:
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet50 = models.resnet50(weights=None)
        
        # Store original final layer info
        self.num_features = self.resnet50.fc.in_features  # 2048 for ResNet50
        
        # Replace the final fully connected layer
        self.resnet50.fc = nn.Linear(self.num_features, num_classes)
        
        # GradCAM support
        self.gradcam_hooks = []
        self.activations = None
        self.gradients = None
        
        print(f"ResNet50 Full Network initialized")
        print(f"Feature dimension: {self.num_features}")
        print(f"Output classes: {num_classes}")
    
    def load_checkpoint(self, checkpoint_path, device=None):
        """
        Load weights from checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            device: Device to load the checkpoint on
            
        Returns:
            dict: Information about the loading process
        """
        if device is None:
            device = get_device()
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load the state dict
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        # Prepare loading information
        loading_info = {
            'checkpoint_path': checkpoint_path,
            'total_keys_in_checkpoint': len(state_dict),
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'device': device.type
        }
        
        print(f"Successfully loaded checkpoint")
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        return loading_info
    
    def enable_gradcam(self):
        """Enable GradCAM by registering hooks on layer4."""
        # Remove existing hooks
        self.disable_gradcam()
        
        # Get layer4 (last convolutional layer)
        target_layer = self.resnet50.layer4
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        self.gradcam_hooks = [forward_handle, backward_handle]
        print("GradCAM hooks enabled on layer4")
    
    def disable_gradcam(self):
        """Disable GradCAM by removing hooks."""
        for hook in self.gradcam_hooks:
            hook.remove()
        self.gradcam_hooks = []
        self.activations = None
        self.gradients = None
    
    def forward(self, x):
        """Forward pass through the complete network."""
        return self.resnet50(x)
    
    def predict_with_gradcam(self, x, target_class=None):
        """
        Perform prediction and generate GradCAM heatmap.
        
        Args:
            x (torch.Tensor): Input tensor
            target_class (int, optional): Target class for GradCAM. If None, uses predicted class.
            
        Returns:
            dict: Contains predictions, probabilities, and GradCAM heatmap
        """
        # Enable gradients and GradCAM
        x.requires_grad_(True)
        self.enable_gradcam()
        
        try:
            # Forward pass
            self.eval()
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            
            # Determine target class for GradCAM
            if target_class is None:
                target_class = predicted_class
            
            # Backward pass for GradCAM
            self.zero_grad()
            target_score = logits[0, target_class]
            target_score.backward(retain_graph=True)
            
            # Check if gradients and activations were captured
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Failed to capture gradients or activations for GradCAM")
            
            # Compute GradCAM
            gradcam_heatmap = self._compute_gradcam()
            
            return {
                'success': True,
                'logits': logits[0].detach().cpu().numpy().tolist(),
                'probabilities': probabilities[0].detach().cpu().numpy().tolist(),
                'predicted_class': predicted_class,
                'target_class': target_class,
                'gradcam_heatmap': gradcam_heatmap.tolist(),
                'confidence': probabilities[0][predicted_class].item()
            }
            
        except Exception as e:
            print(f"Error in predict_with_gradcam: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'logits': None,
                'probabilities': None,
                'predicted_class': None,
                'target_class': target_class,
                'gradcam_heatmap': None,
                'confidence': 0.0
            }
        finally:
            self.disable_gradcam()
    
    def _compute_gradcam(self):
        """Compute GradCAM heatmap from stored gradients and activations."""
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not available for GradCAM computation")
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W] 
        activations = self.activations[0]  # [C, H, W]
        
        # Compute weights using global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.detach().cpu().numpy()


class MultiDiseaseModelManager:
    """Manager class for multiple disease models."""
    
    def __init__(self, device=None):
        self.device = device if device else get_device()
        self.models = {}
        self.model_configs = {
            'pneumonia': {
                'num_classes': 2,
                'checkpoint_path': 'checkpoints/resnet50_pneumonia_full/best_resnet50_pneumonia_full_trained.pth',
                'class_names': ['Normal', 'Pneumonia']
            },
            'brain_tumor': {
                'num_classes': 3,  
                'checkpoint_path': 'checkpoints/resnet50_brain_tumor_full/best_resnet50_brain_tumor_full_trained.pth',
                'class_names': ['Glioma', 'Meningioma', 'Tumor']
            },
            'tb': {
                'num_classes': 2,
                'checkpoint_path': 'checkpoints/resnet50_tb_full/best_resnet50_tb_full_trained.pth', 
                'class_names': ['Normal', 'TB']
            }
        }
        
    def load_model(self, disease_type: str) -> bool:
        """Load a specific disease model."""
        if disease_type not in self.model_configs:
            raise ValueError(f"Unknown disease type: {disease_type}")
            
        config = self.model_configs[disease_type]
        
        try:
            # Create model
            model = ResNet50Full(num_classes=config['num_classes'])
            
            # Load checkpoint
            model.load_checkpoint(config['checkpoint_path'], self.device)
            
            # Move to device and set to eval mode
            model.to(self.device)
            model.eval()
            
            self.models[disease_type] = model
            print(f"Successfully loaded {disease_type} model")
            return True
            
        except Exception as e:
            print(f"Failed to load {disease_type} model: {str(e)}")
            return False
    
    def load_all_models(self):
        """Load all disease models."""
        success_count = 0
        for disease_type in self.model_configs.keys():
            if self.load_model(disease_type):
                success_count += 1
        
        print(f"Successfully loaded {success_count}/{len(self.model_configs)} models")
        return success_count == len(self.model_configs)
    
    def predict(self, disease_type: str, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Make prediction for a specific disease type."""
        if disease_type not in self.models:
            raise ValueError(f"Model for {disease_type} not loaded")
        
        model = self.models[disease_type]
        config = self.model_configs[disease_type]
        
        # Get prediction with gradcam
        result = model.predict_with_gradcam(image_tensor, target_class)
        
        if result['success']:
            # Add class names and disease-specific info
            predicted_idx = result['predicted_class']
            result['prediction_label'] = config['class_names'][predicted_idx]
            result['class_names'] = config['class_names']
            result['disease_type'] = disease_type
        
        return result
    
    def get_model_info(self, disease_type: str = None) -> Dict[str, Any]:
        """Get information about loaded models."""
        if disease_type:
            if disease_type not in self.models:
                return {'error': f'Model for {disease_type} not loaded'}
            
            config = self.model_configs[disease_type]
            return {
                'disease_type': disease_type,
                'num_classes': config['num_classes'],
                'class_names': config['class_names'],
                'checkpoint_path': config['checkpoint_path'],
                'device': str(self.device),
                'loaded': True
            }
        else:
            # Return info for all models
            info = {
                'device': str(self.device),
                'total_models': len(self.model_configs),
                'loaded_models': len(self.models),
                'models': {}
            }
            
            for disease in self.model_configs:
                config = self.model_configs[disease]
                info['models'][disease] = {
                    'num_classes': config['num_classes'],
                    'class_names': config['class_names'],
                    'loaded': disease in self.models
                }
            
            return info


# Global model manager
model_manager = None

def initialize_models():
    """Initialize all disease models."""
    global model_manager
    
    device = get_device()
    print(f"Initializing models on device: {device}")
    
    model_manager = MultiDiseaseModelManager(device)
    success = model_manager.load_all_models()
    
    if not success:
        print("Warning: Not all models loaded successfully")
    
    return model_manager


def preprocess_image(image_base64: str):
    """
    Preprocess base64 image for model inference.
    
    Args:
        image_base64 (str): Base64 encoded image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Decode base64 image
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Multi-Disease ResNet50 API", version="2.0.0")

# Pydantic models
class ImageInput(BaseModel):
    """Request model for image input"""
    image_base64: str
    generate_gradcam: bool = True
    target_class: Optional[int] = None

class PredictionResponse(BaseModel):
    """Response model for prediction with GradCAM"""
    success: bool
    disease_type: str
    predicted_class: int
    confidence: float
    probabilities: List[float]
    prediction_label: str
    class_names: List[str]
    gradcam_heatmap: Optional[List[List[float]]] = None
    target_class: Optional[int] = None
    message: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]
    device: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("Starting Multi-Disease ResNet50 API...")
    initialize_models()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global model_manager
    
    if model_manager is None:
        return HealthResponse(
            status="error",
            models_loaded={},
            device="unknown"
        )
    
    models_status = {}
    for disease in model_manager.model_configs:
        models_status[disease] = disease in model_manager.models
    
    return HealthResponse(
        status="healthy" if all(models_status.values()) else "partial",
        models_loaded=models_status,
        device=str(model_manager.device)
    )

@app.post("/predict/pneumonia", response_model=PredictionResponse)
async def predict_pneumonia(request: ImageInput):
    """Predict pneumonia from chest X-ray."""
    return await _predict_disease("pneumonia", request)

@app.post("/predict/brain_tumor", response_model=PredictionResponse)  
async def predict_brain_tumor(request: ImageInput):
    """Predict brain tumor from MRI scan."""
    return await _predict_disease("brain_tumor", request)

@app.post("/predict/tb", response_model=PredictionResponse)
async def predict_tb(request: ImageInput):
    """Predict tuberculosis from chest X-ray.""" 
    return await _predict_disease("tb", request)

async def _predict_disease(disease_type: str, request: ImageInput):
    """Common prediction function for all diseases."""
    global model_manager
    
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    if disease_type not in model_manager.models:
        raise HTTPException(status_code=500, detail=f"{disease_type} model not loaded")
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(request.image_base64)
        image_tensor = image_tensor.to(model_manager.device)
        
        # Make prediction
        result = model_manager.predict(disease_type, image_tensor, request.target_class)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {result.get('error', 'Unknown error')}")
        
        return PredictionResponse(
            success=True,
            disease_type=disease_type,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            prediction_label=result['prediction_label'],
            class_names=result['class_names'],
            gradcam_heatmap=result['gradcam_heatmap'] if request.generate_gradcam else None,
            target_class=result['target_class'],
            message=f"Successfully predicted {result['prediction_label']} with {result['confidence']:.2%} confidence"
        )
        
    except Exception as e:
        print(f"Prediction error for {disease_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about all loaded models."""
    global model_manager
    
    if model_manager is None:
        return {"error": "Models not initialized"}
    
    return model_manager.get_model_info()

@app.get("/model_info/{disease_type}")
async def get_disease_model_info(disease_type: str):
    """Get information about a specific disease model."""
    global model_manager
    
    if model_manager is None:
        return {"error": "Models not initialized"}
    
    return model_manager.get_model_info(disease_type)

def run_api(port: int = 6010, host: str = "0.0.0.0"):
    """Run the FastAPI application."""
    print(f"Starting Multi-Disease ResNet50 API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_api() 