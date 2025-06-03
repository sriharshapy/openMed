"""
OpenMed Models Package - Inference Only
Contains abstract base models and implementations for medical image classification inference.
"""

from .base_model import BaseModel
from .resnet50 import ResNet50Model
from .vit import ViTModel
from .inception_v3 import InceptionV3Model
from .model_factory import (
    ModelFactory, 
    create_resnet50, 
    create_vit, 
    create_inception_v3,
    compare_models
)
from ..utils.gradcam import (
    GradCAM,
    ModelGradCAM,
    overlay_heatmap,
    visualize_gradcam,
    batch_gradcam_analysis,
    create_gradcam_for_model
)

__all__ = [
    # Base model
    'BaseModel',
    
    # Model implementations
    'ResNet50Model', 
    'ViTModel',
    'InceptionV3Model',
    
    # Factory and utilities
    'ModelFactory',
    'create_resnet50',
    'create_vit', 
    'create_inception_v3',
    'compare_models',
    
    # GradCAM functionality
    'GradCAM',
    'ModelGradCAM',
    'overlay_heatmap',
    'visualize_gradcam',
    'batch_gradcam_analysis',
    'create_gradcam_for_model'
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "OpenMed Team" 