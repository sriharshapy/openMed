#!/usr/bin/env python3
"""
Model Factory for OpenMed - Inference Only
Provides easy instantiation and management of different model architectures for inference.
"""

from typing import Dict, Any, Type, Union
import logging

from .base_model import BaseModel
from .resnet50 import ResNet50Model
from .vit import ViTModel
from .inception_v3 import InceptionV3Model

# Registry of available models
MODEL_REGISTRY = {
    'resnet50': ResNet50Model,
    'vit': ViTModel,
    'inception_v3': InceptionV3Model,
}

class ModelFactory:
    """
    Factory class for creating model instances for inference.
    """
    
    @staticmethod
    def get_available_models() -> Dict[str, Type[BaseModel]]:
        """
        Get all available model architectures.
        
        Returns:
            Dictionary mapping model names to model classes
        """
        return MODEL_REGISTRY.copy()
    
    @staticmethod
    def list_models() -> None:
        """Print all available models with their descriptions."""
        print("\n=== Available Models ===")
        for name, model_class in MODEL_REGISTRY.items():
            print(f"- {name}: {model_class.__doc__.split('.')[0] if model_class.__doc__ else 'No description'}")
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseModel:
        """
        Create a model instance by name for inference.
        
        Args:
            model_name: Name of the model architecture
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model_name is not recognized
        """
        model_name = model_name.lower()
        
        if model_name not in MODEL_REGISTRY:
            available = ', '.join(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
        
        model_class = MODEL_REGISTRY[model_name]
        logger = logging.getLogger('ModelFactory')
        logger.info(f"Creating {model_name} model with args: {kwargs}")
        
        return model_class(**kwargs)
    
    @staticmethod
    def get_model_recommendations(task_type: str = 'medical_classification') -> Dict[str, str]:
        """
        Get model recommendations for different task types.
        
        Args:
            task_type: Type of task (e.g., 'medical_classification', 'general_classification')
            
        Returns:
            Dictionary with model recommendations and reasons
        """
        recommendations = {}
        
        if task_type == 'medical_classification':
            recommendations = {
                'resnet50': 'Excellent for medical imaging, proven performance, fast inference',
                'vit': 'State-of-the-art accuracy, good for larger datasets, attention visualization',
                'inception_v3': 'Good multi-scale feature extraction, suitable for complex medical images'
            }
        elif task_type == 'general_classification':
            recommendations = {
                'resnet50': 'Versatile and reliable, good starting point',
                'vit': 'Best for large datasets, transformer architecture',
                'inception_v3': 'Good for images with multiple scales of features'
            }
        else:
            recommendations = {model: 'General purpose model' for model in MODEL_REGISTRY.keys()}
        
        return recommendations
    
    @staticmethod
    def get_model_requirements(model_name: str) -> Dict[str, Any]:
        """
        Get specific requirements for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model requirements
        """
        model_name = model_name.lower()
        
        requirements = {
            'resnet50': {
                'input_size': (224, 224),
                'channels': 3,
                'min_image_size': (32, 32),
                'preprocessing': 'ImageNet normalization',
                'memory_usage': 'Low',
                'inference_speed': 'Fast',
                'extra_dependencies': []
            },
            'vit': {
                'input_size': (224, 224),
                'channels': 3,
                'min_image_size': (224, 224),  # ViT requires specific patch sizes
                'preprocessing': 'ImageNet normalization',
                'memory_usage': 'High',
                'inference_speed': 'Medium',
                'extra_dependencies': ['einops']
            },
            'inception_v3': {
                'input_size': (299, 299),
                'channels': 3,
                'min_image_size': (75, 75),
                'preprocessing': 'ImageNet normalization (special input transform available)',
                'memory_usage': 'Medium',
                'inference_speed': 'Medium',
                'extra_dependencies': []
            }
        }
        
        if model_name not in requirements:
            raise ValueError(f"Unknown model '{model_name}'")
        
        return requirements[model_name]

def create_resnet50(num_classes: int = 2, **kwargs) -> ResNet50Model:
    """Convenience function to create ResNet50 model for inference."""
    return ModelFactory.create_model('resnet50', num_classes=num_classes, **kwargs)

def create_vit(num_classes: int = 2, **kwargs) -> ViTModel:
    """Convenience function to create ViT model for inference."""
    return ModelFactory.create_model('vit', num_classes=num_classes, **kwargs)

def create_inception_v3(num_classes: int = 2, **kwargs) -> InceptionV3Model:
    """Convenience function to create InceptionV3 model for inference."""
    return ModelFactory.create_model('inception_v3', num_classes=num_classes, **kwargs)

def compare_models(models_to_compare: list = None) -> None:
    """
    Compare different models and their characteristics.
    
    Args:
        models_to_compare: List of model names to compare. If None, compares all available models.
    """
    if models_to_compare is None:
        models_to_compare = list(MODEL_REGISTRY.keys())
    
    print("\n=== Model Comparison ===")
    print(f"{'Model':<15} {'Input Size':<12} {'Memory':<8} {'Speed':<8} {'Best For'}")
    print("-" * 70)
    
    for model_name in models_to_compare:
        if model_name in MODEL_REGISTRY:
            req = ModelFactory.get_model_requirements(model_name)
            recommendations = ModelFactory.get_model_recommendations('medical_classification')
            
            input_size = f"{req['input_size'][0]}x{req['input_size'][1]}"
            memory = req['memory_usage']
            speed = req['inference_speed']
            best_for = recommendations.get(model_name, 'General use')[:30] + "..." if len(recommendations.get(model_name, '')) > 30 else recommendations.get(model_name, 'General use')
            
            print(f"{model_name:<15} {input_size:<12} {memory:<8} {speed:<8} {best_for}")
        else:
            print(f"{model_name:<15} {'Unknown':<12} {'?':<8} {'?':<8} {'Model not found'}")
    
    print("\nNote: All models support ImageNet pretrained weights and can be used for medical image classification.")

if __name__ == "__main__":
    # Example usage
    ModelFactory.list_models()
    compare_models()
    
    # Create a model for medical classification
    print("\n=== Creating ResNet50 for medical classification ===")
    model = ModelFactory.create_model('resnet50', num_classes=2)
    model.print_model_info() 