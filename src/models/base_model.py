#!/usr/bin/env python3
"""
Abstract Base Model for OpenMed - Inference Only
Defines the common interface and functionality that all models should implement for inference.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all OpenMed models - Inference Only.
    
    This class defines the common interface that all models should implement
    to ensure consistent integration with the rest of the system for inference tasks.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.model_name = self.__class__.__name__
        
        # Initialize logging
        self.logger = logging.getLogger(self.model_name)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    @abstractmethod
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extraction part of the model.
        Used for feature extraction.
        
        Returns:
            Feature extractor module
        """
        pass
    
    @abstractmethod
    def get_classifier(self) -> nn.Module:
        """
        Get the classification head of the model.
        
        Returns:
            Classifier module
        """
        pass
    
    def get_num_total_params(self) -> int:
        """
        Get the total number of parameters in the model.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_params': self.get_num_total_params()
        }
    
    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        print(f"\n=== {info['model_name']} Information ===")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Total parameters: {info['total_params']:,}")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        probabilities = self.predict_proba(x)
        return torch.argmax(probabilities, dim=1)
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu', **model_kwargs) -> Tuple['BaseModel', Dict]:
        """
        Load model from checkpoint for inference.
        
        Args:
            filepath: Path to the checkpoint file
            device: Device to load the model on
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Tuple of (model, checkpoint_info)
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        # Extract model info from checkpoint if available
        if 'model_info' in checkpoint:
            model_info = checkpoint['model_info']
            # Use checkpoint info to override kwargs if not explicitly provided
            for key in ['num_classes']:
                if key not in model_kwargs and key in model_info:
                    model_kwargs[key] = model_info[key]
        
        # Create model instance
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Return model and checkpoint info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch'),
            'loss': checkpoint.get('loss'),
            'metrics': checkpoint.get('metrics', {}),
            'model_info': checkpoint.get('model_info', {})
        }
        
        return model, checkpoint_info
    
    def get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """
        Get a layer by its name for hooks or visualization.
        Supports both exact matches and partial matches for nested modules.
        
        Args:
            name: Name of the layer (supports partial matching)
            
        Returns:
            The layer module if found, None otherwise
        """
        # First try exact match
        for layer_name, layer in self.named_modules():
            if layer_name == name:
                return layer
        
        # Try partial match (useful for nested modules)
        for layer_name, layer in self.named_modules():
            if name in layer_name:
                return layer
                
        # Try without prefix (e.g., looking for 'layer4' in 'backbone.layer4')
        if '.' in name:
            short_name = name.split('.')[-1]
            for layer_name, layer in self.named_modules():
                if layer_name.endswith(short_name):
                    return layer
        
        return None
    
    def get_layer_names(self) -> List[str]:
        """
        Get all layer names in the model.
        
        Returns:
            List of layer names
        """
        return [name for name, _ in self.named_modules()]
    
    # GradCAM functionality
    def generate_gradcam_image(self, 
                              image_path: str, 
                              class_idx: Optional[int] = None,
                              device: Optional[torch.device] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Generate GradCAM visualization for an image file.
        
        Args:
            image_path: Path to the input image
            class_idx: Target class index (if None, uses predicted class)
            device: Device to run computations on
            **kwargs: Additional arguments passed to ModelGradCAM
            
        Returns:
            Dictionary containing GradCAM results with standardized keys:
                - 'cam_heatmap': GradCAM heatmap array
                - 'predicted_class': Predicted class index
                - 'confidence': Prediction confidence
                - 'class_probabilities': Full probability distribution
                - 'original_image': Original image (if available)
                - 'input_tensor': Preprocessed input tensor
        """
        from ..utils.gradcam import ModelGradCAM
        
        gradcam = ModelGradCAM(self, device)
        try:
            raw_result = gradcam.generate_gradcam(image_path, class_idx, **kwargs)
            
            # Standardize the keys for consistent interface
            standardized_result = {
                'cam_heatmap': raw_result.get('cam'),
                'predicted_class': raw_result.get('prediction'),
                'confidence': raw_result.get('confidence'),
                'class_probabilities': raw_result.get('probabilities'),
                'original_image': raw_result.get('original_image'),
                'input_tensor': raw_result.get('input_tensor')
            }
            
            return standardized_result
        finally:
            gradcam.gradcam.remove_hooks()
    
    def generate_gradcam_tensor(self, 
                               input_tensor: torch.Tensor, 
                               class_idx: Optional[int] = None,
                               device: Optional[torch.device] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        Generate GradCAM visualization for a preprocessed tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            class_idx: Target class index (if None, uses predicted class)
            device: Device to run computations on
            **kwargs: Additional arguments passed to ModelGradCAM
            
        Returns:
            Dictionary containing GradCAM results with standardized keys:
                - 'cam_heatmap': GradCAM heatmap array
                - 'predicted_class': Predicted class index
                - 'confidence': Prediction confidence
                - 'class_probabilities': Full probability distribution
        """
        from ..utils.gradcam import ModelGradCAM
        
        gradcam = ModelGradCAM(self, device)
        try:
            raw_result = gradcam.generate_gradcam_for_tensor(input_tensor, class_idx, **kwargs)
            
            # Standardize the keys for consistent interface
            standardized_result = {
                'cam_heatmap': raw_result.get('cam'),
                'predicted_class': raw_result.get('prediction'),
                'confidence': raw_result.get('confidence'),
                'class_probabilities': raw_result.get('probabilities')
            }
            
            return standardized_result
        finally:
            gradcam.gradcam.remove_hooks()
    
    def visualize_gradcam_image(self, 
                               image_path: str, 
                               class_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None,
                               device: Optional[torch.device] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        Generate and visualize GradCAM for an image with automatic plotting.
        
        Args:
            image_path: Path to the input image
            class_names: List of class names for labeling
            save_path: Optional path to save the visualization
            device: Device to run computations on
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing GradCAM results
        """
        from ..utils.gradcam import visualize_gradcam
        
        result = self.generate_gradcam_image(image_path, device=device, **kwargs)
        visualize_gradcam(result, class_names, save_path, **kwargs)
        return result
    
    def batch_gradcam_analysis(self,
                              image_paths: List[str],
                              class_names: Optional[List[str]] = None,
                              save_dir: str = "./gradcam_results",
                              device: Optional[torch.device] = None,
                              **kwargs) -> List[Dict[str, Any]]:
        """
        Perform GradCAM analysis on multiple images.
        
        Args:
            image_paths: List of paths to images
            class_names: List of class names for labeling
            save_dir: Directory to save results
            device: Device to run on
            **kwargs: Additional arguments
            
        Returns:
            List of GradCAM results for each image
        """
        from ..utils.gradcam import batch_gradcam_analysis
        
        return batch_gradcam_analysis(self, image_paths, class_names, save_dir, device, **kwargs) 