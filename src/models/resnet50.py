#!/usr/bin/env python3
"""
ResNet50 Model for OpenMed - Inference Only
A ResNet50 implementation with ImageNet pretrained weights for medical image classification.
Compatible with the existing ResNet50 implementation in rd/ but with improved modularity.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import logging
import os
from typing import Optional, Dict, Any

from .base_model import BaseModel

class ResNet50Model(BaseModel):
    """
    ResNet50 model with ImageNet pretrained weights - Inference Only.
    
    This implementation provides the same functionality as the original ResNet50
    in the rd/ directory but follows the BaseModel interface for better integration.
    """
    
    def __init__(self, 
                 num_classes: int = 2, 
                 pretrained: bool = True,
                 dropout_rate: float = 0.0,
                 **kwargs):
        """
        Initialize ResNet50 model for inference.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate for the classifier (0.0 = no dropout)
        """
        super(ResNet50Model, self).__init__(num_classes=num_classes, **kwargs)
        
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.logger.info("Loaded ResNet50 with ImageNet pretrained weights")
        else:
            self.backbone = models.resnet50(weights=None)
            self.logger.info("Loaded ResNet50 without pretrained weights")
        
        # Get the number of features from the original classifier
        num_features = self.backbone.fc.in_features
        
        # Create new classifier
        if dropout_rate > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Set to evaluation mode by default
        self.eval()
        
        self.logger.info(f"ResNet50 initialized: {num_features} -> {num_classes} classes")
        if dropout_rate > 0:
            self.logger.info(f"Dropout rate: {dropout_rate}")
        
        # Print model info
        self.print_model_info()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet50 model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extraction part of the model (everything except fc layer).
        
        Returns:
            Feature extractor module
        """
        # Create a sequential model with all layers except the final classifier
        features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool
        )
        return features
    
    def get_classifier(self) -> nn.Module:
        """
        Get the classification head of the model.
        
        Returns:
            Classifier module
        """
        return self.backbone.fc
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input without classification.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        features = self.get_feature_extractor()
        x = features(x)
        x = torch.flatten(x, 1)  # Flatten for feature vector
        return x
    
    def get_layer_for_gradcam(self) -> nn.Module:
        """
        Get the layer typically used for GradCAM visualization.
        For ResNet50, this is usually the last convolutional layer (layer4).
        
        Returns:
            Layer module suitable for GradCAM
        """
        return self.backbone.layer4
    
    def get_attention_maps(self, x: torch.Tensor, layer_name: str = "layer4") -> torch.Tensor:
        """
        Get attention maps from a specific layer.
        
        Args:
            x: Input tensor
            layer_name: Name of the layer to extract attention from
            
        Returns:
            Attention maps
        """
        layer = self.get_layer_by_name(f"backbone.{layer_name}")
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found")
        
        # Register hook to capture activations
        activations = {}
        def hook(module, input, output):
            activations['features'] = output.detach()
        
        handle = layer.register_forward_hook(hook)
        
        try:
            # Forward pass
            _ = self.forward(x)
            return activations['features']
        finally:
            handle.remove()
    
    @staticmethod
    def _map_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Map checkpoint keys from old format to new format.
        
        This handles the case where the checkpoint has keys with 'resnet50.' prefix
        but the new model expects 'backbone.' prefix.
        
        Args:
            state_dict: Original state dict from checkpoint
            
        Returns:
            Mapped state dict with corrected keys
        """
        mapped_dict = {}
        
        for key, value in state_dict.items():
            # Map old 'resnet50.' prefix to 'backbone.'
            if key.startswith('resnet50.'):
                new_key = key.replace('resnet50.', 'backbone.')
                mapped_dict[new_key] = value
            # Keep other keys as is
            else:
                mapped_dict[key] = value
        
        return mapped_dict

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str, device: str = 'cpu', **kwargs) -> 'ResNet50Model':
        """
        Load ResNet50 model from a checkpoint saved by the original rd/ implementation.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded ResNet50Model instance
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is corrupted or incompatible
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Create model instance
        model = cls(**kwargs)
        
        try:
            # Extract state dict and handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict):
                # Check if this is a direct state dict or has other keys
                if any(key.startswith(('resnet50.', 'backbone.')) for key in checkpoint.keys()):
                    # This looks like a state dict
                    state_dict = checkpoint
                else:
                    # Try to find the actual state dict in the checkpoint
                    for key in ['state_dict', 'model', 'net']:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            break
                    else:
                        # Assume the whole checkpoint is the state dict
                        state_dict = checkpoint
            else:
                # Try to load directly
                state_dict = checkpoint
            
            # Map checkpoint keys to match current model structure
            mapped_state_dict = cls._map_checkpoint_keys(state_dict)
            
            # Load the mapped state dict
            model.load_state_dict(mapped_state_dict, strict=False)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model state dict. This might be due to incompatible "
                             f"checkpoint format or different model configuration. Error: {e}")
        
        model.to(device)
        model.eval()
        
        logging.getLogger(cls.__name__).info(f"Model loaded successfully from {checkpoint_path}")
        return model
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """
        Get ResNet50-specific model information.
        
        Returns:
            Dictionary with ResNet50-specific information
        """
        base_info = self.get_model_info()
        resnet_info = {
            'architecture': 'ResNet50',
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'backbone_features': 2048,  # ResNet50 feature dimension
            'input_size': '224x224 (recommended)',
            'normalization': 'ImageNet normalization recommended'
        }
        return {**base_info, **resnet_info} 