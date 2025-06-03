#!/usr/bin/env python3
"""
InceptionV3 Model for OpenMed - Inference Only
An InceptionV3 implementation with ImageNet pretrained weights for medical image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
import os
from typing import Optional, Dict, Any, Tuple

from .base_model import BaseModel

class InceptionV3Model(BaseModel):
    """
    InceptionV3 model with ImageNet pretrained weights - Inference Only.
    
    This implementation adapts the standard InceptionV3 architecture for medical
    image classification while maintaining compatibility with the BaseModel interface.
    """
    
    def __init__(self, 
                 num_classes: int = 2, 
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 transform_input: bool = False,
                 **kwargs):
        """
        Initialize InceptionV3 model for inference.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate for the classifier
            transform_input: Whether to transform input from [0,1] to [-1,1]
        """
        super(InceptionV3Model, self).__init__(num_classes=num_classes, **kwargs)
        
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.transform_input = transform_input
        
        # Load pretrained InceptionV3 (no aux_logits for inference)
        if pretrained:
            self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, 
                                              aux_logits=False,
                                              transform_input=transform_input)
            self.logger.info("Loaded InceptionV3 with ImageNet pretrained weights")
        else:
            self.backbone = models.inception_v3(weights=None, 
                                              aux_logits=False,
                                              transform_input=transform_input)
            self.logger.info("Loaded InceptionV3 without pretrained weights")
        
        # Get the number of features from the original classifier
        num_features = self.backbone.fc.in_features
        
        # Replace the main classifier
        if dropout_rate > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Set to evaluation mode by default
        self.eval()
        
        self.logger.info(f"InceptionV3 initialized: {num_features} -> {num_classes} classes")
        if dropout_rate > 0:
            self.logger.info(f"Dropout rate: {dropout_rate}")
        
        # Print model info
        self.print_model_info()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InceptionV3 model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
               Expected input size is 299x299 for InceptionV3
            
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
        class InceptionFeatureExtractor(nn.Module):
            def __init__(self, inception_model):
                super().__init__()
                
                # Copy all attributes except the classifier
                for name, module in inception_model.named_children():
                    if name != 'fc':  # Skip the final classifier
                        setattr(self, name, module)
                
                # Ensure we have the transform_input attribute
                self.transform_input = getattr(inception_model, 'transform_input', False)
            
            def forward(self, x):
                # Input transformation (if enabled)
                if self.transform_input:
                    x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
                    x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
                    x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
                    x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
                
                # Forward through the network (following torchvision InceptionV3 structure)
                try:
                    # Initial convolution layers
                    x = self.Conv2d_1a_3x3(x)
                    x = self.Conv2d_2a_3x3(x)
                    x = self.Conv2d_2b_3x3(x)
                    x = self.maxpool1(x)
                    x = self.Conv2d_3b_1x1(x)
                    x = self.Conv2d_4a_3x3(x)
                    x = self.maxpool2(x)
                    
                    # Inception blocks
                    x = self.Mixed_5b(x)
                    x = self.Mixed_5c(x)
                    x = self.Mixed_5d(x)
                    x = self.Mixed_6a(x)
                    x = self.Mixed_6b(x)
                    x = self.Mixed_6c(x)
                    x = self.Mixed_6d(x)
                    x = self.Mixed_6e(x)
                    x = self.Mixed_7a(x)
                    x = self.Mixed_7b(x)
                    x = self.Mixed_7c(x)
                    
                    # Final pooling and flattening
                    x = self.avgpool(x)
                    if hasattr(self, 'dropout'):
                        x = self.dropout(x)
                    x = torch.flatten(x, 1)
                    
                    return x
                    
                except AttributeError as e:
                    # Fallback: if any layer is missing, try a more generic approach
                    # This handles cases where the model structure might be different
                    print(f"Warning: Expected layer missing ({e}). Using fallback approach.")
                    
                    # Apply all non-fc modules in order
                    for name, module in self.named_children():
                        if name != 'fc' and not name.startswith('_'):
                            try:
                                x = module(x)
                            except Exception as forward_error:
                                print(f"Warning: Could not apply layer {name}: {forward_error}")
                                continue
                    
                    # Ensure output is flattened
                    if len(x.shape) > 2:
                        x = torch.flatten(x, 1)
                    
                    return x
        
        return InceptionFeatureExtractor(self.backbone)
    
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
        return features(x)
    
    def get_layer_for_gradcam(self) -> nn.Module:
        """
        Get the layer typically used for GradCAM visualization.
        For InceptionV3, this is usually the last mixed layer (Mixed_7c).
        
        Returns:
            Layer module suitable for GradCAM
        """
        return self.backbone.Mixed_7c
    
    def get_attention_maps(self, x: torch.Tensor, layer_name: str = "Mixed_7c") -> torch.Tensor:
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
    
    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str, device: str = 'cpu', **kwargs) -> 'InceptionV3Model':
        """
        Load InceptionV3 model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded InceptionV3Model instance
            
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
            # Load state dict - handle both direct state dict and checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                # Assume it's a direct state dict
                model.load_state_dict(checkpoint)
            else:
                # Try to load directly
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load model state dict. This might be due to incompatible "
                             f"checkpoint format or different model configuration. Error: {e}")
        
        model.to(device)
        model.eval()
        
        logging.getLogger(cls.__name__).info(f"Model loaded successfully from {checkpoint_path}")
        return model
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """
        Get InceptionV3-specific model information.
        
        Returns:
            Dictionary with InceptionV3-specific information
        """
        base_info = self.get_model_info()
        inception_info = {
            'architecture': 'InceptionV3',
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'transform_input': self.transform_input,
            'backbone_features': 2048,  # InceptionV3 feature dimension
            'input_size': '299x299 (required)',
            'normalization': 'ImageNet normalization recommended'
        }
        return {**base_info, **inception_info} 