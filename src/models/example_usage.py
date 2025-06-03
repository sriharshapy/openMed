#!/usr/bin/env python3
"""
Example Usage of OpenMed Models - Inference Only
Demonstrates how to use the new model implementations for medical image classification inference.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import logging

# Import the models
try:
    # Try relative import first (when imported as module)
    from . import (
        ModelFactory, 
        ResNet50Model, 
        ViTModel, 
        InceptionV3Model,
        create_resnet50,
        create_vit,
        create_inception_v3,
        compare_models
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from model_factory import (
        ModelFactory,
        create_resnet50,
        create_vit,
        create_inception_v3,
        compare_models
    )
    from resnet50 import ResNet50Model
    from vit import ViTModel
    from inception_v3 import InceptionV3Model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_model_creation():
    """Demonstrate different ways to create models for inference."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL CREATION FOR INFERENCE")
    print("="*60)
    
    # Method 1: Direct instantiation
    print("\n1. Direct instantiation:")
    resnet = ResNet50Model(num_classes=3)
    vit = ViTModel(num_classes=3, img_size=224)
    inception = InceptionV3Model(num_classes=3)
    
    # Method 2: Using factory
    print("\n2. Using ModelFactory:")
    resnet_factory = ModelFactory.create_model('resnet50', num_classes=3)
    
    # Method 3: Using convenience functions
    print("\n3. Using convenience functions:")
    resnet_conv = create_resnet50(num_classes=3, dropout_rate=0.2)

def demonstrate_model_features():
    """Demonstrate key features of the models for inference."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL FEATURES FOR INFERENCE")
    print("="*60)
    
    # Create a sample model
    model = create_resnet50(num_classes=2)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print("\n1. Basic forward pass:")
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)
        predictions = model.predict(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    print("\n2. Feature extraction:")
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    print("\n3. Model information:")
    info = model.get_model_specific_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

def demonstrate_inference():
    """Demonstrate inference capabilities."""
    print("\n" + "="*60)
    print("DEMONSTRATING INFERENCE")
    print("="*60)
    
    # Create models for different inference scenarios
    models = {
        'ResNet50': create_resnet50(num_classes=2),
        'ViT': create_vit(num_classes=2),
        'InceptionV3': create_inception_v3(num_classes=2)
    }
    
    for name, model in models.items():
        print(f"\n{name} inference:")
        
        # Create appropriate input size for each model
        if name == 'InceptionV3':
            dummy_input = torch.randn(2, 3, 299, 299)  # InceptionV3 requires 299x299
        else:
            dummy_input = torch.randn(2, 3, 224, 224)  # ResNet50 and ViT use 224x224
        
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
            probabilities = model.predict_proba(dummy_input)
            predictions = model.predict(dummy_input)
        
        print(f"  Output shape: {outputs.shape}")
        print(f"  Predictions: {predictions.cpu().numpy()}")
        print(f"  Max probability: {probabilities.max(dim=1)[0].cpu().numpy()}")

def demonstrate_model_loading():
    """Demonstrate model loading from checkpoints."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL LOADING")
    print("="*60)
    
    # Create and save a simple model
    model = create_resnet50(num_classes=2)
    
    # Save checkpoint
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, 'model_checkpoint.pth')
        
        # Save model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': model.get_model_info()
        }, checkpoint_path)
        
        print(f"Model saved to: {checkpoint_path}")
        
        # Load model using class method
        try:
            loaded_model, checkpoint_info = ResNet50Model.load_checkpoint(
                checkpoint_path, 
                num_classes=2
            )
            print("Model loaded successfully using BaseModel.load_checkpoint()")
            print(f"Checkpoint info: {checkpoint_info}")
            
            # Test loaded model
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = loaded_model(dummy_input)
            print(f"Loaded model inference successful, output shape: {output.shape}")
            
        except Exception as e:
            print(f"Error loading model: {e}")

def demonstrate_attention_visualization():
    """Demonstrate attention and feature visualization capabilities."""
    print("\n" + "="*60)
    print("DEMONSTRATING ATTENTION/FEATURE VISUALIZATION")
    print("="*60)
    
    # Create models
    resnet = create_resnet50(num_classes=2)
    vit = create_vit(num_classes=2)
    
    # Create dummy inputs
    dummy_input_224 = torch.randn(1, 3, 224, 224)
    
    print("\n1. ResNet50 attention maps:")
    try:
        attention_maps = resnet.get_attention_maps(dummy_input_224, layer_name="layer4")
        print(f"  Attention maps shape: {attention_maps.shape}")
    except Exception as e:
        print(f"  Error getting attention maps: {e}")
    
    print("\n2. ViT attention weights:")
    try:
        attention_weights = vit.get_attention_weights(dummy_input_224, layer_idx=-1)
        print(f"  Attention weights shape: {attention_weights.shape}")
        
        patch_attention = vit.get_patch_attention_map(dummy_input_224, layer_idx=-1)
        print(f"  Patch attention shape: {patch_attention.shape}")
    except Exception as e:
        print(f"  Error getting attention weights: {e}")

def demonstrate_compatibility_with_rd():
    """Demonstrate compatibility with existing rd/ implementation."""
    print("\n" + "="*60)
    print("DEMONSTRATING COMPATIBILITY WITH RD/ IMPLEMENTATION")
    print("="*60)
    
    print("The new ResNet50Model is designed to be compatible with the existing")
    print("implementation in openMed/rd/resnet50.py")
    print("\nKey compatibility features:")
    print("- Same ImageNet pretrained weights")
    print("- Compatible input/output shapes")
    print("- Can load checkpoints from rd/ implementation")
    print("- Provides same core functionality")
    
    # Create model
    model = create_resnet50(num_classes=2)
    
    # Show model info
    print(f"\nModel architecture: {model.get_model_specific_info()['architecture']}")
    print(f"Backbone features: {model.get_model_specific_info()['backbone_features']}")
    print(f"Input size: {model.get_model_specific_info()['input_size']}")

def main():
    """Run all demonstrations."""
    print("OpenMed Models - Inference Only Examples")
    print("=" * 80)
    
    # Run demonstrations
    demonstrate_model_creation()
    demonstrate_model_features()
    demonstrate_inference()
    demonstrate_model_loading()
    demonstrate_attention_visualization()
    demonstrate_compatibility_with_rd()
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    compare_models()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main() 