#!/usr/bin/env python3
"""
Test script to verify OpenMed models work correctly.
Run this to check for bugs and ensure everything is functioning.
"""

import torch
import sys
import os

# Add current directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from model_factory import ModelFactory, create_resnet50, create_vit, create_inception_v3
        from resnet50 import ResNet50Model
        from vit import ViTModel
        from inception_v3 import InceptionV3Model
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation and basic functionality."""
    print("\nTesting model creation...")
    
    try:
        from model_factory import create_resnet50, create_vit, create_inception_v3
        
        # Test ResNet50
        resnet = create_resnet50(num_classes=2)
        print(f"✓ ResNet50 created: {resnet.get_num_trainable_params():,} trainable params")
        
        # Test ViT
        vit = create_vit(num_classes=2, img_size=224)
        print(f"✓ ViT created: {vit.get_num_trainable_params():,} trainable params")
        
        # Test InceptionV3
        inception = create_inception_v3(num_classes=2)
        print(f"✓ InceptionV3 created: {inception.get_num_trainable_params():,} trainable params")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass for all models."""
    print("\nTesting forward pass...")
    
    try:
        from model_factory import create_resnet50, create_vit, create_inception_v3
        
        # Test ResNet50
        resnet = create_resnet50(num_classes=2)
        x_resnet = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            out_resnet = resnet(x_resnet)
            prob_resnet = resnet.predict_proba(x_resnet)
            pred_resnet = resnet.predict(x_resnet)
        
        assert out_resnet.shape == (2, 2), f"ResNet output shape wrong: {out_resnet.shape}"
        assert prob_resnet.shape == (2, 2), f"ResNet probabilities shape wrong: {prob_resnet.shape}"
        assert pred_resnet.shape == (2,), f"ResNet predictions shape wrong: {pred_resnet.shape}"
        print("✓ ResNet50 forward pass successful")
        
        # Test ViT
        vit = create_vit(num_classes=2, img_size=224)
        x_vit = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            out_vit = vit(x_vit)
            prob_vit = vit.predict_proba(x_vit)
            pred_vit = vit.predict(x_vit)
        
        assert out_vit.shape == (2, 2), f"ViT output shape wrong: {out_vit.shape}"
        print("✓ ViT forward pass successful")
        
        # Test InceptionV3
        inception = create_inception_v3(num_classes=2)
        x_inception = torch.randn(2, 3, 299, 299)  # InceptionV3 uses 299x299
        
        with torch.no_grad():
            out_inception = inception(x_inception)
            if hasattr(out_inception, 'logits'):
                # Handle InceptionOutputs during training
                out_inception = out_inception.logits
            
        print("✓ InceptionV3 forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_weights():
    """Test the fixed ViT attention weights extraction."""
    print("\nTesting ViT attention weights...")
    
    try:
        from model_factory import create_vit
        
        vit = create_vit(num_classes=2, img_size=224)
        x = torch.randn(1, 3, 224, 224)
        
        # Test attention weights extraction
        attention_weights = vit.get_attention_weights(x, layer_idx=-1)
        
        expected_shape = (1, vit.num_heads, vit.n_patches + 1, vit.n_patches + 1)
        assert attention_weights.shape == expected_shape, f"Attention weights shape wrong: {attention_weights.shape}, expected: {expected_shape}"
        
        # Should not be all zeros (unless it falls back)
        if torch.all(attention_weights == 0):
            print("⚠ Attention weights are zeros (fallback mode)")
        else:
            print("✓ ViT attention weights extraction successful")
        
        return True
    except Exception as e:
        print(f"✗ Attention weights test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test feature extraction for all models."""
    print("\nTesting feature extraction...")
    
    try:
        from model_factory import create_resnet50, create_vit, create_inception_v3
        
        # Test ResNet50 features
        resnet = create_resnet50(num_classes=2)
        x_resnet = torch.randn(2, 3, 224, 224)
        features_resnet = resnet.extract_features(x_resnet)
        assert features_resnet.shape == (2, 2048), f"ResNet features shape wrong: {features_resnet.shape}"
        print("✓ ResNet50 feature extraction successful")
        
        # Test ViT features
        vit = create_vit(num_classes=2, img_size=224, embed_dim=768)
        x_vit = torch.randn(2, 3, 224, 224)
        features_vit = vit.extract_features(x_vit)
        assert features_vit.shape == (2, 768), f"ViT features shape wrong: {features_vit.shape}"
        print("✓ ViT feature extraction successful")
        
        # Test InceptionV3 features
        inception = create_inception_v3(num_classes=2)
        x_inception = torch.randn(2, 3, 299, 299)
        features_inception = inception.extract_features(x_inception)
        assert features_inception.shape == (2, 2048), f"InceptionV3 features shape wrong: {features_inception.shape}"
        print("✓ InceptionV3 feature extraction successful")
        
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layer_access():
    """Test the improved layer access method."""
    print("\nTesting layer access...")
    
    try:
        from model_factory import create_resnet50
        
        resnet = create_resnet50(num_classes=2)
        
        # Test exact match
        layer = resnet.get_layer_by_name("backbone.layer4")
        assert layer is not None, "Could not find backbone.layer4"
        
        # Test partial match
        layer = resnet.get_layer_by_name("layer4")
        assert layer is not None, "Could not find layer4 with partial match"
        
        # Test non-existent layer
        layer = resnet.get_layer_by_name("nonexistent_layer")
        assert layer is None, "Should return None for non-existent layer"
        
        print("✓ Layer access improvements working")
        return True
    except Exception as e:
        print(f"✗ Layer access test failed: {e}")
        return False

def test_model_factory():
    """Test model factory functionality."""
    print("\nTesting model factory...")
    
    try:
        from model_factory import ModelFactory
        
        # Test model listing
        models = ModelFactory.get_available_models()
        assert 'resnet50' in models, "ResNet50 not in available models"
        assert 'vit' in models, "ViT not in available models"
        assert 'inception_v3' in models, "InceptionV3 not in available models"
        
        # Test model creation via factory
        model = ModelFactory.create_model('resnet50', num_classes=3)
        assert model.num_classes == 3, "Model created with wrong number of classes"
        
        # Test dataset-specific configuration
        model_small = ModelFactory.create_model_for_dataset('resnet50', num_classes=2, dataset_size='small')
        assert model_small.freeze_features == True, "Small dataset should have frozen features"
        
        print("✓ Model factory tests passed")
        return True
    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("OpenMed Models - Bug Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_attention_weights,
        test_feature_extraction,
        test_layer_access,
        test_model_factory
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! No critical bugs detected.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return failed == 0

if __name__ == "__main__":
    main() 