#!/usr/bin/env python3
"""
Test Suite for GradCAM Functionality
Tests the GradCAM implementation with all supported models.
"""

import os
import sys
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add the models directory to the path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add the utils directory for GradCAM imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

def create_test_image(size=(224, 224), filename="test_image.jpg"):
    """Create a test image for GradCAM testing."""
    # Create a simple pattern image
    height, width = size
    image = np.zeros((height, width, 3))
    
    # Create a pattern that should be detectable by models
    center_h, center_w = height // 2, width // 2
    radius = min(height, width) // 4
    
    for i in range(height):
        for j in range(width):
            # Create a circular pattern
            dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
            if dist < radius:
                image[i, j, 0] = 1.0  # Red circle
            else:
                image[i, j, 1] = 0.5  # Green background
    
    # Add some noise
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # Save the image
    plt.imsave(filename, image)
    return filename

def test_gradcam_imports():
    """Test that GradCAM modules can be imported."""
    print("Testing GradCAM imports...")
    
    try:
        from gradcam import GradCAM, ModelGradCAM, overlay_heatmap, visualize_gradcam
        from model_factory import create_resnet50, create_vit, create_inception_v3
        print("✓ All GradCAM imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_gradcam_creation():
    """Test creating ModelGradCAM instances for all models."""
    print("\nTesting ModelGradCAM creation...")
    
    try:
        from gradcam import ModelGradCAM
        from model_factory import create_resnet50, create_vit, create_inception_v3
        
        models = {
            'ResNet50': create_resnet50(num_classes=2),
            'ViT': create_vit(num_classes=2),
            'InceptionV3': create_inception_v3(num_classes=2)
        }
        
        success_count = 0
        for model_name, model in models.items():
            try:
                gradcam = ModelGradCAM(model, torch.device('cpu'))
                print(f"✓ {model_name} ModelGradCAM created successfully")
                gradcam.gradcam.remove_hooks()
                success_count += 1
            except Exception as e:
                print(f"✗ {model_name} ModelGradCAM creation failed: {e}")
        
        return success_count == len(models)
        
    except Exception as e:
        print(f"✗ ModelGradCAM creation test failed: {e}")
        return False

def test_gradcam_generation():
    """Test GradCAM generation with different models."""
    print("\nTesting GradCAM generation...")
    
    try:
        from gradcam import ModelGradCAM
        from model_factory import create_resnet50
        
        # Create a simple model and test image
        model = create_resnet50(num_classes=2)
        test_image_path = create_test_image()
        
        try:
            gradcam = ModelGradCAM(model, torch.device('cpu'))
            result = gradcam.generate_gradcam(test_image_path)
            
            # Check result structure
            required_keys = ['cam', 'prediction', 'confidence', 'original_image', 'probabilities']
            if all(key in result for key in required_keys):
                print("✓ GradCAM generation successful")
                print(f"  - CAM shape: {result['cam'].shape}")
                print(f"  - Prediction: {result['prediction']}")
                print(f"  - Confidence: {result['confidence']:.3f}")
                success = True
            else:
                print(f"✗ Missing required keys in result: {set(required_keys) - set(result.keys())}")
                success = False
                
            gradcam.gradcam.remove_hooks()
            
        finally:
            # Clean up test image
            try:
                os.remove(test_image_path)
            except:
                pass
        
        return success
        
    except Exception as e:
        print(f"✗ GradCAM generation test failed: {e}")
        return False

def test_model_builtin_methods():
    """Test the built-in GradCAM methods in models."""
    print("\nTesting model built-in GradCAM methods...")
    
    try:
        from model_factory import create_resnet50
        
        model = create_resnet50(num_classes=2)
        test_image_path = create_test_image()
        
        try:
            # Test generate_gradcam_image
            result1 = model.generate_gradcam_image(test_image_path)
            print("✓ model.generate_gradcam_image() works")
            
            # Test generate_gradcam_tensor
            dummy_tensor = torch.randn(1, 3, 224, 224)
            result2 = model.generate_gradcam_tensor(dummy_tensor)
            print("✓ model.generate_gradcam_tensor() works")
            
            # Test visualize_gradcam_image (without showing the plot)
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = os.path.join(temp_dir, "test_gradcam.png")
                result3 = model.visualize_gradcam_image(
                    test_image_path, 
                    class_names=['Class0', 'Class1'],
                    save_path=save_path
                )
                
                # Check if file was created
                if os.path.exists(save_path):
                    print("✓ model.visualize_gradcam_image() works")
                    success = True
                else:
                    print("✗ Visualization file not created")
                    success = False
            
        finally:
            try:
                os.remove(test_image_path)
            except:
                pass
        
        return success
        
    except Exception as e:
        print(f"✗ Model built-in methods test failed: {e}")
        return False

def test_overlay_functionality():
    """Test the heatmap overlay functionality."""
    print("\nTesting heatmap overlay functionality...")
    
    try:
        from gradcam import overlay_heatmap
        
        # Create test data
        heatmap = np.random.rand(224, 224)
        image = np.random.rand(224, 224, 3)
        
        # Test overlay
        colored_heatmap, overlay_img = overlay_heatmap(heatmap, image, alpha=0.5)
        
        # Check outputs
        if colored_heatmap.shape == (224, 224, 3) and overlay_img.shape == (224, 224, 3):
            print("✓ Heatmap overlay functionality works")
            print(f"  - Colored heatmap shape: {colored_heatmap.shape}")
            print(f"  - Overlay image shape: {overlay_img.shape}")
            return True
        else:
            print(f"✗ Unexpected output shapes: {colored_heatmap.shape}, {overlay_img.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Overlay functionality test failed: {e}")
        return False

def test_batch_analysis():
    """Test batch GradCAM analysis."""
    print("\nTesting batch GradCAM analysis...")
    
    try:
        from model_factory import create_resnet50
        
        model = create_resnet50(num_classes=2)
        
        # Create multiple test images
        test_image_paths = []
        for i in range(3):
            path = create_test_image(filename=f"test_batch_{i}.jpg")
            test_image_paths.append(path)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                results = model.batch_gradcam_analysis(
                    image_paths=test_image_paths,
                    class_names=['Normal', 'Pneumonia'],
                    save_dir=temp_dir
                )
                
                if len(results) == len(test_image_paths):
                    print(f"✓ Batch analysis successful for {len(results)} images")
                    return True
                else:
                    print(f"✗ Expected {len(test_image_paths)} results, got {len(results)}")
                    return False
                    
        finally:
            # Clean up test images
            for path in test_image_paths:
                try:
                    os.remove(path)
                except:
                    pass
        
    except Exception as e:
        print(f"✗ Batch analysis test failed: {e}")
        return False

def test_different_models():
    """Test GradCAM with different model architectures."""
    print("\nTesting GradCAM with different models...")
    
    try:
        from model_factory import create_resnet50, create_vit, create_inception_v3
        
        models = {
            'ResNet50': (create_resnet50(num_classes=2), (224, 224)),
            'ViT': (create_vit(num_classes=2), (224, 224)),
            'InceptionV3': (create_inception_v3(num_classes=2), (299, 299))
        }
        
        success_count = 0
        
        for model_name, (model, input_size) in models.items():
            try:
                test_image_path = create_test_image(size=input_size, filename=f"test_{model_name.lower()}.jpg")
                
                result = model.generate_gradcam_image(test_image_path)
                
                print(f"✓ {model_name} GradCAM successful")
                print(f"  - Input size: {input_size}")
                print(f"  - CAM shape: {result['cam'].shape}")
                print(f"  - Prediction: {result['prediction']}")
                
                success_count += 1
                
                # Clean up
                os.remove(test_image_path)
                
            except Exception as e:
                print(f"✗ {model_name} GradCAM failed: {e}")
        
        return success_count == len(models)
        
    except Exception as e:
        print(f"✗ Different models test failed: {e}")
        return False

def test_error_handling():
    """Test error handling in GradCAM functionality."""
    print("\nTesting error handling...")
    
    try:
        from gradcam import ModelGradCAM
        from model_factory import create_resnet50
        
        model = create_resnet50(num_classes=2)
        
        # Test with non-existent image
        try:
            result = model.generate_gradcam_image("non_existent_image.jpg")
            print("✗ Should have failed with non-existent image")
            return False
        except Exception:
            print("✓ Correctly handles non-existent image")
        
        # Test with invalid tensor shape
        try:
            invalid_tensor = torch.randn(1, 2, 224, 224)  # Wrong number of channels
            result = model.generate_gradcam_tensor(invalid_tensor)
            print("✗ Should have failed with invalid tensor shape")
            return False
        except Exception:
            print("✓ Correctly handles invalid tensor shape")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def main():
    """Run all GradCAM tests."""
    print("GradCAM Functionality Test Suite")
    print("=" * 50)
    
    tests = [
        test_gradcam_imports,
        test_model_gradcam_creation,
        test_gradcam_generation,
        test_model_builtin_methods,
        test_overlay_functionality,
        test_batch_analysis,
        test_different_models,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"GradCAM Tests Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All GradCAM tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 