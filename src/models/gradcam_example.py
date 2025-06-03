#!/usr/bin/env python3
"""
GradCAM Example Usage for OpenMed Models
Demonstrates how to use the GradCAM functionality with all supported models.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the models and GradCAM functionality
try:
    # Try relative import first (when imported as module)
    from . import (
        create_resnet50, 
        create_vit, 
        create_inception_v3,
        ModelGradCAM,
        visualize_gradcam,
        batch_gradcam_analysis
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from model_factory import create_resnet50, create_vit, create_inception_v3
    # Import from utils directory for standalone execution
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
    from gradcam import ModelGradCAM, visualize_gradcam, batch_gradcam_analysis

def demonstrate_single_image_gradcam():
    """Demonstrate GradCAM on a single image with all models."""
    print("\n" + "="*60)
    print("DEMONSTRATING SINGLE IMAGE GRADCAM")
    print("="*60)
    
    # Create models
    models = {
        'ResNet50': create_resnet50(num_classes=2),
        'ViT': create_vit(num_classes=2),
        'InceptionV3': create_inception_v3(num_classes=2)
    }
    
    # Create a dummy image for demonstration
    # In practice, you would use a real medical image path
    dummy_image_path = create_dummy_image()
    
    class_names = ['Normal', 'Pneumonia']  # Example class names
    
    for model_name, model in models.items():
        print(f"\n{model_name} GradCAM Analysis:")
        
        try:
            # Method 1: Using the model's built-in method
            result = model.visualize_gradcam_image(
                image_path=dummy_image_path,
                class_names=class_names,
                save_path=f"./gradcam_{model_name.lower()}_example.png"
            )
            
            print(f"  Predicted class: {class_names[result['prediction']]}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Visualization saved")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")

def demonstrate_tensor_gradcam():
    """Demonstrate GradCAM on preprocessed tensors."""
    print("\n" + "="*60)
    print("DEMONSTRATING TENSOR GRADCAM")
    print("="*60)
    
    # Create a model
    model = create_resnet50(num_classes=2)
    
    # Create dummy input tensors (batch of images)
    dummy_batch = torch.randn(3, 3, 224, 224)  # 3 images, 3 channels, 224x224
    
    print("Analyzing batch of 3 images...")
    
    for i in range(dummy_batch.shape[0]):
        input_tensor = dummy_batch[i:i+1]  # Single image
        
        # Generate GradCAM
        result = model.generate_gradcam_tensor(input_tensor)
        
        print(f"  Image {i+1}: Predicted class {result['prediction']}, "
              f"Confidence: {result['confidence']:.3f}")

def demonstrate_batch_analysis():
    """Demonstrate batch GradCAM analysis."""
    print("\n" + "="*60)
    print("DEMONSTRATING BATCH GRADCAM ANALYSIS")
    print("="*60)
    
    # Create model
    model = create_resnet50(num_classes=2)
    
    # Create dummy images for demonstration
    dummy_image_paths = [create_dummy_image(f"dummy_image_{i}.jpg") for i in range(3)]
    class_names = ['Normal', 'Pneumonia']
    
    print("Running batch analysis on 3 images...")
    
    try:
        # Perform batch analysis
        results = model.batch_gradcam_analysis(
            image_paths=dummy_image_paths,
            class_names=class_names,
            save_dir="./gradcam_batch_results"
        )
        
        print(f"Batch analysis complete! Processed {len(results)} images.")
        
        # Summary of results
        for i, result in enumerate(results):
            pred_class = class_names[result['prediction']]
            confidence = result['confidence']
            print(f"  Image {i+1}: {pred_class} ({confidence:.3f})")
            
    except Exception as e:
        print(f"  Error in batch analysis: {e}")
    finally:
        # Clean up dummy images
        for path in dummy_image_paths:
            try:
                os.remove(path)
            except:
                pass

def demonstrate_model_comparison():
    """Compare GradCAM results across different models."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL COMPARISON")
    print("="*60)
    
    # Create all models
    models = {
        'ResNet50': create_resnet50(num_classes=2),
        'ViT': create_vit(num_classes=2), 
        'InceptionV3': create_inception_v3(num_classes=2)
    }
    
    # Create a dummy image
    dummy_image_path = create_dummy_image("comparison_image.jpg")
    
    print("Comparing GradCAM results across models...")
    
    results = {}
    
    try:
        for model_name, model in models.items():
            print(f"\nAnalyzing with {model_name}...")
            
            try:
                result = model.generate_gradcam_image(dummy_image_path)
                results[model_name] = result
                
                print(f"  Predicted class: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  CAM shape: {result['cam'].shape}")
                
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
        
        # Create comparison visualization
        if len(results) > 0:
            create_comparison_visualization(results, dummy_image_path)
            
    finally:
        # Clean up
        try:
            os.remove(dummy_image_path)
        except:
            pass

def demonstrate_custom_gradcam():
    """Demonstrate using ModelGradCAM directly for custom analysis."""
    print("\n" + "="*60)
    print("DEMONSTRATING CUSTOM GRADCAM USAGE")
    print("="*60)
    
    # Create model
    model = create_resnet50(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create ModelGradCAM instance
    gradcam = ModelGradCAM(model, device)
    
    # Create dummy image
    dummy_image_path = create_dummy_image("custom_analysis.jpg")
    
    try:
        print("Performing custom GradCAM analysis...")
        
        # Generate GradCAM for predicted class
        result_pred = gradcam.generate_gradcam(dummy_image_path)
        print(f"  Predicted class analysis: Class {result_pred['prediction']} "
              f"(Confidence: {result_pred['confidence']:.3f})")
        
        # Generate GradCAM for specific class (e.g., class 0)
        result_class0 = gradcam.generate_gradcam(dummy_image_path, class_idx=0)
        print(f"  Class 0 analysis: Confidence for class 0: "
              f"{result_class0['probabilities'][0][0]:.3f}")
        
        # Generate GradCAM for specific class (e.g., class 1)
        result_class1 = gradcam.generate_gradcam(dummy_image_path, class_idx=1)
        print(f"  Class 1 analysis: Confidence for class 1: "
              f"{result_class1['probabilities'][0][1]:.3f}")
        
        print("  Custom analysis complete!")
        
    finally:
        # Clean up
        gradcam.gradcam.remove_hooks()
        try:
            os.remove(dummy_image_path)
        except:
            pass

def demonstrate_target_layer_analysis():
    """Demonstrate analysis of different target layers."""
    print("\n" + "="*60)
    print("DEMONSTRATING TARGET LAYER ANALYSIS")
    print("="*60)
    
    model = create_resnet50(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Different target layers for ResNet50
    target_layers = {
        'layer4': model.backbone.layer4[-1].conv3,  # Default (last conv)
        'layer3': model.backbone.layer3[-1].conv3,  # Earlier layer
        'layer2': model.backbone.layer2[-1].conv3   # Even earlier
    }
    
    dummy_image_path = create_dummy_image("layer_analysis.jpg")
    
    try:
        print("Analyzing different target layers...")
        
        for layer_name, target_layer in target_layers.items():
            print(f"\n  Analyzing {layer_name}:")
            
            # Create GradCAM with specific target layer
            try:
                # Import GradCAM from utils
                from ..utils.gradcam import GradCAM
            except ImportError:
                # Fallback for standalone execution
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
                from gradcam import GradCAM
            
            gradcam = GradCAM(model, target_layer)
            
            try:
                # Load and preprocess image
                from PIL import Image
                import torchvision.transforms as transforms
                
                original_image = Image.open(dummy_image_path).convert('RGB')
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                input_tensor = preprocess(original_image).unsqueeze(0).to(device)
                
                # Generate CAM
                cam = gradcam.generate_cam(input_tensor)
                
                print(f"    CAM shape: {cam.shape}")
                print(f"    CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
                print(f"    CAM mean: {cam.mean():.3f}")
                
            except Exception as e:
                print(f"    Error: {e}")
            finally:
                gradcam.remove_hooks()
                
    finally:
        try:
            os.remove(dummy_image_path)
        except:
            pass

def create_dummy_image(filename: str = "dummy_image.jpg") -> str:
    """Create a dummy image for demonstration purposes."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a simple gradient image
    height, width = 224, 224
    image = np.zeros((height, width, 3))
    
    # Create a gradient pattern
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = i / height  # Red gradient
            image[i, j, 1] = j / width   # Green gradient
            image[i, j, 2] = (i + j) / (height + width)  # Blue gradient
    
    # Add some noise to make it more interesting
    noise = np.random.normal(0, 0.1, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # Save the image
    plt.imsave(filename, image)
    return filename

def create_comparison_visualization(results: dict, image_path: str):
    """Create a side-by-side comparison of GradCAM results."""
    try:
        # Import overlay_heatmap from utils
        from ..utils.gradcam import overlay_heatmap
    except ImportError:
        # Fallback for standalone execution
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
        from gradcam import overlay_heatmap
    
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Load original image for display
    original_image = Image.open(image_path).convert('RGB')
    display_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    display_image = display_transform(original_image).permute(1, 2, 0).numpy()
    
    # Create comparison plot
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    for i, (model_name, result) in enumerate(results.items()):
        # Original image
        axes[0, i].imshow(display_image)
        axes[0, i].set_title(f'{model_name}\nOriginal Image')
        axes[0, i].axis('off')
        
        # GradCAM overlay
        colored_heatmap, overlay_img = overlay_heatmap(result['cam'], display_image, alpha=0.4)
        axes[1, i].imshow(overlay_img)
        axes[1, i].set_title(f'{model_name} GradCAM\nPred: Class {result["prediction"]} ({result["confidence"]:.3f})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('./gradcam_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Model comparison visualization saved as 'gradcam_model_comparison.png'")

def main():
    """Run all GradCAM demonstrations."""
    print("OpenMed Models - GradCAM Examples")
    print("=" * 60)
    
    print("This script demonstrates GradCAM functionality with OpenMed models.")
    print("Note: This uses dummy images for demonstration. In practice, use real medical images.")
    
    # Run demonstrations
    demonstrate_single_image_gradcam()
    demonstrate_tensor_gradcam()
    demonstrate_batch_analysis()
    demonstrate_model_comparison()
    demonstrate_custom_gradcam()
    demonstrate_target_layer_analysis()
    
    print("\n" + "="*60)
    print("ALL GRADCAM DEMONSTRATIONS COMPLETED!")
    print("="*60)
    
    print("\nFiles created:")
    print("- Individual model GradCAM images")
    print("- Batch analysis results in './gradcam_batch_results/'")
    print("- Model comparison: './gradcam_model_comparison.png'")
    
    print("\nUsage Summary:")
    print("1. model.visualize_gradcam_image(image_path) - Quick visualization")
    print("2. model.generate_gradcam_image(image_path) - Get GradCAM data")
    print("3. model.generate_gradcam_tensor(tensor) - Use with preprocessed tensors")
    print("4. model.batch_gradcam_analysis(image_paths) - Batch processing")
    print("5. ModelGradCAM(model) - Advanced custom analysis")

if __name__ == "__main__":
    main() 