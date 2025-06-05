#!/usr/bin/env python3
"""
ResNet50 GradCAM Demo Script
This script demonstrates how to use the updated ResNet50 model 
with integrated GradCAM functionality.
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Import the updated ResNet50 model
from resnet50 import ResNet50FineTuned, get_device

def demo_gradcam():
    """Demonstrate GradCAM functionality with ResNet50."""
    
    # Get device
    device = get_device()
    
    # Create model with GradCAM enabled
    print("Creating ResNet50 model with GradCAM enabled...")
    model = ResNet50FineTuned(num_classes=2, freeze_features=True, enable_gradcam=True)
    model.to(device)
    model.eval()
    
    # Load a pre-trained model if available
    checkpoint_path = "./checkpoints/resnet50/best_resnet50_finetuned.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found. Using randomly initialized model for demo.")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Example with a dummy image (you can replace this with actual image path)
    print("\nDemo 1: Using model without GradCAM")
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Standard prediction
    with torch.no_grad():
        predictions = model(dummy_image)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities[0].cpu().numpy()}")
    
    print("\nDemo 2: Using model with GradCAM")
    # Prediction with GradCAM
    predictions, gradcam_heatmap = model(dummy_image, return_gradcam=True)
    probabilities = torch.softmax(predictions, dim=1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities[0].cpu().numpy()}")
    print(f"GradCAM heatmap shape: {gradcam_heatmap.shape}")
    print(f"GradCAM heatmap range: [{gradcam_heatmap.min():.3f}, {gradcam_heatmap.max():.3f}]")
    
    print("\nDemo 3: Getting GradCAM overlay")
    # Get GradCAM overlay
    try:
        colored_heatmap, overlay = model.get_gradcam_overlay(dummy_image)
        print(f"Colored heatmap shape: {colored_heatmap.shape}")
        print(f"Overlay shape: {overlay.shape}")
    except Exception as e:
        print(f"GradCAM overlay failed: {e}")
    
    print("\nDemo 4: Enable/Disable GradCAM dynamically")
    # Disable GradCAM
    model.disable_gradcam_mode()
    
    # This should return only predictions
    result = model(dummy_image, return_gradcam=True)
    if isinstance(result, tuple):
        print("GradCAM was still enabled (unexpected)")
    else:
        print("GradCAM successfully disabled - only predictions returned")
    
    # Re-enable GradCAM
    model.enable_gradcam_mode()
    
    # This should return predictions and GradCAM
    result = model(dummy_image, return_gradcam=True)
    if isinstance(result, tuple):
        print("GradCAM successfully re-enabled - predictions and heatmap returned")
    else:
        print("GradCAM was not re-enabled (unexpected)")
    
    print("\nDemo completed successfully!")

def demo_with_real_image(image_path):
    """Demonstrate GradCAM with a real image."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Get device
    device = get_device()
    
    # Create model with GradCAM enabled
    model = ResNet50FineTuned(num_classes=2, freeze_features=True, enable_gradcam=True)
    model.to(device)
    model.eval()
    
    # Load checkpoint if available
    checkpoint_path = "./checkpoints/resnet50/best_resnet50_finetuned.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get predictions and GradCAM
    predictions, gradcam_heatmap = model(input_tensor, return_gradcam=True)
    probabilities = torch.softmax(predictions, dim=1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = probabilities[0, predicted_class].item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.3f}")
    
    # Get GradCAM overlay
    colored_heatmap, overlay = model.get_gradcam_overlay(input_tensor, original_image)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM heatmap
    axes[1].imshow(colored_heatmap)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (Class: {predicted_class}, Conf: {confidence:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    output_dir = "./gradcam_demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, f"gradcam_demo_{os.path.basename(image_path)}.png"))
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    print("ResNet50 GradCAM Integration Demo")
    print("=" * 50)
    
    # Run basic demo
    demo_gradcam()
    
    # Example usage with real image
    # Uncomment the following lines and provide a valid image path
    # print("\n" + "=" * 50)
    # print("Demo with real image:")
    # demo_with_real_image("path/to/your/chest_xray_image.jpg") 