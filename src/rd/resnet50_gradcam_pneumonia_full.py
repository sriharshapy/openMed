#!/usr/bin/env python3
"""
ResNet50 GradCAM Visualization Script for Pneumonia Classification (Full Network Training)
This script provides GradCAM (Gradient-weighted Class Activation Mapping) visualization
for the ResNet50 model trained with full network training on pneumonia chest X-ray images.
Based on the TB full training GradCAM framework but adapted for pneumonia classification.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2

# Import the model classes from the training script
# Assuming they're in the same directory or properly installed
from resnet50_pneumonia_full import ResNet50FullTraining, PneumoniaDataset, get_device

def load_trained_model(model_path, num_classes=2, device=None):
    """Load a trained ResNet50 model from checkpoint."""
    if device is None:
        device = get_device()
    
    # Initialize model with GradCAM enabled
    model = ResNet50FullTraining(
        num_classes=num_classes,
        freeze_features=False,  # Full training model
        enable_gradcam=True
    )
    
    # Load state dict
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Device: {device}")
    print(f"GradCAM enabled: {model.enable_gradcam}")
    
    return model

def generate_gradcam_visualization(
    model,
    image_path,
    class_names,
    device,
    transform=None,
    save_path=None,
    show_plot=True,
    alpha=0.5,
    target_class=None
):
    """
    Generate GradCAM visualization for a single image.
    
    Args:
        model: Trained ResNet50 model with GradCAM enabled
        image_path: Path to the input image
        class_names: List of class names
        device: Device for computation
        transform: Image preprocessing transform
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot
        alpha: Transparency for heatmap overlay
        target_class: Specific class to visualize (if None, uses predicted class)
    
    Returns:
        dict: Dictionary containing predictions, heatmap, and other information
    """
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    original_np = np.array(original_image)
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Preprocess image
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
        probabilities = F.softmax(predictions, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Generate GradCAM
    if target_class is None:
        target_class = predicted_class
    
    heatmap = model.gradcam.generate_cam(input_tensor, target_class)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # GradCAM heatmap
    im1 = axes[1].imshow(heatmap, cmap='jet', alpha=0.8)
    axes[1].set_title(f'GradCAM Heatmap\n(Target Class: {class_names[target_class]})', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    original_resized = cv2.resize(original_np, (224, 224))
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_resized, 1-alpha, heatmap_colored, alpha, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'GradCAM Overlay\n(α={alpha})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add prediction information
    pred_text = f'Predicted: {class_names[predicted_class]} ({confidence:.3f})'
    if target_class != predicted_class:
        target_confidence = probabilities[0, target_class].item()
        pred_text += f'\nTarget: {class_names[target_class]} ({target_confidence:.3f})'
    
    fig.suptitle(f'Pneumonia Classification GradCAM Analysis\n{pred_text}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return {
        'predictions': predictions.cpu().numpy(),
        'probabilities': probabilities.cpu().numpy(),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'target_class': target_class,
        'heatmap': heatmap,
        'overlay': overlay,
        'original_image': original_np
    }

def batch_gradcam_analysis(
    model,
    data_dir,
    class_names,
    device,
    num_samples_per_class=5,
    save_dir="./gradcam_results_pneumonia_full",
    transform=None
):
    """
    Perform GradCAM analysis on multiple samples from each class.
    
    Args:
        model: Trained ResNet50 model with GradCAM enabled
        data_dir: Directory containing test data
        class_names: List of class names
        device: Device for computation
        num_samples_per_class: Number of samples to analyze per class
        save_dir: Directory to save results
        transform: Image preprocessing transform
    
    Returns:
        dict: Analysis results for all samples
    """
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\nAnalyzing class: {class_name}")
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Get image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        
        if len(image_files) == 0:
            print(f"No images found in {class_dir}")
            continue
        
        # Randomly sample images
        random.seed(42)
        sampled_files = random.sample(image_files, min(num_samples_per_class, len(image_files)))
        
        class_results = []
        
        for i, img_file in enumerate(sampled_files):
            img_path = os.path.join(class_dir, img_file)
            print(f"  Processing {img_file}...")
            
            try:
                # Generate GradCAM
                result = generate_gradcam_visualization(
                    model=model,
                    image_path=img_path,
                    class_names=class_names,
                    device=device,
                    transform=transform,
                    save_path=os.path.join(save_dir, f"{class_name}_{i+1}_{img_file}_gradcam.png"),
                    show_plot=False,
                    alpha=0.5
                )
                
                result['image_path'] = img_path
                result['image_file'] = img_file
                result['true_class'] = class_idx
                result['true_class_name'] = class_name
                
                class_results.append(result)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        results[class_name] = class_results
    
    return results

def create_comparison_visualization(
    model,
    image_paths,
    class_names,
    device,
    save_path=None,
    transform=None
):
    """
    Create a comparison visualization showing GradCAM for multiple images.
    
    Args:
        model: Trained ResNet50 model with GradCAM enabled
        image_paths: List of image paths to compare
        class_names: List of class names
        device: Device for computation
        save_path: Path to save the comparison plot
        transform: Image preprocessing transform
    
    Returns:
        dict: Results for all images
    """
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    num_images = len(image_paths)
    fig, axes = plt.subplots(3, num_images, figsize=(6*num_images, 18))
    
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    results = []
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{num_images}: {os.path.basename(img_path)}")
        
        # Load and preprocess image
        original_image = Image.open(img_path).convert('RGB')
        original_np = np.array(original_image)
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            predictions = model(input_tensor)
            probabilities = F.softmax(predictions, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Generate GradCAM
        heatmap = model.gradcam.generate_cam(input_tensor, predicted_class)
        
        # Create overlay
        original_resized = cv2.resize(original_np, (224, 224))
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_resized, 0.5, heatmap_colored, 0.5, 0)
        
        # Plot original image
        axes[0, i].imshow(original_image)
        axes[0, i].set_title(f'Original\n{os.path.basename(img_path)}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Plot heatmap
        im = axes[1, i].imshow(heatmap, cmap='jet')
        axes[1, i].set_title(f'GradCAM Heatmap\nPred: {class_names[predicted_class]}', fontsize=12, fontweight='bold')
        axes[1, i].axis('off')
        
        # Plot overlay
        axes[2, i].imshow(overlay)
        axes[2, i].set_title(f'Overlay\nConfidence: {confidence:.3f}', fontsize=12, fontweight='bold')
        axes[2, i].axis('off')
        
        results.append({
            'image_path': img_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'heatmap': heatmap
        })
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()
    
    return results

def analyze_model_predictions(
    model,
    test_data_dir,
    class_names,
    device,
    num_samples=50,
    transform=None
):
    """
    Analyze model predictions and generate summary statistics.
    
    Args:
        model: Trained ResNet50 model
        test_data_dir: Directory containing test data
        class_names: List of class names
        device: Device for computation
        num_samples: Number of samples to analyze per class
        transform: Image preprocessing transform
    
    Returns:
        dict: Analysis results and statistics
    """
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    results = {
        'correct_predictions': 0,
        'total_predictions': 0,
        'class_accuracies': {},
        'confusion_matrix': np.zeros((len(class_names), len(class_names))),
        'confidence_scores': [],
        'predictions_by_class': {}
    }
    
    model.eval()
    
    for true_class_idx, class_name in enumerate(class_names):
        print(f"\nAnalyzing {class_name} samples...")
        class_dir = os.path.join(test_data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        # Get image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        
        # Sample images
        random.seed(42)
        sampled_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        class_correct = 0
        class_total = 0
        class_predictions = []
        
        for img_file in sampled_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    predictions = model(input_tensor)
                    probabilities = F.softmax(predictions, dim=1)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                
                # Update statistics
                class_total += 1
                results['total_predictions'] += 1
                results['confusion_matrix'][true_class_idx, predicted_class] += 1
                results['confidence_scores'].append(confidence)
                
                class_predictions.append({
                    'image_file': img_file,
                    'true_class': true_class_idx,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities.cpu().numpy().flatten()
                })
                
                if predicted_class == true_class_idx:
                    class_correct += 1
                    results['correct_predictions'] += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        results['class_accuracies'][class_name] = class_accuracy
        results['predictions_by_class'][class_name] = class_predictions
        
        print(f"{class_name} accuracy: {class_accuracy:.3f} ({class_correct}/{class_total})")
    
    # Calculate overall accuracy
    overall_accuracy = results['correct_predictions'] / results['total_predictions']
    results['overall_accuracy'] = overall_accuracy
    
    print(f"\nOverall accuracy: {overall_accuracy:.3f} ({results['correct_predictions']}/{results['total_predictions']})")
    
    return results

def main():
    """Main function demonstrating GradCAM visualization for pneumonia classification."""
    
    # Configuration
    config = {
        "model_path": "C:/Users/sriha/NEU/shlabs/HP_NVIDIA/openMed/checkpoints/resnet50_pneumonia_full/best_resnet50_pneumonia_full_trained.pth",
        "data_path": "C:/Users/sriha/NEU/shlabs/HP_NVIDIA/CellData/chest_xray/test",
        "num_classes": 2,
        "class_names": ["NORMAL", "PNEUMONIA"],  # Alphabetical order (as loaded by dataset)
        "save_dir": "./gradcam_results_pneumonia_full",
        "num_samples_per_class": 3
    }
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(config["model_path"]):
        print(f"Error: Model file not found at {config['model_path']}")
        print("Please make sure you have trained the model first using resnet50_pneumonia_full.py")
        return
    
    # Check if data exists
    if not os.path.exists(config["data_path"]):
        print(f"Error: Test data not found at {config['data_path']}")
        print("Please make sure the pneumonia dataset exists.")
        return
    
    # Load trained model
    print("Loading trained ResNet50 model...")
    model = load_trained_model(
        model_path=config["model_path"],
        num_classes=config["num_classes"],
        device=device
    )
    
    # Create image preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*60)
    print("ResNet50 Pneumonia Classification - GradCAM Analysis")
    print("Full Network Training Model")
    print("="*60)
    
    # Perform batch GradCAM analysis
    print("\n1. Performing batch GradCAM analysis...")
    batch_results = batch_gradcam_analysis(
        model=model,
        data_dir=config["data_path"],
        class_names=config["class_names"],
        device=device,
        num_samples_per_class=config["num_samples_per_class"],
        save_dir=config["save_dir"],
        transform=transform
    )
    
    # Analyze model predictions
    print("\n2. Analyzing model predictions...")
    prediction_results = analyze_model_predictions(
        model=model,
        test_data_dir=config["data_path"],
        class_names=config["class_names"],
        device=device,
        num_samples=20,  # Analyze 20 samples per class
        transform=transform
    )
    
    # Create confusion matrix visualization
    print("\n3. Creating confusion matrix visualization...")
    plt.figure(figsize=(8, 6))
    conf_matrix = prediction_results['confusion_matrix']
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=config["class_names"],
                yticklabels=config["class_names"])
    plt.title('Confusion Matrix - Pneumonia Classification\n(Full Network Training)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    
    conf_matrix_path = os.path.join(config["save_dir"], "confusion_matrix_pneumonia_full.png")
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {prediction_results['overall_accuracy']:.3f}")
    print(f"Total Samples Analyzed: {prediction_results['total_predictions']}")
    
    print("\nClass-wise Accuracies:")
    for class_name, accuracy in prediction_results['class_accuracies'].items():
        print(f"  {class_name}: {accuracy:.3f}")
    
    print(f"\nAverage Confidence: {np.mean(prediction_results['confidence_scores']):.3f}")
    print(f"Confidence Std: {np.std(prediction_results['confidence_scores']):.3f}")
    
    print(f"\nGradCAM visualizations saved in: {config['save_dir']}")
    print(f"Confusion matrix saved as: {conf_matrix_path}")
    
    # Example of single image analysis
    print("\n4. Single image GradCAM example...")
    
    # Find a sample image for demonstration
    for class_name in config["class_names"]:
        class_dir = os.path.join(config["data_path"], class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                sample_image = os.path.join(class_dir, images[0])
                print(f"Analyzing sample image: {sample_image}")
                
                single_result = generate_gradcam_visualization(
                    model=model,
                    image_path=sample_image,
                    class_names=config["class_names"],
                    device=device,
                    transform=transform,
                    save_path=os.path.join(config["save_dir"], f"single_example_{class_name}.png"),
                    show_plot=True,
                    alpha=0.5
                )
                break
    
    print("\nGradCAM analysis completed!")
    print("Note: This analysis used the FULL NETWORK TRAINED ResNet50 model.")

if __name__ == "__main__":
    main()