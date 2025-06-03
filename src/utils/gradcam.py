#!/usr/bin/env python3
"""
GradCAM Implementation for OpenMed Models - Inference Only
Provides GradCAM visualization functionality for ResNet50, ViT, and InceptionV3 models.
Based on the original implementation in openMed/src/rd/resnet50_gradcam.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Try to import cv2, provide fallback if not available
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available. Some visualization features may be limited.")

class GradCAM:
    """
    GradCAM implementation for generating activation heatmaps.
    
    This implementation works with any PyTorch model and can target specific layers
    for gradient-based class activation mapping.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize GradCAM with a model and target layer.
        
        Args:
            model: The model to analyze
            target_layer: The layer to hook for gradient computation
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook function to save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook function to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate GradCAM heatmap for the given input.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W)
            class_idx: Target class index. If None, uses predicted class.
            
        Returns:
            CAM heatmap as numpy array
        """
        # Ensure input requires gradients
        input_tensor.requires_grad_(True)
        
        # Set model to evaluation mode but enable gradient computation
        self.model.eval()
        
        # Reset stored gradients and activations
        self.gradients = None
        self.activations = None
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = class_idx.item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass on the target class score
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Verify hooks captured data
        if self.gradients is None:
            raise RuntimeError("Gradients were not captured. Check hook registration.")
        if self.activations is None:
            raise RuntimeError("Activations were not captured. Check hook registration.")
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients to get importance weights
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU to keep only positive influence
        cam = F.relu(cam)
        
        # Normalize to 0-1 range
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.detach().cpu().numpy()
    
    def remove_hooks(self):
        """Remove the registered hooks."""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()
    
    def __del__(self):
        """Ensure hooks are removed when object is destroyed."""
        try:
            self.remove_hooks()
        except:
            pass

class ModelGradCAM:
    """
    High-level interface for GradCAM visualization with OpenMed models.
    
    This class provides model-specific GradCAM functionality and handles
    the complexities of different architectures automatically.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize ModelGradCAM with a trained model.
        
        Args:
            model: Trained model instance
            device: Device to run computations on
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
        # Get the appropriate target layer for each model type
        self.target_layer = self._get_target_layer()
        
        # Initialize GradCAM
        self.gradcam = GradCAM(self.model, self.target_layer)
    
    def _get_target_layer(self) -> nn.Module:
        """Get the appropriate target layer for GradCAM based on model type."""
        model_name = self.model.__class__.__name__.lower()
        
        try:
            if 'resnet' in model_name:
                # For ResNet50: use the last convolutional layer in layer4
                if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layer4'):
                    return self.model.backbone.layer4[-1].conv3
                elif hasattr(self.model, 'layer4'):
                    return self.model.layer4[-1].conv3
                else:
                    raise AttributeError("ResNet model structure not as expected")
                    
            elif 'vit' in model_name:
                # For ViT: use the last transformer block's norm layer
                if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
                    return self.model.blocks[-1].norm1
                else:
                    raise AttributeError("ViT model structure not as expected")
                    
            elif 'inception' in model_name:
                # For InceptionV3: use the last mixed layer
                if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'Mixed_7c'):
                    return self.model.backbone.Mixed_7c
                elif hasattr(self.model, 'Mixed_7c'):
                    return self.model.Mixed_7c
                else:
                    raise AttributeError("InceptionV3 model structure not as expected")
            else:
                # Unknown model type, try fallback
                raise ValueError(f"Unknown model type: {model_name}")
                
        except (AttributeError, ValueError) as e:
            # Fallback: try to find a suitable layer automatically
            print(f"Warning: {e}. Attempting automatic layer detection...")
            return self._find_suitable_layer_fallback()
    
    def _find_suitable_layer_fallback(self) -> nn.Module:
        """
        Fallback method to find a suitable layer for GradCAM when model structure is unexpected.
        
        Returns:
            A suitable layer for GradCAM
            
        Raises:
            ValueError: If no suitable layer is found
        """
        # Strategy 1: Find the last convolutional layer
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is not None:
            print(f"Using last convolutional layer for GradCAM")
            return last_conv
        
        # Strategy 2: Find the last layer before classifier that has spatial dimensions
        suitable_layers = []
        for name, module in self.model.named_modules():
            # Look for layers that typically have spatial features
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
                suitable_layers.append((name, module))
        
        if suitable_layers:
            # Use the last suitable layer
            layer_name, layer = suitable_layers[-1]
            print(f"Using layer '{layer_name}' for GradCAM")
            return layer
        
        # Strategy 3: For transformer models, find the last norm layer
        last_norm = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                last_norm = module
        
        if last_norm is not None:
            print(f"Using last normalization layer for GradCAM")
            return last_norm
        
        # If all strategies fail
        raise ValueError(
            f"Could not determine appropriate target layer for model type: {self.model.__class__.__name__}. "
            f"Available modules: {list(dict(self.model.named_modules()).keys())[:10]}..."
        )
    
    def generate_gradcam(self, 
                        image_path: str, 
                        class_idx: Optional[int] = None,
                        input_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Generate GradCAM visualization for an image.
        
        Args:
            image_path: Path to the input image
            class_idx: Target class index (if None, uses predicted class)
            input_size: Input size for the model (if None, uses model defaults)
            
        Returns:
            Dictionary containing:
                - 'cam': GradCAM heatmap
                - 'prediction': Model prediction
                - 'confidence': Prediction confidence
                - 'original_image': Original image as numpy array
                - 'input_tensor': Preprocessed input tensor
        """
        # Determine input size based on model
        if input_size is None:
            model_name = self.model.__class__.__name__.lower()
            if 'inception' in model_name:
                input_size = (299, 299)
            else:
                input_size = (224, 224)
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        
        # Preprocessing transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Display transform (without normalization)
        display_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        
        # Prepare tensors
        input_tensor = preprocess(original_image).unsqueeze(0).to(self.device)
        display_image = display_transform(original_image).permute(1, 2, 0).numpy()
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Generate GradCAM
        cam = self.gradcam.generate_cam(input_tensor, class_idx or predicted_class)
        
        return {
            'cam': cam,
            'prediction': predicted_class,
            'confidence': confidence,
            'original_image': display_image,
            'input_tensor': input_tensor,
            'probabilities': probabilities.cpu().numpy()
        }
    
    def generate_gradcam_for_tensor(self,
                                   input_tensor: torch.Tensor,
                                   class_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate GradCAM visualization for a preprocessed tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            Dictionary containing CAM results
        """
        # Ensure tensor is on the correct device
        if input_tensor.device != self.device:
            print(f"Warning: Input tensor device ({input_tensor.device}) differs from model device ({self.device}). Moving tensor.")
            input_tensor = input_tensor.to(self.device)
        
        # Ensure model is on the correct device
        self.model.to(self.device)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            try:
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            except RuntimeError as e:
                if "device" in str(e).lower():
                    raise RuntimeError(f"Device mismatch error: {e}. "
                                     f"Model device: {self.device}, Input device: {input_tensor.device}")
                else:
                    raise e
        
        # Generate GradCAM
        try:
            cam = self.gradcam.generate_cam(input_tensor, class_idx or predicted_class)
        except RuntimeError as e:
            if "device" in str(e).lower():
                raise RuntimeError(f"Device mismatch in GradCAM generation: {e}")
            else:
                raise e
        
        return {
            'cam': cam,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()
        }

def overlay_heatmap(heatmap: np.ndarray, 
                   image: np.ndarray, 
                   alpha: float = 0.5, 
                   colormap: str = 'jet') -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlay a heatmap on an image for visualization.
    
    Args:
        heatmap: Heatmap array (0-1 normalized)
        image: Original image array (0-1 normalized RGB)
        alpha: Transparency factor for the heatmap (0-1)
        colormap: Colormap to apply ('jet', 'hot', 'viridis', etc.)
        
    Returns:
        Tuple of (colored_heatmap, overlaid_image)
    """
    if HAS_OPENCV:
        return _overlay_heatmap_opencv(heatmap, image, alpha, colormap)
    else:
        return _overlay_heatmap_matplotlib(heatmap, image, alpha, colormap)

def _overlay_heatmap_opencv(heatmap: np.ndarray, 
                           image: np.ndarray, 
                           alpha: float, 
                           colormap: str) -> Tuple[np.ndarray, np.ndarray]:
    """OpenCV-based heatmap overlay (preferred method)."""
    # Ensure proper formats
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    if heatmap.max() <= 1.0:
        heatmap = (heatmap * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    
    # Resize heatmap to match image size if needed
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    colormap_cv = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
    colored_heatmap = cv2.applyColorMap(heatmap, colormap_cv)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = image
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(image_rgb, 1-alpha, colored_heatmap_rgb, alpha, 0)
    
    return colored_heatmap_rgb, overlay

def _overlay_heatmap_matplotlib(heatmap: np.ndarray, 
                               image: np.ndarray, 
                               alpha: float, 
                               colormap: str) -> Tuple[np.ndarray, np.ndarray]:
    """Matplotlib-based heatmap overlay (fallback method)."""
    import matplotlib.cm as cm
    
    # Resize heatmap if needed
    if heatmap.shape != image.shape[:2]:
        from scipy.ndimage import zoom
        scale_h = image.shape[0] / heatmap.shape[0]
        scale_w = image.shape[1] / heatmap.shape[1]
        heatmap = zoom(heatmap, (scale_h, scale_w))
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Create overlay
    overlay = (1 - alpha) * image + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return colored_heatmap, overlay

def visualize_gradcam(gradcam_result: Dict[str, Any], 
                     class_names: Optional[list] = None,
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (15, 5),
                     alpha: float = 0.4) -> None:
    """
    Create a comprehensive GradCAM visualization.
    
    Args:
        gradcam_result: Result dictionary from generate_gradcam
        class_names: List of class names for labeling
        save_path: Optional path to save the visualization
        figsize: Figure size for the plot
        alpha: Transparency for heatmap overlay
    """
    cam = gradcam_result['cam']
    prediction = gradcam_result['prediction']
    confidence = gradcam_result['confidence']
    original_image = gradcam_result['original_image']
    
    # Create overlay
    colored_heatmap, overlay_img = overlay_heatmap(cam, original_image, alpha=alpha)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM heatmap
    axes[1].imshow(colored_heatmap)
    pred_label = class_names[prediction] if class_names else f"Class {prediction}"
    axes[1].set_title(f'GradCAM Heatmap\nPred: {pred_label} ({confidence:.3f})')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay_img)
    axes[2].set_title('Overlay\nFocus Areas')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save visualization to {save_path}: {e}")
    
    # Show the plot only if in an interactive environment
    try:
        # Check if we're in an interactive backend
        backend = plt.get_backend()
        if backend != 'Agg' and backend != 'svg':
            # Additional check for display availability
            import os
            if os.environ.get('DISPLAY') or os.name == 'nt':  # Windows or X11 display available
                plt.show()
            else:
                print("Display not available. Visualization saved to file only.")
                if not save_path:
                    print("Note: No save_path provided. Visualization not saved.")
        else:
            print(f"Non-interactive backend ({backend}) detected. Visualization saved to file only.")
            if not save_path:
                print("Note: No save_path provided. Visualization not saved.")
    except Exception as e:
        print(f"Warning: Could not display visualization: {e}")
        if not save_path:
            print("Note: No save_path provided. Visualization not saved.")
    finally:
        # Close the figure to free memory
        plt.close(fig)

def batch_gradcam_analysis(model: nn.Module,
                          image_paths: list,
                          class_names: Optional[list] = None,
                          save_dir: str = "./gradcam_results",
                          device: Optional[torch.device] = None) -> list:
    """
    Perform GradCAM analysis on a batch of images.
    
    Args:
        model: Trained model instance
        image_paths: List of paths to images
        class_names: List of class names
        save_dir: Directory to save results
        device: Device to run on
        
    Returns:
        List of GradCAM results for each image
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize ModelGradCAM
    model_gradcam = ModelGradCAM(model, device)
    
    results = []
    
    try:
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Generate GradCAM
                result = model_gradcam.generate_gradcam(image_path)
                results.append(result)
                
                # Create visualization
                save_path = os.path.join(save_dir, f"gradcam_{i+1}_{os.path.splitext(os.path.basename(image_path))[0]}.png")
                visualize_gradcam(result, class_names, save_path)
                
                pred_label = class_names[result['prediction']] if class_names else f"Class {result['prediction']}"
                print(f"  Prediction: {pred_label} (Confidence: {result['confidence']:.3f})")
                
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue
    
    finally:
        # Clean up
        model_gradcam.gradcam.remove_hooks()
    
    print(f"\nBatch analysis complete. Results saved in: {save_dir}")
    return results

# Convenience functions for model-specific GradCAM
def create_gradcam_for_model(model: nn.Module, device: Optional[torch.device] = None) -> ModelGradCAM:
    """
    Create a ModelGradCAM instance for any supported model.
    
    Args:
        model: Trained model instance
        device: Device to run on
        
    Returns:
        ModelGradCAM instance
    """
    return ModelGradCAM(model, device) 