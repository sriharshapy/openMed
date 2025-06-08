#!/usr/bin/env python3
"""
ResNet50 Fine-tuning Script for Chest X-ray Classification
This script provides fine-tuning of ResNet50 pretrained on ImageNet,
where only the last fully connected layer is trainable.
Based on the SICDN training framework but simplified for transfer learning.
"""

import os
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.backends import mps

import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, confusion_matrix
from imblearn.metrics import sensitivity_score, specificity_score

def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA device detected: {torch.cuda.get_device_name()}")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device detected (Apple Silicon)")
        return device
    else:
        print("No GPU available, using CPU")
        return torch.device("cpu")

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================================
# GRADCAM IMPLEMENTATION
# ============================================================================

class GradCAM:
    """GradCAM implementation for visualization."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate GradCAM heatmap."""
        # Enable gradients and set model to train mode for gradient computation
        input_image.requires_grad_(True)
        original_mode = self.model.training
        self.model.train()
        
        # Reset gradients and activations
        self.gradients = None
        self.activations = None
        
        # Forward pass - call resnet50 directly to avoid recursion
        output = self.model.resnet50(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = class_idx.item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass on the specific class score
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Check if gradients and activations were captured
        if self.gradients is None:
            raise RuntimeError("Gradients were not captured. Check hook registration.")
        if self.activations is None:
            raise RuntimeError("Activations were not captured. Check hook registration.")
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Restore original model mode
        self.model.train(original_mode)
        
        return cam.detach().cpu().numpy()
    
    def remove_hooks(self):
        """Remove the registered hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ResNet50FineTuned(nn.Module):
    """ResNet50 with ImageNet pretrained weights, fine-tuning only the last layer."""
    
    def __init__(self, num_classes=2, freeze_features=True, enable_gradcam=False):
        super(ResNet50FineTuned, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze all layers except the final classifier if freeze_features=True
        if freeze_features:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        # ResNet50 has 2048 features in the final layer
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)
        
        # Ensure the final layer is trainable
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True
            
        # Initialize GradCAM if enabled
        self.enable_gradcam = enable_gradcam
        self.gradcam = None
        if enable_gradcam:
            # Use the last convolutional layer before avgpool (layer4[-1].conv3 in ResNet50)
            target_layer = self.resnet50.layer4[-1].conv3
            self.gradcam = GradCAM(self, target_layer)
            
        print(f"ResNet50 loaded with ImageNet weights. Final layer: {num_features} -> {num_classes}")
        if freeze_features:
            print("All layers frozen except final classifier layer")
        else:
            print("All layers trainable")
        if enable_gradcam:
            print("GradCAM functionality enabled")
    
    def forward(self, x, return_gradcam=False, gradcam_class_idx=None):
        """
        Forward pass with optional GradCAM generation.
        
        Args:
            x: Input tensor
            return_gradcam: Whether to return GradCAM heatmap along with predictions
            gradcam_class_idx: Specific class index for GradCAM (if None, uses predicted class)
            
        Returns:
            If return_gradcam=False: predictions tensor
            If return_gradcam=True: tuple of (predictions, gradcam_heatmap)
        """
        # Standard forward pass
        predictions = self.resnet50(x)
        
        # Return only predictions if GradCAM not requested or not enabled
        if not return_gradcam or not self.enable_gradcam or self.gradcam is None:
            return predictions
        
        # Generate GradCAM heatmap
        try:
            gradcam_heatmap = self.gradcam.generate_cam(x, gradcam_class_idx)
            return predictions, gradcam_heatmap
        except Exception as e:
            print(f"Warning: GradCAM generation failed: {e}")
            return predictions
    
    def enable_gradcam_mode(self):
        """Enable GradCAM functionality if not already enabled."""
        if not self.enable_gradcam:
            target_layer = self.resnet50.layer4[-1].conv3
            self.gradcam = GradCAM(self, target_layer)
            self.enable_gradcam = True
            print("GradCAM functionality enabled")
    
    def disable_gradcam_mode(self):
        """Disable GradCAM functionality and remove hooks."""
        if self.enable_gradcam and self.gradcam is not None:
            self.gradcam.remove_hooks()
            self.gradcam = None
            self.enable_gradcam = False
            print("GradCAM functionality disabled")
    
    def get_gradcam_overlay(self, input_image, original_image=None, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Generate GradCAM overlay on the original image.
        
        Args:
            input_image: Preprocessed input tensor for the model
            original_image: Original image (PIL Image or numpy array) for overlay
            alpha: Transparency factor for the heatmap overlay
            colormap: OpenCV colormap for the heatmap
            
        Returns:
            tuple: (colored_heatmap, overlaid_image) or None if GradCAM not enabled
        """
        if not self.enable_gradcam or self.gradcam is None:
            print("GradCAM not enabled. Call enable_gradcam_mode() first.")
            return None
        
        # Get predictions and GradCAM heatmap - gradients are needed for GradCAM
        predictions, gradcam_heatmap = self.forward(input_image, return_gradcam=True)
        
        # If no original image provided, try to create one from input tensor
        if original_image is None:
            # Denormalize the input tensor (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            
            denorm_image = input_image * std + mean
            denorm_image = torch.clamp(denorm_image, 0, 1)
            
            # Convert to numpy
            original_image = denorm_image[0].permute(1, 2, 0).cpu().numpy()
        
        # Convert PIL Image to numpy if needed
        if hasattr(original_image, 'convert'):  # PIL Image
            original_image = np.array(original_image.convert('RGB'))
        
        # Ensure image is in the right format (0-255 uint8)
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)
        
        # Ensure heatmap is in the right format (0-255 uint8)
        if gradcam_heatmap.max() <= 1.0:
            gradcam_heatmap = (gradcam_heatmap * 255).astype(np.uint8)
        else:
            gradcam_heatmap = gradcam_heatmap.astype(np.uint8)
        
        # Resize heatmap to match image size if needed
        if gradcam_heatmap.shape != original_image.shape[:2]:
            gradcam_heatmap = cv2.resize(gradcam_heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap to heatmap
        colored_heatmap = cv2.applyColorMap(gradcam_heatmap, colormap)
        
        # Convert BGR to RGB for display
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is RGB
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            image_rgb = original_image
        else:
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Create overlay using addWeighted
        overlay = cv2.addWeighted(image_rgb, 1-alpha, colored_heatmap_rgb, alpha, 0)
        
        return colored_heatmap_rgb, overlay
    
    def get_trainable_params(self):
        """Return only trainable parameters."""
        return [param for param in self.parameters() if param.requires_grad]
    
    def get_num_trainable_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SklearnResNet50Wrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for PyTorch ResNet50 model."""
    
    def __init__(self, pytorch_model, device, transform=None):
        self.pytorch_model = pytorch_model
        self.device = device
        self.transform = transform
        self.classes_ = np.array([0, 1])  # Binary classification
        
    def fit(self, X, y):
        """Dummy fit method for scikit-learn compatibility."""
        # The actual training is done outside this wrapper
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        self.pytorch_model.eval()
        all_probabilities = []
        
        with torch.no_grad():
            for x in X:
                if self.transform and not isinstance(x, torch.Tensor):
                    if isinstance(x, np.ndarray):
                        # Convert numpy array to PIL Image for transforms
                        x = Image.fromarray((x * 255).astype(np.uint8))
                    x = self.transform(x)
                
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                
                x = x.unsqueeze(0).to(self.device)  # Add batch dimension
                outputs = self.pytorch_model(x)
                probabilities = F.softmax(outputs, dim=1)
                all_probabilities.append(probabilities.cpu().numpy()[0])
        
        return np.array(all_probabilities)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# ============================================================================
# DATASET AND DATA UTILITIES
# ============================================================================

class ChestXrayDataset(Dataset):
    """Custom dataset for chest x-ray classification."""
    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        """
        Args:
            root_dir (string): Directory with subdirectories for each class (NORMAL, PNEUMONIA)
            transform (callable, optional): Optional transform to be applied on a sample.
            max_samples_per_class (int, optional): Maximum number of samples per class to load.
                                                  If None, loads all samples.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.max_samples_per_class = max_samples_per_class
        
        # Get class directories
        class_dirs = [d for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))]
        class_dirs.sort()  # Ensure consistent ordering
        self.class_names = class_dirs
        
        print(f"Found {len(class_dirs)} classes: {class_dirs}")
        if max_samples_per_class:
            print(f"Limiting to {max_samples_per_class} samples per class")
        
        # Create mapping from class name to index
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
        
        # Collect all image paths and labels
        for class_name in class_dirs:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files for this class
            class_images = []
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_images.append(os.path.join(class_dir, img_name))
            
            # Limit samples per class if specified
            if max_samples_per_class and len(class_images) > max_samples_per_class:
                # Randomly sample max_samples_per_class images
                import random
                random.seed(42)  # For reproducibility
                class_images = random.sample(class_images, max_samples_per_class)
            
            # Add to dataset
            for img_path in class_images:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
        
        print(f"Total samples: {len(self.image_paths)}")
        for class_name, class_idx in self.class_to_idx.items():
            count = sum(1 for label in self.labels if label == class_idx)
            print(f"  {class_name}: {count} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def test_model(model, test_loader, epoch: int = -1, dataset: str = "", criterion=None, device=None):
    """Evaluate model performance on test set."""
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {dataset}"):
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            
            outputs = model(images)
            if criterion:
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # For binary classification, calculate additional metrics
    if all_probabilities.shape[1] == 2:
        try:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except ValueError:
            auc = None
            
        sensitivity = sensitivity_score(all_labels, all_predictions, average='binary')
        specificity = specificity_score(all_labels, all_predictions, average='binary')
        ppv = precision_score(all_labels, all_predictions, average='binary')
    else:
        auc = None
        sensitivity = sensitivity_score(all_labels, all_predictions, average='macro')
        specificity = specificity_score(all_labels, all_predictions, average='macro')
        ppv = precision_score(all_labels, all_predictions, average='weighted')
    
    confusion_mat = confusion_matrix(all_labels, all_predictions)
    test_loss = test_loss / len(test_loader) if criterion else 0
    
    return auc, sensitivity, specificity, accuracy, f1, ppv, confusion_mat, test_loss

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_resnet50_model(
        root_path: str,
        freeze_features: bool = True,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        load_path: str = "",
        save_checkpoint: bool = True,
        save_interval: int = 5,
        save_path: str = "./checkpoints/resnet50",
        use_mlflow: bool = True,
        experiment_name: str = "ResNet50_ChestXray_FineTuning",
        device=None,
        amp: bool = False,
        num_classes: int = 2,
        max_samples_per_class: int = None
):
    """Main training function for ResNet50 fine-tuning."""
    
    if device is None:
        device = get_device()
    
    set_seed(seed)
    
    # Start MLflow run
    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_param("model_type", "ResNet50_FineTuned")
        mlflow.log_param("freeze_features", freeze_features)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("seed", seed)
        mlflow.log_param("device", device.type)
        mlflow.log_param("amp", amp)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("normalize_mean", normalize_mean)
        mlflow.log_param("normalize_std", normalize_std)
        mlflow.log_param("max_samples_per_class", max_samples_per_class)
    
    # Create model
    model = ResNet50FineTuned(num_classes=num_classes, freeze_features=freeze_features)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.get_num_trainable_params()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    if use_mlflow:
        mlflow.log_param("total_params", total_params)
        mlflow.log_param("trainable_params", trainable_params)
        mlflow.log_param("trainable_percentage", 100 * trainable_params / total_params)
    
    # Create datasets with ImageNet normalization
    normalize = transforms.Normalize(normalize_mean, normalize_std)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Use chest x-ray dataset structure
    train_dir = os.path.join(root_path, "train")
    test_dir = os.path.join(root_path, "test")
    
    if not all(os.path.exists(path) for path in [train_dir, test_dir]):
        raise FileNotFoundError(f"Required data directories not found in {root_path}. "
                                f"Ensure train/ and test/ directories exist.")
    
    dataset_train = ChestXrayDataset(root_dir=train_dir, transform=train_transform, max_samples_per_class=max_samples_per_class)
    dataset_test = ChestXrayDataset(root_dir=test_dir, transform=test_transform, max_samples_per_class=max_samples_per_class)
    
    train_loader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size, shuffle=False, num_workers=0)
    
    # Log dataset information
    if use_mlflow:
        mlflow.log_param("train_size", len(dataset_train))
        mlflow.log_param("test_size", len(dataset_test))
        mlflow.log_param("data_path", root_path)
    
    # Training setup
    info = f'''
        Model: ResNet50 Fine-tuned (ImageNet pretrained)
        Freeze features: {freeze_features}
        Seed: {seed}, Batch size: {batch_size}, Epochs: {epochs}
        Learning rate: {lr}, Data path: {root_path}
        Training size: {len(dataset_train)}, Test size: {len(dataset_test)}
        Device: {device.type}, Save checkpoints: {save_checkpoint}
        Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)
    '''
    print(info)
    
    if load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Loaded checkpoint from {load_path}")
        if use_mlflow:
            mlflow.log_param("loaded_checkpoint", load_path)
    
    model.to(device)
    
    # Only optimize trainable parameters
    optimizer = optim.Adam(model.get_trainable_params(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    # Use device-specific GradScaler to avoid warnings
    if device.type == 'cuda' and amp:
        grad_scaler = torch.amp.GradScaler('cuda', enabled=True)
    elif device.type == 'cpu' and amp:
        # For CPU, disable AMP as it's not supported
        print("Warning: AMP is not supported on CPU, disabling AMP")
        grad_scaler = torch.amp.GradScaler('cpu', enabled=False)
        amp = False
    else:
        grad_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(epochs):
        if optimizer.param_groups[0].get("lr", 0) == 0:
            print("Learning rate reached 0, stopping training")
            break
        
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            labels = labels.to(device=device, dtype=torch.long)
            
            # Use device-appropriate autocast
            if amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            elif amp and device.type == 'cpu':
                # CPU doesn't support autocast with float16, use bfloat16 if available
                with torch.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=torch.cpu.amp.is_autocast_available()):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                # No autocast
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad(set_to_none=True)
            
            if amp:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        mean_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        
        print(f"Epoch {epoch+1}: Loss={mean_loss:.4f}, "
              f"Train Acc={train_accuracy:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Evaluation
        model.eval()
        auc, sensitivity, specificity, test_accuracy, f1, ppv, confusion_mat, test_loss = test_model(
            model, test_loader, epoch, "test", criterion, device)
        
        print(f"Test Results - Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}, "
              f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        
        scheduler.step(test_accuracy)
        
        # Log metrics to MLflow
        if use_mlflow:
            mlflow.log_metric("train_loss", mean_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)
            mlflow.log_metric("test_f1", f1, step=epoch)
            mlflow.log_metric("test_sensitivity", sensitivity, step=epoch)
            mlflow.log_metric("test_specificity", specificity, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)
            if auc is not None:
                mlflow.log_metric("test_auc", auc, step=epoch)
            mlflow.log_metric("test_precision", ppv, step=epoch)
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if save_checkpoint:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                best_model_path = os.path.join(save_path, f"best_resnet50_finetuned.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved with accuracy: {best_accuracy:.4f}')
                
                # Log best model to MLflow
                if use_mlflow:
                    mlflow.log_artifact(best_model_path, "models")
                    mlflow.log_metric("best_accuracy", best_accuracy)
        
        # Save periodic checkpoints
        if save_checkpoint and (epoch + 1) % save_interval == 0:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            checkpoint_path = os.path.join(save_path,
                                           f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                                           f"_epoch{epoch+1}_resnet50_finetuned.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
            
            if use_mlflow:
                mlflow.log_artifact(checkpoint_path, "checkpoints")
    
    # Log final model and end MLflow run
    if use_mlflow:
        # Create sklearn wrapper for the PyTorch model
        sklearn_model = SklearnResNet50Wrapper(
            pytorch_model=model,
            device=device,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
        )
        
        # Get a sample from training data for input_example
        # Convert a few training samples to the format expected by sklearn wrapper
        sample_images = []
        sample_count = 0
        for images, _ in train_loader:
            for img in images:
                if sample_count >= 5:  # Get 5 sample images
                    break
                # Convert tensor back to numpy array (0-1 range)
                img_np = img.permute(1, 2, 0).cpu().numpy()
                # Ensure values are in 0-1 range
                img_np = np.clip(img_np, 0, 1)
                sample_images.append(img_np)
                sample_count += 1
            if sample_count >= 5:
                break
        
        X_train_sample = np.array(sample_images)
        
        mlflow.sklearn.log_model(
            sk_model=sklearn_model,
            artifact_path="sklearn-model",
            input_example=X_train_sample,
            registered_model_name="resnet50-chest-xray-model"
        )
        mlflow.end_run()
    
    return model, best_accuracy

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function demonstrating ResNet50 fine-tuning."""
    
    # Get device and configure AMP accordingly
    device = get_device()
    print(f"Using device: {device}")
    
    # Enable AMP only for CUDA devices
    use_amp = device.type == 'cuda'
    if use_amp:
        print("Mixed precision training enabled (AMP)")
    else:
        print("Mixed precision training disabled (not supported on this device)")
    
    # Configuration for chest x-ray dataset with ResNet50 fine-tuning
    config = {
        "root_path": "C:/Users/sriha/NEU/shlabs/HP_NVIDIA/CellData/chest_xray",  # Updated path for chest x-ray data
        "freeze_features": True,  # Only fine-tune the last layer
        "epochs": 10,  # More epochs for full dataset training
        "batch_size": 32,  # Larger batch size for full dataset
        "lr": 1e-3,  # Higher learning rate for the new classifier layer
        "seed": 42,
        "save_checkpoint": True,
        "save_interval": 5,  # Save every 5 epochs
        "save_path": "./checkpoints/resnet50",
        "use_mlflow": True,  # Use MLflow for experiment tracking
        "experiment_name": "ResNet50_ChestXray_FineTuning_Full",
        "amp": use_amp,  # Use AMP only if CUDA is available
        "device": device,  # Pass device explicitly
        "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "normalize_std": [0.229, 0.224, 0.225],   # ImageNet normalization
        "num_classes": 2,  # Binary classification (NORMAL vs PNEUMONIA)
        "max_samples_per_class": None,  # Use full dataset - no sample limit
    }
    
    # Train model
    print("Starting ResNet50 fine-tuning on chest x-ray dataset...")
    print("Using ImageNet pretrained weights, fine-tuning only the last layer")
    print("Using full dataset for complete training")
    model, best_accuracy = train_resnet50_model(**config)
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    print(f"Model checkpoints saved in: {config['save_path']}")
    print("Check MLflow UI for detailed experiment tracking and metrics visualization.")
    print("Note: This was trained on the full dataset for complete training.")

if __name__ == "__main__":
    import subprocess
    import time

    # Start MLflow UI in the background
    mlflow.end_run()
    mlflow_process = subprocess.Popen([
        'mlflow', 'ui', 
        '--host', '127.0.0.1', 
        '--port', '5000'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("MLflow UI started in background")
    print("Access the dashboard at: http://127.0.0.1:5000")
    print(f"Process ID: {mlflow_process.pid}")
    main() 