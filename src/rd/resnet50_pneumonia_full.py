#!/usr/bin/env python3
"""
ResNet50 Full Network Training Script for Pneumonia Chest X-ray Classification
This script provides full network training of ResNet50 pretrained on ImageNet,
where ALL layers are trainable (not just the last layer).
Based on the TB full training framework but adapted for pneumonia classification.
Adapted for Pneumonia vs Normal classification.
"""

import os
import random
import shutil
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, confusion_matrix, classification_report
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

class ResNet50FullTraining(nn.Module):
    """ResNet50 with ImageNet pretrained weights, full network training."""
    
    def __init__(self, num_classes=2, freeze_features=False, enable_gradcam=False):
        super(ResNet50FullTraining, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # For full training, we typically keep all layers trainable
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
            print("ALL LAYERS TRAINABLE - Full network training enabled")
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
        if not return_gradcam or not self.enable_gradcam:
            return predictions
        
        # Generate GradCAM
        try:
            heatmap = self.gradcam.generate_cam(x, gradcam_class_idx)
            return predictions, heatmap
        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            return predictions
    
    def enable_gradcam_mode(self):
        """Enable GradCAM functionality."""
        if not self.enable_gradcam:
            target_layer = self.resnet50.layer4[-1].conv3
            self.gradcam = GradCAM(self, target_layer)
            self.enable_gradcam = True
    
    def disable_gradcam_mode(self):
        """Disable GradCAM functionality to save memory."""
        if self.enable_gradcam and self.gradcam:
            self.gradcam.remove_hooks()
            self.gradcam = None
            self.enable_gradcam = False
    
    def get_gradcam_overlay(self, input_image, original_image=None, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Get GradCAM overlay on original image.
        
        Args:
            input_image: Preprocessed input tensor
            original_image: Original image (PIL or numpy array)
            alpha: Transparency factor for overlay
            colormap: OpenCV colormap for heatmap
            
        Returns:
            tuple: (heatmap, overlay) where both are numpy arrays
        """
        if not self.enable_gradcam:
            raise RuntimeError("GradCAM not enabled. Call enable_gradcam_mode() first.")
        
        # Generate heatmap
        heatmap = self.gradcam.generate_cam(input_image)
        
        # Process original image if provided
        if original_image is not None:
            if isinstance(original_image, Image.Image):
                original_np = np.array(original_image.resize((224, 224)))
            else:
                original_np = original_image
            
            # Ensure image is in right format
            if original_np.max() <= 1.0:
                original_np = (original_np * 255).astype(np.uint8)
            
            # Resize heatmap to match image
            heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            if len(original_np.shape) == 2:
                original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
            
            overlay = cv2.addWeighted(original_np, 1-alpha, heatmap_colored, alpha, 0)
            
            return heatmap, overlay
        
        return heatmap, None
    
    def get_trainable_params(self):
        """Get parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SklearnResNet50Wrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for PyTorch ResNet50 model."""
    
    def __init__(self, pytorch_model, device, transform=None):
        self.pytorch_model = pytorch_model
        self.device = device
        self.transform = transform
    
    def fit(self, X, y):
        # This wrapper assumes the model is already trained
        return self
    
    def predict(self, X):
        """Predict class labels."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        self.pytorch_model.eval()
        
        probas = []
        with torch.no_grad():
            for img in X:
                # Ensure image is PIL or tensor
                if isinstance(img, np.ndarray):
                    # Convert numpy to PIL
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                
                # Apply transform
                if self.transform:
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                else:
                    img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
                
                # Get prediction
                output = self.pytorch_model(img_tensor)
                proba = F.softmax(output, dim=1).cpu().numpy()[0]
                probas.append(proba)
        
        return np.array(probas)
    
    def score(self, X, y):
        """Return accuracy score."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# ============================================================================
# DATASET
# ============================================================================

class PneumoniaDataset(Dataset):
    """Dataset class for pneumonia chest X-ray images."""
    
    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        # Get class directories (should be NORMAL and PNEUMONIA)
        class_dirs = [d for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))]
        class_dirs.sort()  # Ensure consistent ordering
        self.class_names = class_dirs
        
        # Create mapping from class name to index
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
        
        print(f"Found classes: {self.class_names}")
        print(f"Class mapping: {self.class_to_idx}")
        
        # Collect all image paths and labels
        for class_name in class_dirs:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            class_images = []
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    class_images.append(os.path.join(class_dir, img_name))
            
            # Limit samples if specified
            if max_samples_per_class and len(class_images) > max_samples_per_class:
                random.seed(42)
                class_images = random.sample(class_images, max_samples_per_class)
            
            for img_path in class_images:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
            
            print(f"Class '{class_name}' ({class_idx}): {len(class_images)} images")
        
        print(f"Total samples: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a dummy image and label
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

def test_model(model, test_loader, epoch: int = -1, dataset: str = "", criterion=None, device=None):
    """Test the model and return comprehensive metrics."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    test_loss = test_loss / len(test_loader) if criterion else 0
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate detailed metrics
    try:
        # For binary classification (Pneumonia vs Normal)
        f1 = f1_score(all_labels, all_predictions, average='binary')
        precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])  # Use positive class probabilities
        except ValueError:
            # If there's an issue with AUC calculation
            auc = None
        
        # Sensitivity and Specificity for binary classification
        sensitivity = sensitivity_score(all_labels, all_predictions, average='binary')
        specificity = specificity_score(all_labels, all_predictions, average='binary')
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Classification Report
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        auc = None
        f1 = accuracy
        precision = accuracy
        sensitivity = accuracy
        specificity = accuracy
        conf_matrix = np.eye(2)
        class_report = {}
    
    return auc, sensitivity, specificity, accuracy, f1, precision, conf_matrix, test_loss, class_report

def train_resnet50_full_model(
    root_path: str,
    freeze_features: bool = False,  # Changed default to False for full training
    normalize_mean: list = [0.485, 0.456, 0.406],
    normalize_std: list = [0.229, 0.224, 0.225],
    epochs: int = 30,  # Increased epochs for full training
    batch_size: int = 16,  # Reduced batch size for full training (more memory intensive)
    lr: float = 1e-4,  # Lower learning rate for full training
    seed: int = 42,
    load_path: str = "",
    save_checkpoint: bool = True,
    save_interval: int = 5,
    save_path: str = "./checkpoints/resnet50_pneumonia_full",
    use_mlflow: bool = True,
    experiment_name: str = "ResNet50_Pneumonia_FullTraining",
    device=None,
    amp: bool = False,
    num_classes: int = 2,
    max_samples_per_class: int = None
):
    """Train ResNet50 model for pneumonia classification with full network training."""
    
    # Set random seed
    set_seed(seed)
    
    # Get device if not provided
    if device is None:
        device = get_device()
    
    print(f"Training ResNet50 for Pneumonia classification - FULL NETWORK TRAINING")
    print(f"Device: {device}")
    print(f"Dataset path: {root_path}")
    print(f"Freeze features: {freeze_features}")
    print(f"AMP enabled: {amp}")
    print(f"Full network training: ALL LAYERS TRAINABLE")
    
    # Initialize MLflow
    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        
        # Log hyperparameters
        mlflow.log_params({
            "model": "ResNet50_FullTraining",
            "freeze_features": freeze_features,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "num_classes": num_classes,
            "max_samples_per_class": max_samples_per_class,
            "normalize_mean": normalize_mean,
            "normalize_std": normalize_std,
            "device": str(device),
            "amp": amp,
            "training_type": "full_network"
        })
    
    # Data transforms with enhanced augmentation for full training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Increased rotation
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Added random crop
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Enhanced augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    # Load datasets
    train_path = os.path.join(root_path, "train")
    test_path = os.path.join(root_path, "test")
    
    print(f"Loading training data from: {train_path}")
    train_dataset = PneumoniaDataset(
        root_dir=train_path,
        transform=train_transform,
        max_samples_per_class=max_samples_per_class
    )
    
    print(f"Loading test data from: {test_path}")
    test_dataset = PneumoniaDataset(
        root_dir=test_path,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.class_names}")
    
    # Initialize model for full training
    model = ResNet50FullTraining(
        num_classes=num_classes,
        freeze_features=freeze_features,  # Should be False for full training
        enable_gradcam=False  # Disable during training for efficiency
    )
    
    # Load pretrained model if specified
    if load_path and os.path.exists(load_path):
        print(f"Loading model from: {load_path}")
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    print(f"Model has {model.get_num_trainable_params():,} trainable parameters")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for different parts of the network
    if not freeze_features:
        # Use different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'fc' in name:  # Final classifier layer
                classifier_params.append(param)
            else:  # Backbone layers
                backbone_params.append(param)
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': lr}       # Higher LR for classifier
        ])
        print(f"Using different learning rates: Backbone {lr * 0.1:.2e}, Classifier {lr:.2e}")
    else:
        optimizer = optim.Adam(model.get_trainable_params(), lr=lr)
    
    # Learning rate scheduler - more aggressive for full training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-7
    )
    
    # Mixed precision training
    grad_scaler = torch.cuda.amp.GradScaler() if amp else None
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass with optional AMP
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
        
        # Get current learning rates
        if not freeze_features and len(optimizer.param_groups) > 1:
            backbone_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
            print(f"Epoch {epoch+1}: Loss={mean_loss:.4f}, Train Acc={train_accuracy:.4f}, "
                  f"Backbone LR={backbone_lr:.2e}, Classifier LR={classifier_lr:.2e}")
        else:
            print(f"Epoch {epoch+1}: Loss={mean_loss:.4f}, Train Acc={train_accuracy:.4f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Evaluation
        model.eval()
        auc, sensitivity, specificity, test_accuracy, f1, ppv, confusion_mat, test_loss, class_report = test_model(
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
            if not freeze_features and len(optimizer.param_groups) > 1:
                mlflow.log_metric("backbone_learning_rate", optimizer.param_groups[0]["lr"], step=epoch)
                mlflow.log_metric("classifier_learning_rate", optimizer.param_groups[1]["lr"], step=epoch)
            else:
                mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)
            if auc is not None:
                mlflow.log_metric("test_auc", auc, step=epoch)
            mlflow.log_metric("test_precision", ppv, step=epoch)
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if save_checkpoint:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                best_model_path = os.path.join(save_path, f"best_resnet50_pneumonia_full_trained.pth")
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
                                           f"_epoch{epoch+1}_resnet50_pneumonia_full_trained.pth")
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
            registered_model_name="resnet50-pneumonia-full-model"
        )
        mlflow.end_run()
    
    return model, best_accuracy

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function demonstrating ResNet50 full network training for pneumonia classification."""
    
    # Get device and configure AMP accordingly
    device = get_device()
    print(f"Using device: {device}")
    
    # Enable AMP only for CUDA devices
    use_amp = device.type == 'cuda'
    if use_amp:
        print("Mixed precision training enabled (AMP)")
    else:
        print("Mixed precision training disabled (not supported on this device)")
    
    # Configuration for pneumonia dataset with ResNet50 full network training
    data_dir = "C:/Users/sriha/NEU/shlabs/HP_NVIDIA/CellData/chest_xray"
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Error: Pneumonia dataset not found at {data_dir}")
        print("Please make sure the pneumonia dataset exists.")
        return
    else:
        print(f"Using pneumonia dataset at: {data_dir}")
    
    config = {
        "root_path": data_dir,
        "freeze_features": False,  # IMPORTANT: Full network training
        "epochs": 30,  # More epochs for full training
        "batch_size": 16,  # Smaller batch size due to more memory usage
        "lr": 1e-4,  # Lower learning rate for full training
        "seed": 42,
        "save_checkpoint": True,
        "save_interval": 5,  # Save every 5 epochs
        "save_path": "./checkpoints/resnet50_pneumonia_full",
        "use_mlflow": True,  # Use MLflow for experiment tracking
        "experiment_name": "ResNet50_Pneumonia_FullTraining",
        "amp": use_amp,  # Use AMP only if CUDA is available
        "device": device,  # Pass device explicitly
        "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "normalize_std": [0.229, 0.224, 0.225],   # ImageNet normalization
        "num_classes": 2,  # Binary classification (Normal vs Pneumonia)
        "max_samples_per_class": None,  # Use full dataset
    }
    
    # Train model
    print("Starting ResNet50 FULL NETWORK training on pneumonia chest X-ray dataset...")
    print("Using ImageNet pretrained weights, training ALL LAYERS")
    print("Classification: Normal vs Pneumonia")
    print("Training approach: Full network training (all parameters trainable)")
    model, best_accuracy = train_resnet50_full_model(**config)
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    print(f"Model checkpoints saved in: {config['save_path']}")
    print("Check MLflow UI for detailed experiment tracking and metrics visualization.")
    print("Note: This was trained with FULL NETWORK TRAINING on the pneumonia dataset.")

if __name__ == "__main__":
    import subprocess
    import time

    # Start MLflow UI in the background
    mlflow.end_run()
    mlflow_process = subprocess.Popen([
        'mlflow', 'ui', 
        '--host', '127.0.0.1', 
        '--port', '5004'  # Different port for pneumonia full training experiment
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("MLflow UI started in background")
    print("Access the dashboard at: http://127.0.0.1:5004")
    print(f"Process ID: {mlflow_process.pid}")
    main()