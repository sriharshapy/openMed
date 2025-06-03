# Chest X-ray Classification with ResNet50 and Vision Transformer (ViT)

This directory contains implementations for fine-tuning both ResNet50 and Vision Transformer (ViT) models on chest X-ray classification tasks.

## Directory Structure

```
openMed/src/rd/
├── README.md                    # This file
├── resnet50.py                  # ResNet50 fine-tuning script
├── resnet50_gradcam.py          # ResNet50 GradCAM visualization script
├── vit.py                       # ViT fine-tuning script
├── vit_gradcam.py               # ViT attention visualization script
├── checkpoints/                 # Model checkpoints organized by architecture
│   ├── resnet50/               # ResNet50 model checkpoints
│   └── vit/                    # ViT model checkpoints
└── gradcam_results/            # Visualization results organized by architecture
    ├── resnet50/               # ResNet50 GradCAM results
    └── vit/                    # ViT attention map results
```

## Models

### ResNet50 (`resnet50.py`)
- **Architecture**: ResNet50 with ImageNet pretrained weights
- **Fine-tuning Strategy**: Only the final classification layer is trainable
- **Input Size**: 224x224 RGB images
- **Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Checkpoints**: Saved to `./checkpoints/resnet50/`
- **MLflow Port**: 5000 (http://127.0.0.1:5000)

### Vision Transformer (`vit.py`)
- **Architecture**: ViT-B/16 with ImageNet pretrained weights
- **Fine-tuning Strategy**: Only the final classification head is trainable
- **Input Size**: 224x224 RGB images
- **Patch Size**: 16x16 patches
- **Normalization**: ImageNet normalization (same as ResNet50)
- **Checkpoints**: Saved to `./checkpoints/vit/`
- **MLflow Port**: 5001 (http://127.0.0.1:5001)

## Visualization Scripts

### ResNet50 GradCAM (`resnet50_gradcam.py`)
- **Method**: Traditional GradCAM using convolutional layer activations
- **Target Layer**: Last convolutional layer (layer4[-1].conv3)
- **Output**: Heatmaps showing important image regions
- **Results**: Saved to `./gradcam_results/resnet50/`

### ViT Attention Maps (`vit_gradcam.py`)
- **Method**: Attention-based visualization using patch embeddings
- **Target**: Feature magnitudes from patch tokens
- **Output**: Attention maps showing focus areas
- **Results**: Saved to `./gradcam_results/vit/`

## Usage

### Training Models

1. **Train ResNet50**:
   ```bash
   cd openMed/src/rd
   python resnet50.py
   ```

2. **Train ViT**:
   ```bash
   cd openMed/src/rd
   python vit.py
   ```

### Generating Visualizations

1. **ResNet50 GradCAM** (requires trained ResNet50 model):
   ```bash
   python resnet50_gradcam.py
   ```

2. **ViT Attention Maps** (requires trained ViT model):
   ```bash
   python vit_gradcam.py
   ```

## Configuration

Both training scripts use the same configuration structure:

- **Data Path**: `C:/Users/sriha/NEU/shlabs/HP_NVIDIA/CellData/chest_xray`
- **Epochs**: 15
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Classes**: 2 (NORMAL vs PNEUMONIA)
- **AMP**: Enabled on CUDA devices only

## Key Features

### Model Architecture Differences
- **ResNet50**: CNN-based, learns hierarchical features through convolutions
- **ViT**: Transformer-based, learns through self-attention mechanisms on image patches

### Visualization Differences
- **ResNet50 GradCAM**: Shows spatial importance based on convolutional activations
- **ViT Attention**: Shows patch-level importance based on attention mechanisms

### Transfer Learning Strategy
Both models use the same fine-tuning approach:
- Freeze all pretrained parameters
- Only train the final classification layer
- Minimal computational requirements
- Fast convergence

## Output Files

### Model Checkpoints
- **Best Models**: `best_resnet50_finetuned.pth` / `best_vit_finetuned.pth`
- **Periodic Checkpoints**: Timestamped files saved every 5 epochs

### Visualization Results
- **ResNet50**: `gradcam_normal_1.png`, `gradcam_pneumonia_1.png`, etc.
- **ViT**: `vit_attention_normal_1.png`, `vit_attention_pneumonia_1.png`, etc.

## Dependencies

- PyTorch >= 1.9.0
- torchvision
- MLflow
- scikit-learn
- imbalanced-learn
- OpenCV (cv2)
- matplotlib
- Pillow
- numpy
- tqdm

## Monitoring

Both scripts support MLflow experiment tracking:
- **ResNet50**: http://127.0.0.1:5000
- **ViT**: http://127.0.0.1:5001

Metrics tracked include:
- Training/validation loss and accuracy
- F1 score, sensitivity, specificity
- AUC (for binary classification)
- Learning rate schedules
- Model parameters and hyperparameters 