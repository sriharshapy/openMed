# ResNet50 Tuberculosis Detection System

This directory contains a complete ResNet50-based system for tuberculosis (TB) detection in chest X-rays, adapted from the pneumonia detection system. The system includes training, evaluation, and GradCAM visualization capabilities.

## 📁 Files Created

### 1. `resnet50_tb.py`
- **Purpose**: Main training script for TB classification
- **Features**:
  - Automatic dataset organization (train/test split)
  - ResNet50 fine-tuning with ImageNet pretrained weights
  - MLflow experiment tracking
  - Mixed precision training (AMP) support
  - Comprehensive metrics (accuracy, F1, sensitivity, specificity, AUC)
  - Automatic checkpointing

### 2. `resnet50_gradcam_tb.py`
- **Purpose**: GradCAM visualization for TB model interpretability
- **Features**:
  - Loads trained TB model
  - Generates GradCAM heatmaps
  - Creates overlaid visualizations
  - Saves results with detailed annotations

## 🗂️ Data Structure

### Original Data Location
```
C:\Users\sriha\NEU\shlabs\HP_NVIDIA\TB_Chest_Radiography_Database\
├── Normal/          (3,500+ images)
└── Tuberculosis/    (X images)
```

### Organized Data Structure (Created Automatically)
```
C:\Users\sriha\NEU\shlabs\HP_NVIDIA\TB_Chest_Radiography_Database_Organized\
├── train/
│   ├── Normal/      (80% of normal images)
│   └── Tuberculosis/ (80% of TB images)
└── test/
    ├── Normal/      (20% of normal images)
    └── Tuberculosis/ (20% of TB images)
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install mlflow
pip install scikit-learn
pip install imbalanced-learn
pip install opencv-python
pip install matplotlib
pip install pillow
pip install tqdm
pip install numpy
```

### 1. Training the Model

```bash
cd src/rd
python resnet50_tb.py
```

**What happens during training:**
1. **Data Organization**: If not already organized, the script will automatically split the TB dataset into train/test (80/20 split)
2. **Model Setup**: Loads ResNet50 with ImageNet weights, freezes features, replaces final layer for binary classification
3. **Training**: Uses data augmentation, mixed precision training (if CUDA available), learning rate scheduling
4. **Monitoring**: MLflow tracks all metrics, hyperparameters, and artifacts
5. **Checkpointing**: Saves best model and periodic checkpoints

**Expected Output:**
```
Organizing TB dataset for the first time...
Found 3500 images in Normal class
  Training: 2800 images
  Testing: 700 images
Found XXXX images in Tuberculosis class
  Training: XXXX images
  Testing: XXXX images

Training ResNet50 for TB classification
Device: cuda
ResNet50 loaded with ImageNet weights. Final layer: 2048 -> 2
All layers frozen except final classifier layer
Model has 2,049 trainable parameters

Epoch 1: Loss=0.6234, Train Acc=0.7123, LR=0.001000
Test Results - Accuracy: 0.8456, F1: 0.8234, Sensitivity: 0.8123, Specificity: 0.8789
...
```

### 2. GradCAM Visualization

**After training is complete**, run:

```bash
python resnet50_gradcam_tb.py
```

**What this does:**
1. Loads the trained model from `./checkpoints/resnet50_tb/best_resnet50_tb_finetuned.pth`
2. Loads test images from the organized dataset
3. Generates GradCAM heatmaps showing what the model focuses on
4. Creates visualizations with original image, heatmap, and overlay
5. Saves results to `./gradcam_results/resnet50_tb/`

## 📊 Model Architecture

### ResNet50 Fine-tuning Strategy
- **Base Model**: ResNet50 pretrained on ImageNet
- **Frozen Layers**: All convolutional layers (feature extractor)
- **Trainable**: Only the final fully connected layer (2048 → 2)
- **Benefits**: 
  - Fast training (~2K parameters vs 25M)
  - Good generalization with limited data
  - Leverages powerful pretrained features

### Training Configuration
```python
config = {
    "epochs": 15,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau",
    "augmentation": ["HorizontalFlip", "Rotation", "ColorJitter"],
    "normalization": "ImageNet_stats",
    "train_split": 0.8
}
```

## 📈 Monitoring and Results

### MLflow Tracking
- **URL**: http://127.0.0.1:5001
- **Tracks**: Loss, accuracy, F1, sensitivity, specificity, AUC, learning rate
- **Artifacts**: Model checkpoints, final model, sample predictions

### Model Outputs
```
./checkpoints/resnet50_tb/
├── best_resnet50_tb_finetuned.pth  # Best model by test accuracy
├── YYYYMMDD-HHMMSS_epoch5_resnet50_tb_finetuned.pth  # Periodic checkpoints
└── YYYYMMDD-HHMMSS_epoch10_resnet50_tb_finetuned.pth
```

### GradCAM Results
```
./gradcam_results/resnet50_tb/
├── gradcam_normal_1.png
├── gradcam_normal_2.png
├── gradcam_tuberculosis_1.png
├── gradcam_tuberculosis_2.png
└── ...
```

## 🔍 Key Differences from Pneumonia Version

### Data Handling
- **Classes**: Normal vs Tuberculosis (instead of Normal vs Pneumonia)
- **Organization**: Automatic dataset reorganization from flat structure
- **Paths**: Updated to TB dataset location

### Model Configuration
- **Checkpoint Path**: `./checkpoints/resnet50_tb/`
- **Model Name**: `best_resnet50_tb_finetuned.pth`
- **MLflow Experiment**: `ResNet50_TB_ChestXray_FineTuning`
- **Port**: MLflow UI on port 5001 (to avoid conflicts)

### Dataset Class
- **Class Name**: `TBChestXrayDataset` (instead of `ChestXrayDataset`)
- **Extensions**: Support for `.bmp`, `.tiff`, `.tif` in addition to standard formats
- **Error Handling**: Robust error handling for corrupt images

## 🛠️ Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   Error: C:\Users\sriha\NEU\shlabs\HP_NVIDIA\TB_Chest_Radiography_Database does not exist
   ```
   **Solution**: Verify the data path in the script matches your actual data location

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch_size in the config (e.g., from 32 to 16 or 8)

3. **Model Not Found for GradCAM**
   ```
   FileNotFoundError: Model checkpoint not found
   ```
   **Solution**: Ensure you've trained the model first using `resnet50_tb.py`

### Performance Tips

1. **GPU Memory**: If you have limited GPU memory, reduce batch size
2. **Training Speed**: Enable AMP (automatic mixed precision) for faster training on modern GPUs
3. **Data Loading**: Adjust `num_workers` in DataLoader based on your CPU cores

## 📋 Expected Performance

### Baseline Expectations
- **Training Time**: ~15-30 minutes for 15 epochs (with GPU)
- **Memory Usage**: ~4-6GB GPU memory with batch_size=32
- **Accuracy**: Should achieve >85% test accuracy with proper data quality

### Metrics to Monitor
- **Sensitivity**: Ability to detect TB cases (minimize false negatives)
- **Specificity**: Ability to identify normal cases (minimize false positives)
- **AUC**: Overall discriminative ability
- **F1-Score**: Balanced precision and recall

## 🔗 Integration with Existing System

This TB detection system is designed to work alongside the existing pneumonia detection system:

- **Shared Infrastructure**: Uses same MLflow setup, similar model architecture
- **Independent Models**: TB and pneumonia models are separate and can run independently
- **Common Utilities**: Shares visualization and evaluation tools
- **Scalable Design**: Easy to add more diseases following the same pattern

## 📝 Next Steps

1. **Train the Model**: Run `python resnet50_tb.py`
2. **Evaluate Results**: Check MLflow UI for training metrics
3. **Visualize Predictions**: Run `python resnet50_gradcam_tb.py`
4. **Analyze GradCAM**: Review the attention maps to understand model behavior
5. **Fine-tune if Needed**: Adjust hyperparameters based on results

## 🎯 Key Features

- ✅ **Automatic Data Organization**: No manual file moving required
- ✅ **Transfer Learning**: Efficient training with ImageNet pretrained weights
- ✅ **Experiment Tracking**: Complete MLflow integration
- ✅ **Visualization**: GradCAM for model interpretability
- ✅ **Robust Evaluation**: Multiple metrics for thorough assessment
- ✅ **Production Ready**: Checkpointing, error handling, logging

For questions or issues, refer to the existing pneumonia detection system documentation or the original ResNet50 implementation. 