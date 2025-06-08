# ResNet50 Brain Tumor Classification System

This directory contains a complete ResNet50-based system for brain tumor classification in MRI scans, adapted from the pneumonia and TB detection systems. The system supports 3-class classification: Glioma, Meningioma, and Tumor, including training, evaluation, and GradCAM visualization capabilities.

## 📁 Files Created

### 1. `resnet50_brain_tumor.py`
- **Purpose**: Main training script for brain tumor classification
- **Features**:
  - Automatic dataset organization (train/test split)
  - ResNet50 fine-tuning with ImageNet pretrained weights
  - 3-class classification support
  - MLflow experiment tracking
  - Mixed precision training (AMP) support
  - Comprehensive multi-class metrics (accuracy, F1, sensitivity, specificity, AUC)
  - Automatic checkpointing
  - Enhanced metrics for multi-class evaluation

### 2. `resnet50_gradcam_brain_tumor.py`
- **Purpose**: GradCAM visualization for brain tumor model interpretability
- **Features**:
  - Loads trained brain tumor model
  - Generates GradCAM heatmaps for 3-class predictions
  - Creates advanced visualizations with probability distributions
  - Enhanced multi-class visualization layout
  - Saves results with detailed annotations

## 🗂️ Data Structure

### Original Data Location
```
C:\Users\sriha\NEU\shlabs\HP_NVIDIA\Brain_Cancer raw MRI data\Brain_Cancer\
├── brain_glioma/     (Glioma MRI images)
├── brain_menin/      (Meningioma MRI images)
└── brain_tumor/      (General tumor MRI images)
```

### Organized Data Structure (Created Automatically)
```
C:\Users\sriha\NEU\shlabs\HP_NVIDIA\Brain_Cancer_Organized\
├── train/
│   ├── brain_glioma/     (80% of glioma images)
│   ├── brain_menin/      (80% of meningioma images)
│   └── brain_tumor/      (80% of tumor images)
└── test/
    ├── brain_glioma/     (20% of glioma images)
    ├── brain_menin/      (20% of meningioma images)
    └── brain_tumor/      (20% of tumor images)
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
python resnet50_brain_tumor.py
```

**What happens during training:**
1. **Data Organization**: If not already organized, the script will automatically split the brain tumor dataset into train/test (80/20 split)
2. **Model Setup**: Loads ResNet50 with ImageNet weights, freezes features, replaces final layer for 3-class classification
3. **Training**: Uses data augmentation, mixed precision training (if CUDA available), learning rate scheduling
4. **Monitoring**: MLflow tracks all metrics, hyperparameters, and artifacts
5. **Checkpointing**: Saves best model and periodic checkpoints

**Expected Output:**
```
Organizing Brain Tumor dataset for the first time...
Found XXXX images in brain_glioma class
  Training: XXXX images
  Testing: XXXX images
Found XXXX images in brain_menin class
  Training: XXXX images
  Testing: XXXX images
Found XXXX images in brain_tumor class
  Training: XXXX images
  Testing: XXXX images

Training ResNet50 for Brain Tumor classification
Device: cuda
ResNet50 loaded with ImageNet weights. Final layer: 2048 -> 3
All layers frozen except final classifier layer
Model has 3,073 trainable parameters

Epoch 1: Loss=0.8234, Train Acc=0.6123, LR=0.001000
Test Results - Accuracy: 0.7456, F1: 0.7234, Sensitivity: 0.7123, Specificity: 0.8789
...
```

### 2. GradCAM Visualization

**After training is complete**, run:

```bash
python resnet50_gradcam_brain_tumor.py
```

**What this does:**
1. Loads the trained model from `./checkpoints/resnet50_brain_tumor/best_resnet50_brain_tumor_finetuned.pth`
2. Loads test images from the organized dataset
3. Generates GradCAM heatmaps showing what the model focuses on
4. Creates advanced visualizations with 4-panel layout:
   - Original MRI image
   - GradCAM heatmap
   - Overlaid visualization
   - Class probability distribution
5. Saves results to `./gradcam_results/resnet50_brain_tumor/`

## 📊 Model Architecture

### ResNet50 Fine-tuning Strategy
- **Base Model**: ResNet50 pretrained on ImageNet
- **Frozen Layers**: All convolutional layers (feature extractor)
- **Trainable**: Only the final fully connected layer (2048 → 3)
- **Benefits**: 
  - Fast training (~3K parameters vs 25M)
  - Good generalization with limited medical data
  - Leverages powerful pretrained features

### Training Configuration
```python
config = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau",
    "augmentation": ["HorizontalFlip", "Rotation", "ColorJitter"],
    "normalization": "ImageNet_stats",
    "train_split": 0.8,
    "num_classes": 3  # Glioma, Meningioma, Tumor
}
```

## 📈 Monitoring and Results

### MLflow Tracking
- **URL**: http://127.0.0.1:5002
- **Tracks**: Loss, accuracy, F1, sensitivity, specificity, AUC, learning rate
- **Artifacts**: Model checkpoints, final model, sample predictions
- **Multi-class Metrics**: Weighted averages and per-class performance

### Model Outputs
```
./checkpoints/resnet50_brain_tumor/
├── best_resnet50_brain_tumor_finetuned.pth  # Best model by test accuracy
├── YYYYMMDD-HHMMSS_epoch5_resnet50_brain_tumor_finetuned.pth  # Periodic checkpoints
└── YYYYMMDD-HHMMSS_epoch10_resnet50_brain_tumor_finetuned.pth
```

### GradCAM Results
```
./gradcam_results/resnet50_brain_tumor/
├── gradcam_glioma_1.png
├── gradcam_glioma_2.png
├── gradcam_menin_1.png
├── gradcam_menin_2.png
├── gradcam_tumor_1.png
├── gradcam_tumor_2.png
└── ...
```

## 🔍 Key Differences from Binary Classification

### Multi-class Considerations
- **Classes**: 3-class classification (vs 2-class for TB/Pneumonia)
- **Metrics**: Multi-class AUC using one-vs-rest approach
- **Visualization**: Enhanced 4-panel layout with probability distributions
- **Complexity**: More challenging classification task

### Enhanced Features
- **Advanced GradCAM**: 4-panel visualization showing original image, heatmap, overlay, and probability distribution
- **Multi-class Metrics**: Weighted F1, precision, recall, and per-class performance
- **Probability Analysis**: Bar charts showing confidence across all 3 classes
- **Detailed Logging**: Classification reports and confusion matrices

### Dataset Class
- **Class Name**: `BrainTumorDataset` (instead of binary datasets)
- **Classes**: brain_glioma, brain_menin, brain_tumor
- **Extensions**: Support for multiple medical image formats
- **Error Handling**: Robust error handling for medical imaging data

## 🛠️ Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   Error: C:\Users\sriha\NEU\shlabs\HP_NVIDIA\Brain_Cancer raw MRI data\Brain_Cancer does not exist
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
   **Solution**: Ensure you've trained the model first using `resnet50_brain_tumor.py`

4. **Imbalanced Classes**
   ```
   Warning: Class imbalance detected
   ```
   **Solution**: Consider using class weights or additional data augmentation

### Performance Tips

1. **GPU Memory**: If you have limited GPU memory, reduce batch size
2. **Training Speed**: Enable AMP (automatic mixed precision) for faster training on modern GPUs
3. **Data Loading**: Adjust `num_workers` in DataLoader based on your CPU cores
4. **Class Balance**: Monitor per-class performance in MLflow

## 📋 Expected Performance

### Baseline Expectations
- **Training Time**: ~20-45 minutes for 20 epochs (with GPU)
- **Memory Usage**: ~4-6GB GPU memory with batch_size=32
- **Accuracy**: Should achieve >70% test accuracy with proper data quality (3-class is more challenging)

### Metrics to Monitor
- **Per-class Accuracy**: Monitor individual class performance
- **Weighted F1**: Balanced performance across classes
- **Confusion Matrix**: Identify class confusion patterns
- **AUC (One-vs-Rest)**: Overall discriminative ability across classes

## 🎯 Key Features

- ✅ **3-Class Classification**: Full support for multi-class brain tumor detection
- ✅ **Automatic Data Organization**: No manual file moving required
- ✅ **Transfer Learning**: Efficient training with ImageNet pretrained weights
- ✅ **Experiment Tracking**: Complete MLflow integration with multi-class metrics
- ✅ **Advanced Visualization**: Enhanced GradCAM with probability distributions
- ✅ **Robust Evaluation**: Multiple metrics for thorough assessment
- ✅ **Production Ready**: Checkpointing, error handling, logging
- ✅ **Medical Imaging Optimized**: Designed for MRI scan analysis

## 🔬 Medical Imaging Considerations

### Brain Tumor Types
- **Glioma**: Tumors that arise from glial cells
- **Meningioma**: Tumors that develop from the meninges
- **General Tumor**: Other brain tumor types

### Clinical Relevance
- **Early Detection**: Model helps identify tumor presence and type
- **Treatment Planning**: Classification aids in selecting appropriate treatment
- **Monitoring**: Can be used to track tumor progression
- **Second Opinion**: Assists radiologists in diagnosis

### Validation Recommendations
- **Cross-validation**: Consider k-fold validation for robust evaluation
- **External Validation**: Test on data from different hospitals/scanners
- **Clinical Validation**: Validate with expert radiologist annotations
- **Bias Assessment**: Check for demographic and technical biases

## 📝 Next Steps

1. **Train the Model**: Run `python resnet50_brain_tumor.py`
2. **Evaluate Results**: Check MLflow UI for training metrics and confusion matrices
3. **Visualize Predictions**: Run `python resnet50_gradcam_brain_tumor.py`
4. **Analyze GradCAM**: Review the attention maps to understand model behavior
5. **Clinical Validation**: Work with medical experts to validate results
6. **Fine-tune if Needed**: Adjust hyperparameters based on results

## 🔗 Integration with Existing System

This brain tumor detection system is designed to work alongside the existing medical imaging systems:

- **Shared Infrastructure**: Uses same MLflow setup, similar model architecture
- **Independent Models**: Brain tumor, TB, and pneumonia models are separate
- **Common Utilities**: Shares visualization and evaluation tools
- **Scalable Design**: Easy to add more medical imaging tasks

## ⚠️ Important Medical Disclaimer

This system is for research and educational purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult with qualified medical professionals for actual medical diagnosis and treatment decisions.

For questions or issues, refer to the existing medical imaging system documentation or the original ResNet50 implementation. 