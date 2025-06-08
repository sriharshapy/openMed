# ResNet50 TB Detection System - Full Network Training

This directory contains ResNet50-based systems for tuberculosis (TB) detection with **full network training** capability. This system trains ALL layers of the ResNet50 network rather than just fine-tuning the last layer, providing a comparison between transfer learning and full network training approaches.

## 📁 Files for TB Full Network Training

### 1. `resnet50_tb_full.py`
- **Purpose**: Full network training script for TB classification
- **Key Difference**: ALL ResNet50 layers are trainable (not just the last layer)
- **Features**:
  - Full network training with ImageNet pretrained initialization
  - Differential learning rates (lower for backbone, higher for classifier)
  - Enhanced data augmentation for full training
  - Extended training (30 epochs vs 10-20 for fine-tuning)
  - MLflow experiment tracking with separate port (5003)
  - Memory-optimized batch size (16 vs 32 for fine-tuning)

### 2. `resnet50_gradcam_tb_full.py`
- **Purpose**: GradCAM visualization for full network trained TB model
- **Features**:
  - Loads full network trained model
  - Generates attention maps for TB vs Normal classification
  - Comparison-ready visualization with "full_training" labels
  - Enhanced analysis showing differences from transfer learning

## 🔄 Full Training vs Transfer Learning Comparison

| Aspect | Transfer Learning (`resnet50_tb.py`) | Full Network Training (`resnet50_tb_full.py`) |
|--------|-------------------------------------|----------------------------------------------|
| **Trainable Parameters** | ~3,073 (only final layer) | ~25,557,032 (all layers) |
| **Training Time** | 15-30 minutes | 45-90 minutes |
| **Learning Rate** | 1e-3 (single rate) | 1e-5 backbone, 1e-4 classifier |
| **Batch Size** | 32 | 16 (memory constraints) |
| **Epochs** | 10-20 | 30 |
| **Memory Usage** | ~4GB VRAM | ~6-8GB VRAM |
| **Data Augmentation** | Standard | Enhanced |
| **MLflow Port** | 5001 | 5003 |
| **Model File** | `best_resnet50_tb_finetuned.pth` | `best_resnet50_tb_full_trained.pth` |

## 🗂️ Data Structure

### Data Location (Already Organized)
```
C:\Users\sriha\NEU\shlabs\HP_NVIDIA\TB_Chest_Radiography_Database_Organized\
├── train/
│   ├── Normal/           (Normal chest X-rays for training)
│   └── Tuberculosis/     (TB chest X-rays for training)
└── test/
    ├── Normal/           (Normal chest X-rays for testing)
    └── Tuberculosis/     (TB chest X-rays for testing)
```

The full training system uses the **already organized** TB dataset (no data reorganization needed).

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

### 1. Full Network Training

```bash
cd src/rd
python resnet50_tb_full.py
```

**What happens during full training:**
1. **Model Setup**: Loads ResNet50, keeps ALL layers trainable (freeze_features=False)
2. **Differential Learning**: Uses different learning rates for backbone vs classifier
3. **Enhanced Augmentation**: More aggressive data augmentation for robust training
4. **Extended Training**: 30 epochs with patience-based learning rate scheduling
5. **Memory Management**: Smaller batch size to accommodate full network gradients
6. **Monitoring**: MLflow tracks all metrics on port 5003

**Expected Output:**
```
Training ResNet50 for TB classification - FULL NETWORK TRAINING
Device: cuda
ALL LAYERS TRAINABLE - Full network training enabled
Model has 25,557,032 trainable parameters
Using different learning rates: Backbone 1.00e-05, Classifier 1.00e-04

Epoch 1: Loss=0.6234, Train Acc=0.6823, Backbone LR=1.00e-05, Classifier LR=1.00e-04
Test Results - Accuracy: 0.7856, F1: 0.7634, Sensitivity: 0.7523, Specificity: 0.8189
...
```

### 2. GradCAM Visualization

**After full training is complete**, run:

```bash
python resnet50_gradcam_tb_full.py
```

**What this does:**
1. Loads the full network trained model from `./checkpoints/resnet50_tb_full/best_resnet50_tb_full_trained.pth`
2. Loads test images from the organized dataset
3. Generates GradCAM heatmaps showing attention patterns
4. Creates visualizations labeled as "full_training" for comparison
5. Saves results to `./gradcam_results/resnet50_tb_full/`

## 📊 Model Architecture Differences

### Full Network Training Model
```python
class ResNet50FullTraining(nn.Module):
    def __init__(self, num_classes=2, freeze_features=False):  # False for full training
        # ALL layers trainable
        # No parameter freezing
        # Differential learning rate optimization
```

### Training Strategy
```python
# Differential learning rates
optimizer = optim.Adam([
    {'params': backbone_params, 'lr': lr * 0.1},    # Lower LR for backbone
    {'params': classifier_params, 'lr': lr}         # Higher LR for classifier
])

# Enhanced data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Added for full training
    transforms.RandomRotation(degrees=15),                 # Increased from 10
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    # ... other transforms
])
```

## 📈 Expected Performance Differences

### Performance Expectations

| Metric | Transfer Learning | Full Network Training | Notes |
|--------|------------------|----------------------|-------|
| **Accuracy** | 85-90% | 88-93% | Full training may achieve higher accuracy |
| **Training Time** | 15-30 min | 45-90 min | Significantly longer due to all layer updates |
| **Convergence** | Faster (5-10 epochs) | Slower (20-30 epochs) | More parameters to optimize |
| **Overfitting Risk** | Lower | Higher | Requires more data and regularization |
| **Generalization** | Good | Variable | Depends on dataset size and quality |

### When to Use Each Approach

#### Transfer Learning (`resnet50_tb.py`) - Recommended for:
- **Limited computational resources**
- **Small datasets** (< 10K images)
- **Quick prototyping**
- **Stable, proven results**
- **Limited time for experimentation**

#### Full Network Training (`resnet50_tb_full.py`) - Consider for:
- **Large datasets** (> 50K images)
- **Domain-specific requirements** (medical imaging specificity)
- **Maximum performance goals**
- **Research and experimentation**
- **Sufficient computational resources**

## 🔍 Results Analysis

### MLflow Monitoring

#### Transfer Learning Dashboard
- **URL**: http://127.0.0.1:5001
- **Experiment**: `ResNet50_TB_FineTuning`
- **Key Metrics**: Fast convergence, stable training

#### Full Network Training Dashboard
- **URL**: http://127.0.0.1:5003
- **Experiment**: `ResNet50_TB_FullTraining`
- **Key Metrics**: Extended learning curves, detailed LR tracking

### Model Outputs

#### Transfer Learning Checkpoints
```
./checkpoints/resnet50_tb/
├── best_resnet50_tb_finetuned.pth
└── YYYYMMDD-HHMMSS_epoch5_resnet50_tb_finetuned.pth
```

#### Full Training Checkpoints
```
./checkpoints/resnet50_tb_full/
├── best_resnet50_tb_full_trained.pth
└── YYYYMMDD-HHMMSS_epoch5_resnet50_tb_full_trained.pth
```

### GradCAM Comparison

#### Transfer Learning GradCAM
```
./gradcam_results/resnet50_tb/
├── gradcam_normal_1.png
├── gradcam_tuberculosis_1.png
└── ...
```

#### Full Training GradCAM
```
./gradcam_results/resnet50_tb_full/
├── gradcam_normal_1_full_training.png
├── gradcam_tuberculosis_1_full_training.png
└── ...
```

## 🛠️ Advanced Configuration

### Memory Optimization

```python
# For limited GPU memory, reduce batch size further
config = {
    "batch_size": 8,  # Reduce from 16
    "amp": True,      # Enable automatic mixed precision
}
```

### Learning Rate Tuning

```python
# For different learning rate strategies
optimizer = optim.Adam([
    {'params': backbone_params, 'lr': 5e-6},    # Even lower for backbone
    {'params': classifier_params, 'lr': 5e-4}   # Higher for classifier
])
```

### Extended Training

```python
# For longer training with more patience
config = {
    "epochs": 50,
    "scheduler_patience": 5,  # More patience before LR reduction
}
```

## 📊 Comparison Workflow

### 1. Train Both Models
```bash
# Train transfer learning model first
python resnet50_tb.py

# Then train full network model
python resnet50_tb_full.py
```

### 2. Compare Results
```bash
# Generate GradCAM for both
python resnet50_gradcam_tb.py        # Transfer learning
python resnet50_gradcam_tb_full.py   # Full training
```

### 3. Analyze Performance
- Compare MLflow dashboards (ports 5001 vs 5003)
- Examine GradCAM attention differences
- Evaluate accuracy, training time, and resource usage

## 🎯 Key Insights

### Transfer Learning Advantages
- ✅ **Faster Training**: Quick convergence with fewer epochs
- ✅ **Lower Resource Requirements**: Less memory and computation
- ✅ **Stable Results**: Proven performance with medical imaging
- ✅ **Less Overfitting**: Pretrained features provide good regularization

### Full Network Training Advantages
- ✅ **Potentially Higher Accuracy**: Can achieve better performance with sufficient data
- ✅ **Task-Specific Features**: Learns features specifically for TB detection
- ✅ **Research Value**: Better for understanding model behavior
- ✅ **Customization**: Full control over all network parameters

### Trade-offs Summary

| Factor | Transfer Learning Wins | Full Training Wins |
|--------|----------------------|-------------------|
| **Speed** | ✓ | |
| **Resource Efficiency** | ✓ | |
| **Simplicity** | ✓ | |
| **Max Performance** | | ✓ |
| **Task Specificity** | | ✓ |
| **Research Depth** | | ✓ |

## 🔬 Medical Validation Considerations

### Transfer Learning Model
- **Validation Focus**: Consistency with known medical imaging patterns
- **Interpretability**: Leverages proven ImageNet features for medical domain
- **Clinical Adoption**: Easier to validate and deploy

### Full Network Training Model
- **Validation Focus**: Medical-specific feature learning
- **Interpretability**: May show novel attention patterns specific to TB
- **Clinical Adoption**: Requires more extensive validation

## 🚨 Important Notes

### Resource Requirements
- **Full Training**: Requires 6-8GB GPU memory minimum
- **Transfer Learning**: Works with 4GB GPU memory
- **CPU Fallback**: Both support CPU training (much slower)

### Data Considerations
- **Minimum Data**: Transfer learning works with smaller datasets
- **Optimal Data**: Full training benefits from larger, diverse datasets
- **Quality**: Both approaches require high-quality, well-labeled data

### Production Deployment
- **Transfer Learning**: Generally recommended for production
- **Full Training**: Use only if demonstrated superior performance
- **Validation**: Both require clinical validation before deployment

## 📝 Next Steps

1. **Compare Both Approaches**: Train both models and compare results
2. **Analyze GradCAM**: Study attention pattern differences
3. **Performance Evaluation**: Use MLflow to compare metrics
4. **Clinical Testing**: Validate with medical experts
5. **Production Choice**: Select best approach for deployment

## ⚠️ Medical Disclaimer

Both systems are for research and educational purposes only. Full network training does not guarantee better clinical performance and may require more extensive validation. Always consult with qualified medical professionals for clinical applications.

## 🔗 Related Documentation

- `README_TB.md`: Transfer learning TB system documentation
- `README_COMPLETE_SYSTEM.md`: Complete medical imaging system overview
- `README_GRADCAM.md`: GradCAM implementation details

---

**Full Network Training: When you need maximum performance and have the resources to achieve it.** 