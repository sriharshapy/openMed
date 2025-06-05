# ResNet50 with Integrated GradCAM

This document describes the enhanced ResNet50 model with integrated GradCAM (Gradient-weighted Class Activation Mapping) functionality for chest X-ray classification.

## Overview

The original `ResNet50FineTuned` class has been enhanced to include built-in GradCAM functionality, allowing you to:
- Generate predictions as before
- Optionally generate GradCAM heatmaps along with predictions
- Create overlaid visualizations
- Enable/disable GradCAM dynamically

## Key Changes

### 1. Enhanced Model Constructor

```python
ResNet50FineTuned(num_classes=2, freeze_features=True, enable_gradcam=False)
```

**New parameter:**
- `enable_gradcam`: Boolean flag to enable GradCAM functionality during model initialization

### 2. Enhanced Forward Method

```python
def forward(self, x, return_gradcam=False, gradcam_class_idx=None):
```

**Parameters:**
- `x`: Input tensor (as before)
- `return_gradcam`: Whether to return GradCAM heatmap along with predictions
- `gradcam_class_idx`: Specific class index for GradCAM (if None, uses predicted class)

**Returns:**
- If `return_gradcam=False`: predictions tensor (same as before)
- If `return_gradcam=True`: tuple of (predictions, gradcam_heatmap)

### 3. New Methods

#### `enable_gradcam_mode()`
Enables GradCAM functionality if not already enabled during initialization.

#### `disable_gradcam_mode()`
Disables GradCAM functionality and removes hooks to free memory.

#### `get_gradcam_overlay(input_image, original_image=None, alpha=0.5, colormap=cv2.COLORMAP_JET)`
Generates GradCAM overlay on the original image.

**Parameters:**
- `input_image`: Preprocessed input tensor for the model
- `original_image`: Original image (PIL Image or numpy array) for overlay
- `alpha`: Transparency factor for the heatmap overlay (0-1)
- `colormap`: OpenCV colormap for the heatmap

**Returns:**
- `tuple`: (colored_heatmap, overlaid_image) or None if GradCAM not enabled

## Usage Examples

### Basic Usage (Backward Compatible)

```python
from resnet50 import ResNet50FineTuned

# Create model without GradCAM (same as before)
model = ResNet50FineTuned(num_classes=2, freeze_features=True)

# Standard prediction (unchanged)
predictions = model(input_tensor)
```

### Using GradCAM

```python
# Create model with GradCAM enabled
model = ResNet50FineTuned(num_classes=2, freeze_features=True, enable_gradcam=True)

# Get predictions only
predictions = model(input_tensor)

# Get predictions and GradCAM heatmap
predictions, gradcam_heatmap = model(input_tensor, return_gradcam=True)

# Get GradCAM for a specific class
predictions, gradcam_heatmap = model(input_tensor, return_gradcam=True, gradcam_class_idx=1)
```

### Dynamic GradCAM Control

```python
# Create model without GradCAM initially
model = ResNet50FineTuned(num_classes=2, freeze_features=True, enable_gradcam=False)

# Enable GradCAM later
model.enable_gradcam_mode()

# Use GradCAM
predictions, gradcam_heatmap = model(input_tensor, return_gradcam=True)

# Disable GradCAM to save memory
model.disable_gradcam_mode()
```

### Visualization

```python
# Get GradCAM overlay
colored_heatmap, overlay = model.get_gradcam_overlay(input_tensor, original_image)

# Display results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title('Original')
axes[1].imshow(colored_heatmap)
axes[1].set_title('GradCAM')
axes[2].imshow(overlay)
axes[2].set_title('Overlay')
plt.show()
```

## Demo Script

Run the provided demo script to see the GradCAM functionality in action:

```bash
python resnet50_demo.py
```

The demo script demonstrates:
1. Basic model usage without GradCAM
2. Model usage with GradCAM
3. GradCAM overlay generation
4. Dynamic enable/disable functionality

## Technical Details

### GradCAM Implementation

- **Target Layer**: Uses the last convolutional layer (`resnet50.layer4[-1].conv3`) for maximum spatial resolution
- **Gradient Handling**: Automatically manages gradient computation and model modes
- **Hook Management**: Properly registers and removes forward/backward hooks
- **Memory Efficiency**: Only computes gradients when GradCAM is requested

### Compatibility

- **Backward Compatible**: Existing code using the model will work unchanged
- **Training Compatible**: GradCAM hooks don't interfere with training
- **Device Agnostic**: Works with CPU, CUDA, and MPS devices

### Performance Considerations

- **Overhead**: GradCAM adds minimal overhead when disabled
- **Memory**: GradCAM requires additional memory for gradient storage
- **Speed**: GradCAM generation requires a backward pass, adding computation time

## Dependencies

The enhanced model requires an additional dependency:
- `opencv-python` (cv2) for heatmap visualization

Install with:
```bash
pip install opencv-python
```

## Files Modified/Added

1. **resnet50.py** - Enhanced with GradCAM functionality
2. **resnet50_demo.py** - New demo script
3. **README_GRADCAM.md** - This documentation

## Migration Guide

### From Original ResNet50

If you have existing code using the original `ResNet50FineTuned`:

1. **No changes required** for basic functionality
2. **Add `enable_gradcam=True`** to constructor to enable GradCAM
3. **Use `return_gradcam=True`** in forward() to get heatmaps

### From Separate GradCAM Script

If you were using the separate `resnet50_gradcam.py`:

1. **Remove separate GradCAM imports**
2. **Use the integrated functionality** in the main model
3. **Update visualization code** to use `get_gradcam_overlay()`

## Troubleshooting

### Common Issues

1. **"GradCAM not enabled" error**
   - Solution: Set `enable_gradcam=True` or call `enable_gradcam_mode()`

2. **Memory issues during GradCAM**
   - Solution: Disable GradCAM when not needed with `disable_gradcam_mode()`

3. **Hook registration errors**
   - Solution: Ensure proper model initialization and avoid multiple GradCAM instances

### Best Practices

1. **Enable GradCAM only when needed** to save memory
2. **Disable GradCAM during training** for better performance
3. **Use appropriate batch sizes** when generating multiple GradCAMs
4. **Clean up hooks** by calling `disable_gradcam_mode()` when done

## Future Enhancements

Potential improvements for future versions:
- Multiple target layer support
- Batch GradCAM processing
- Different CAM variants (GradCAM++, LayerCAM, etc.)
- Integration with other model architectures 