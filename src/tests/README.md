# OpenMed Tests

This directory contains test cases for the OpenMed project, focusing on model inference and validation.

## ResNet50 Models and Utils Test (`test_resnet50_inference.py`)

### Overview

This test case specifically tests the modular **ResNet50Model** from `openMed/src/models/` and **GradCAM utilities** from `openMed/src/utils/`. It loads ResNet50 weights from `openMed/weights/resnet50/` and performs inference on 5 normal and 5 pneumonia chest X-ray samples, testing the new object-oriented architecture.

### What This Tests

**Models (`openMed/src/models/`):**
- `ResNet50Model` class and its methods
- `model_factory.create_resnet50()` function
- Model loading via `from_pretrained_checkpoint()`
- Prediction methods: `predict()`, `predict_proba()`
- Feature extraction via `extract_features()`
- Model component access and introspection

**Utils (`openMed/src/utils/`):**
- `ModelGradCAM` class from `gradcam.py`
- `create_gradcam_for_model()` utility function
- GradCAM visualization and heatmap generation
- Batch GradCAM analysis functionality

### Key Features

1. **Model Architecture Testing**: Tests the new modular `ResNet50Model` class
2. **Factory Pattern Testing**: Tests model creation via `model_factory`
3. **Checkpoint Loading**: Tests loading pre-trained weights into the new architecture
4. **Prediction API**: Tests the clean prediction interface (`predict`, `predict_proba`)
5. **Feature Extraction**: Tests feature extraction capabilities
6. **GradCAM Integration**: Tests both built-in and utils-based GradCAM functionality
7. **Component Access**: Tests layer access and model introspection methods

### Requirements

```bash
torch
torchvision
Pillow
numpy
```

### Usage

#### Running Unit Tests

```bash
# From the openMed/src/tests/ directory
python test_resnet50_inference.py

# Or using pytest
pytest test_resnet50_inference.py -v
```

#### Running Demo

```bash
# From the openMed/src/tests/ directory
python test_resnet50_inference.py demo
```

### Test Cases

1. **`test_resnet50_model_creation`**: Tests model creation via factory and direct instantiation
2. **`test_resnet50_model_loading`**: Tests loading from checkpoint using new architecture
3. **`test_dataset_loading`**: Tests dataset loading functionality with fallback to mock data
4. **`test_inference_on_samples`**: Main test - performs inference on 5 normal + 5 pneumonia samples using `ResNet50Model`
5. **`test_gradcam_functionality`**: Tests GradCAM using both model methods and utils functions
6. **`test_batch_inference`**: Tests batch processing using `ResNet50Model` prediction methods
7. **`test_model_components_and_features`**: Tests feature extraction and component access

### Architecture Differences from rd/

This test focuses on the **new modular architecture** rather than the research code in `rd/`:

**Old (rd/) vs New (models/utils/) Architecture:**

| Aspect | rd/ Implementation | models/utils/ Implementation |
|--------|-------------------|------------------------------|
| Model Class | `ResNet50FineTuned` | `ResNet50Model` |
| Inheritance | Direct `nn.Module` | Inherits from `BaseModel` |
| Loading | Manual state dict loading | `from_pretrained_checkpoint()` |
| Prediction | Manual forward + softmax | `predict()`, `predict_proba()` |
| GradCAM | Manual implementation | Integrated methods + utils |
| Components | Direct access | `get_feature_extractor()`, `get_classifier()` |

### Model Weights

The test looks for model weights at:
```
openMed/weights/resnet50/best_resnet50_finetuned.pth
```

**Note**: The weights trained with the `rd/` implementation are compatible with the new `ResNet50Model` architecture through the checkpoint loading mechanism.

### Dataset Structure Expected

```
C:/Users/sriha/NEU/shlabs/HP_NVIDIA/CellData/chest_xray/test/
├── NORMAL/
│   ├── normal_001.jpeg
│   ├── normal_002.jpeg
│   └── ...
└── PNEUMONIA/
    ├── pneumonia_001.jpeg
    ├── pneumonia_002.jpeg
    └── ...
```

### Fallback Behavior

If the real dataset is not available or model weights are missing, the test automatically:
- Creates mock dataset with synthetic images
- Creates untrained models for API testing
- Provides comprehensive error handling and graceful degradation

### Example Output

```
=== Testing ResNet50Model Creation ===
✓ ResNet50Model created successfully via factory
✓ ResNet50Model created successfully directly
✓ Model info: ResNet50Model, 23,512,130 parameters
✓ Model components accessible

=== Testing Inference on Samples with ResNet50Model ===
Found 2 classes: ['NORMAL', 'PNEUMONIA']
Total samples: 10
  NORMAL: 5 samples
  PNEUMONIA: 5 samples

Selected samples:
  NORMAL: 5 samples
  PNEUMONIA: 5 samples

  normal_000.jpg: NORMAL -> NORMAL (conf: 0.892)
  normal_001.jpg: NORMAL -> NORMAL (conf: 0.756)
  ...

=== Testing GradCAM Functionality ===
✓ ModelGradCAM created successfully using utils
✓ ResNet50Model.generate_gradcam_image() successful
✓ utils.gradcam.ModelGradCAM.generate_gradcam() successful
✓ Batch GradCAM analysis successful on 4 images
```

### Benefits of Testing the New Architecture

1. **Clean API**: Tests the user-friendly prediction methods
2. **Modularity**: Validates separation of concerns between models and utils
3. **Extensibility**: Ensures the base class architecture works correctly
4. **Integration**: Tests how models and utils work together
5. **Consistency**: Validates consistent behavior across different usage patterns

### Troubleshooting

1. **Model not found**: Test will create untrained model for API validation
2. **Dataset not found**: Test automatically creates mock data
3. **Import errors**: Ensure models/ and utils/ directories are in Python path
4. **GradCAM errors**: Test includes error handling for complex visualization code

### Integration with CI/CD

This test validates the core modular architecture and can be used in CI/CD:

```bash
# In CI/CD pipeline
cd openMed/src/tests/
python -m pytest test_resnet50_inference.py --tb=short -v
```

The test is designed to work even without trained weights, making it suitable for continuous integration where model files might not be available. 