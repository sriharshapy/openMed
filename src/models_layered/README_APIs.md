# ResNet50 Layered APIs

This directory contains FastAPI implementations for the ResNet50 layered architecture:
1. **Feature Extractor API** (N-1 layers) - Port 6001
2. **Final Layer Classifier API** (N layer) - Port 6005

## Quick Start

### Starting the APIs

1. **Feature Extractor API (Port 6001):**
   ```bash
   cd src/models_layered
   python run_feature_extractor_api.py
   ```
   Or alternatively:
   ```bash
   python resnet50_n_n_minus_1.py api
   ```

2. **Final Layer Classifier API (Port 6005):**
   ```bash
   cd src/models_layered
   python run_classifier_api.py
   ```
   Or alternatively:
   ```bash
   python resnet50_n.py api
   ```

### API Documentation

Once the APIs are running, you can access the interactive documentation:
- Feature Extractor: http://localhost:6001/docs
- Final Layer Classifier: http://localhost:6005/docs

## API Endpoints

### Feature Extractor API (Port 6001)

#### 1. Health Check
```http
GET /health
```
Returns the health status and model loading information.

#### 2. Extract Features
```http
POST /extract_features
```
Extracts features from input images.

**Request Body:**
```json
{
    "image_base64": "base64_encoded_image_string",
    "return_intermediate_features": false
}
```

**Response:**
```json
{
    "features": [[2048_dimensional_feature_vector]],
    "feature_shape": [batch_size, 2048],
    "intermediate_features": null
}
```

#### 3. Model Information
```http
GET /model_info
```
Returns detailed information about the loaded model.

### Final Layer Classifier API (Port 6005)

#### 1. Health Check
```http
GET /health
```
Returns the health status and model loading information.

#### 2. Classify Features
```http
POST /classify
```
Classifies features from layer4 output.

**Request Body:**
```json
{
    "features": [[[[layer4_feature_maps]]]],
    "return_probabilities": true,
    "return_features": false
}
```

**Response:**
```json
{
    "predictions": [[probability_scores]],
    "predicted_classes": [predicted_class_indices],
    "input_shape": [batch_size, channels, height, width],
    "features": null
}
```

#### 3. Predict Classes (Simple)
```http
POST /predict_classes
```
Simple endpoint that returns only predicted classes.

#### 4. Health Inference Test
```http
POST /health_inference
```
Tests the model with dummy data to verify inference is working.

#### 5. Model Information
```http
GET /model_info
```
Returns detailed information about the loaded model.

## Usage Examples

### Python Example: Feature Extraction

```python
import requests
import base64
import json
from PIL import Image
import io

# Load and encode image
with open("image.jpg", "rb") as f:
    image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

# Extract features
response = requests.post(
    "http://localhost:6001/extract_features",
    json={
        "image_base64": image_base64,
        "return_intermediate_features": False
    }
)

features = response.json()["features"]
print(f"Extracted features shape: {response.json()['feature_shape']}")
```

### Python Example: Classification

```python
import requests
import numpy as np

# Simulate layer4 output (typically 7x7 for 224x224 input)
layer4_output = np.random.randn(1, 2048, 7, 7).tolist()

# Classify features
response = requests.post(
    "http://localhost:6005/classify",
    json={
        "features": layer4_output,
        "return_probabilities": True,
        "return_features": False
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Predicted class: {result['predicted_classes']}")
```

### curl Examples

#### Feature Extraction
```bash
curl -X POST "http://localhost:6001/extract_features" \
     -H "Content-Type: application/json" \
     -d '{
       "image_base64": "YOUR_BASE64_IMAGE_STRING",
       "return_intermediate_features": false
     }'
```

#### Classification
```bash
curl -X POST "http://localhost:6005/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [[[[1.0, 2.0]]]],
       "return_probabilities": true
     }'
```

## Pipeline Usage

You can chain both APIs to create a complete classification pipeline:

1. **Step 1:** Send image to Feature Extractor API (port 6001)
2. **Step 2:** Take the features from step 1 and send to Final Layer API (port 6005)
3. **Step 3:** Get final classification results

### Complete Pipeline Example

```python
import requests
import base64
import numpy as np

# Step 1: Feature extraction
with open("chest_xray.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

features_response = requests.post(
    "http://localhost:6001/extract_features",
    json={"image_base64": image_base64, "return_intermediate_features": True}
)

# Get layer4 features (we need the raw layer4 output, not the final pooled features)
# In practice, you'd modify the API to return layer4 output directly
# For now, we'll simulate it:
layer4_features = np.random.randn(1, 2048, 7, 7).tolist()

# Step 2: Classification
classification_response = requests.post(
    "http://localhost:6005/classify",
    json={
        "features": layer4_features,
        "return_probabilities": True
    }
)

result = classification_response.json()
print(f"Final prediction: {result['predicted_classes'][0]}")
print(f"Confidence: {max(result['predictions'][0]):.4f}")
```

## Model Checkpoints

The APIs will automatically look for trained model checkpoints in:
```
rd/checkpoints/resnet50/best_resnet50_finetuned.pth
```

If no checkpoint is found:
- Feature Extractor will use ImageNet pretrained weights
- Final Layer will use randomly initialized weights

## Error Handling

All endpoints include proper error handling and will return appropriate HTTP status codes:
- 200: Success
- 400: Bad request (invalid input)
- 500: Internal server error (model issues)

## Performance Notes

- Both APIs use GPU if available, fallback to CPU
- Models are loaded once at startup for optimal performance
- Input validation ensures proper tensor shapes
- Batch processing is supported

## Development

To modify the APIs:
1. Edit the respective Python files (`resnet50_n_n_minus_1.py` or `resnet50_n.py`)
2. The FastAPI code is at the bottom of each file
3. Restart the API servers to see changes 