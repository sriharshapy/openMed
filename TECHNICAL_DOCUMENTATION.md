# OpenMed Technical Documentation

## 📋 Table of Contents

1. [System Architecture Deep Dive](#system-architecture-deep-dive)
2. [API Specifications](#api-specifications)
3. [Model Architecture Details](#model-architecture-details)
4. [Deployment Guide](#deployment-guide)
5. [Performance Benchmarks](#performance-benchmarks)
6. [Troubleshooting Guide](#troubleshooting-guide)

## System Architecture Deep Dive

### Service Dependencies

| Service | Dependencies | Port | Health Check |
|---------|-------------|------|--------------|
| OpenWebUI | Middleware API | 3000 | http://localhost:3000/health |
| Middleware API | Agent, OpenAI | 8000 | http://localhost:8000/health |
| Feature Extractor | PyTorch, CUDA | 6001 | http://localhost:6001/health |
| Classifier API | Feature Extractor | 6005 | http://localhost:6005/health |
| MLflow Pneumonia | None | 5000 | http://localhost:5000 |
| MLflow TB | None | 5001 | http://localhost:5001 |
| MLflow Brain Tumor | None | 5002 | http://localhost:5002 |

### Component Interaction Flow

The system follows a layered architecture where:

1. **Frontend Layer**: OpenWebUI provides the user interface
2. **API Gateway**: Middleware handles request routing and OpenAI compatibility
3. **AI Services**: Intelligent agent processes medical intent classification
4. **ML Models**: ResNet50-based disease detection models
5. **Infrastructure**: MLflow tracking and model storage

## API Specifications

### OpenAI Compatible Endpoints

#### Chat Completions API

```http
POST /v1/chat/completions
Content-Type: application/json

{
    "model": "openmed-medical-v1",
    "messages": [
        {
            "role": "system",
            "content": "You are a medical AI assistant."
        },
        {
            "role": "user", 
            "content": "Analyze this chest X-ray for pneumonia",
            "image": "base64_encoded_image"
        }
    ],
    "max_tokens": 1000,
    "temperature": 0.3,
    "stream": false
}
```

### Medical Model APIs

#### Feature Extraction API (Port 6001)

```http
POST /extract_features
Content-Type: application/json

{
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "return_intermediate_features": false
}
```

#### Disease Classification API (Port 6005)

```http
POST /classify
Content-Type: application/json

{
    "features": [[2048_dimensional_feature_vector]],
    "return_probabilities": true,
    "return_gradcam": true
}
```

## Model Architecture Details

### ResNet50 Transfer Learning

The system uses ResNet50 models pretrained on ImageNet with the following configuration:

- **Frozen Layers**: All convolutional layers (feature extraction)
- **Trainable**: Only final fully connected layer
- **Input Size**: 224×224 RGB images
- **Output**: Disease-specific classifications

### Supported Medical Conditions

| Disease | Model | Classes | Expected Accuracy | Data Type |
|---------|-------|---------|------------------|-----------|
| Pneumonia | ResNet50 | 2 (Normal, Pneumonia) | >85% | Chest X-rays |
| Tuberculosis | ResNet50 | 2 (Normal, TB) | >85% | Chest X-rays |
| Brain Tumor | ResNet50 | 3 (Glioma, Meningioma, Tumor) | >70% | Brain MRI |

### GradCAM Interpretability

The system provides visual explanations for model decisions using Gradient-weighted Class Activation Mapping (GradCAM):

- Highlights regions of interest in medical images
- Provides clinical insights for validation
- Supports trust and interpretability

## Deployment Guide

### Local Development Setup

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

2. **CUDA Configuration**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Environment Variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### Production Deployment

#### Docker Deployment
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src/ /app/src/
COPY checkpoints/ /app/checkpoints/
WORKDIR /app

EXPOSE 8000 6001 6005 3000
CMD ["python3", "src/middleware/run_backend.py"]
```

#### System Requirements

**Minimum Requirements:**
- GPU: 4GB VRAM
- RAM: 8GB system memory
- Storage: 20GB free space
- Python: 3.8+

**Recommended Setup:**
- GPU: NVIDIA RTX 3070+ (8GB VRAM)
- RAM: 16GB+ system memory
- Storage: SSD with 50GB+ free space
- CPU: 8+ cores

## Performance Benchmarks

### Inference Latency (Single Image)

| Model | Input Size | CPU (ms) | GPU (ms) | Memory (MB) |
|-------|------------|----------|----------|-------------|
| ResNet50 Pneumonia | 224×224 | 450 | 45 | 2,100 |
| ResNet50 TB | 224×224 | 440 | 43 | 2,100 |
| ResNet50 Brain Tumor | 224×224 | 460 | 47 | 2,100 |

### API Response Times

| Endpoint | Mean (ms) | 95th Percentile (ms) |
|----------|-----------|---------------------|
| /health | 5 | 12 |
| /v1/models | 8 | 18 |
| /extract_features | 450 | 680 |
| /classify | 85 | 150 |
| /v1/chat/completions | 2500 | 4500 |

## Troubleshooting Guide

### Common Issues

#### 1. CUDA Out of Memory
```bash
Error: RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size in model configuration
- Clear CUDA cache: `torch.cuda.empty_cache()`
- Use gradient checkpointing for training

#### 2. Model Loading Failures
```bash
Error: FileNotFoundError: Model checkpoint not found
```

**Solutions:**
- Verify checkpoint paths in configuration
- Check file permissions
- Ensure models are properly saved

#### 3. OpenAI API Key Issues
```bash
Error: OpenAI API key not found
```

**Solutions:**
- Check `.env` file contains `OPENAI_API_KEY`
- Verify environment variable loading
- Test API key validity

#### 4. Port Conflicts
```bash
Error: Port 8000 is already in use
```

**Solutions:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app:app --port 8001
```

### Performance Optimization

#### 1. Enable Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

#### 2. Optimize Data Loading
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

#### 3. Model Inference Optimization
```python
# Convert to TorchScript for faster inference
model.eval()
traced_model = torch.jit.trace(model, example_input)
traced_model = torch.jit.optimize_for_inference(traced_model)
```

### Monitoring and Logging

#### Health Check Implementation
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "gpu": torch.cuda.is_available(),
            "models_loaded": check_models_loaded()
        }
    }
```

#### MLflow Integration
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.pytorch.log_model(model, "model")
```

### Security Considerations

- **Input Validation**: Validate medical image formats and sizes
- **API Authentication**: Implement JWT-based authentication
- **Rate Limiting**: Prevent API abuse with request limits
- **File Upload Security**: Sanitize and validate uploaded files
- **CORS Configuration**: Properly configure cross-origin requests

---

This technical documentation provides comprehensive details for developers and system administrators working with the OpenMed platform. For additional support, refer to the main README.md or create an issue in the project repository. 