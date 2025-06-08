# GradCAM Integration Features

## Overview

The OpenMed backend now includes comprehensive GradCAM (Gradient-weighted Class Activation Mapping) visualization support for medical image analysis. This feature provides real-time visual explanations of AI decisions during pneumonia detection.

## 🚀 New Features

### 1. **Real-time GradCAM Generation**
- GradCAM visualizations are generated during the medical analysis process
- Images are saved to `src/gradcam_images/` for persistent storage
- Base64 encoded images are included in API responses for immediate display

### 2. **Streaming Analysis with Visual Feedback**
- Real-time streaming of analysis progress
- GradCAM images are streamed as they're generated
- Progressive updates showing analysis steps

### 3. **GradCAM Image Management**
- Dedicated endpoints for serving and listing GradCAM images
- Secure file serving with path validation
- Timestamped filenames for easy identification

## 📡 API Endpoints

### Core Analysis Endpoints

#### `POST /v1/chat/completions` (Enhanced)
- **Streaming Support**: When `stream: true`, provides real-time analysis with GradCAM
- **Features**: 
  - Progressive analysis updates
  - Real-time GradCAM visualization streaming
  - Medical disclaimers and recommendations

#### `POST /v1/medical/analyze` (Enhanced)
- **Direct Analysis**: Analyze specific images with GradCAM generation
- **Response**: Includes GradCAM paths and base64 data

### GradCAM-Specific Endpoints

#### `GET /v1/gradcam`
- **Purpose**: List all available GradCAM images
- **Response**:
```json
{
  "gradcam_images": [
    {
      "filename": "gradcam_chest_xray_20241201_143022.png",
      "size": 45632,
      "created": "2024-12-01T14:30:22Z",
      "url": "/v1/gradcam/gradcam_chest_xray_20241201_143022.png"
    }
  ],
  "total_count": 1,
  "gradcam_directory": "/absolute/path/to/gradcam_images"
}
```

#### `GET /v1/gradcam/{filename}`
- **Purpose**: Serve individual GradCAM images
- **Security**: Path validation to prevent directory traversal
- **Response**: PNG image file

### Testing Endpoints

#### `POST /v1/medical/gradcam-test`
- **Purpose**: Test GradCAM generation for a specific uploaded file
- **Request**:
```json
{
  "file_id": "uploaded_file_id"
}
```

#### `GET /v1/stream-demo`
- **Purpose**: Demonstrate streaming capabilities with sample data
- **Response**: Mixed text and image streaming

## 🎯 Usage Examples

### 1. Chat Completion with GradCAM Streaming

```python
import requests
import json

# Start streaming analysis
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "openmed-chat-v1",
        "messages": [
            {
                "role": "user", 
                "content": "Please analyze this chest X-ray for pneumonia"
            }
        ],
        "files": [{"id": "file-abc123"}],
        "stream": True
    },
    stream=True
)

# Process streaming response
for line in response.iter_lines():
    if line:
        if line.startswith(b'data: '):
            data = json.loads(line[6:])
            content = data['choices'][0]['delta'].get('content', '')
            if content:
                print(content, end='')
```

### 2. Direct GradCAM Analysis

```python
import requests

# Upload image first
with open('chest_xray.png', 'rb') as f:
    upload_response = requests.post(
        "http://localhost:8000/v1/files/upload",
        files={"file": f}
    )
file_id = upload_response.json()['file_id']

# Analyze with GradCAM
analysis_response = requests.post(
    "http://localhost:8000/v1/medical/gradcam-test",
    json={"file_id": file_id}
)

result = analysis_response.json()
if result['data']['analysis_result']['gradcam_available']:
    gradcam_base64 = result['data']['analysis_result']['gradcam_base64']
    gradcam_path = result['data']['analysis_result']['gradcam_path']
    print(f"GradCAM saved to: {gradcam_path}")
```

### 3. Retrieve GradCAM Images

```python
import requests

# List all GradCAM images
gradcam_list = requests.get("http://localhost:8000/v1/gradcam")
images = gradcam_list.json()['gradcam_images']

# Download a specific GradCAM image
if images:
    filename = images[0]['filename']
    image_response = requests.get(f"http://localhost:8000/v1/gradcam/{filename}")
    
    with open(f"downloaded_{filename}", 'wb') as f:
        f.write(image_response.content)
```

## 🎨 GradCAM Response Format

### Analysis Result with GradCAM
```json
{
  "success": true,
  "prediction": 0,
  "confidence": 0.923,
  "prediction_label": "Normal",
  "probabilities": {
    "Normal": 0.923,
    "Pneumonia": 0.077
  },
  "gradcam_available": true,
  "gradcam_path": "/absolute/path/to/gradcam_image.png",
  "gradcam_base64": "iVBORw0KGgoAAAANSUhEUgAAA...",
  "gradcam": {
    "success": true,
    "gradcam_path": "/absolute/path/to/gradcam_image.png",
    "gradcam_base64": "iVBORw0KGgoAAAANSUhEUgAAA...",
    "message": "GradCAM visualization generated successfully"
  }
}
```

### Streaming GradCAM Response
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1701436800,
  "model": "openmed-chat-v1",
  "choices": [{
    "index": 0,
    "delta": {
      "content": "![GradCAM for chest_xray.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA...)\n"
    },
    "finish_reason": null
  }]
}
```

## 🔧 Technical Implementation

### GradCAM Generation Process

1. **Feature Extraction**: Send image to ResNet50 feature extractor with GradCAM request
2. **GradCAM API Call**: 
   ```json
   {
     "image_base64": "base64_encoded_image",
     "generate_gradcam": true,
     "target_layer": "layer4",
     "alpha": 0.4
   }
   ```
3. **Image Saving**: Save received GradCAM to `src/gradcam_images/`
4. **Response Integration**: Include in analysis response and streaming

### File Naming Convention
```
gradcam_{original_filename}_{timestamp}.png
```
Example: `gradcam_chest_xray_001_20241201_143022.png`

### Directory Structure
```
src/
├── gradcam_images/          # GradCAM visualizations
│   ├── gradcam_image1_20241201_143022.png
│   └── gradcam_image2_20241201_143055.png
├── middleware/
│   ├── services.py          # MedicalAnalysisService with GradCAM
│   ├── api_utils.py         # StreamingUtils with image support
│   └── openwebui_backend.py # Enhanced endpoints
└── ...
```

## ⚡ Performance Considerations

### Streaming Benefits
- **Real-time Feedback**: Users see progress immediately
- **Better UX**: Progressive loading vs. waiting for complete analysis
- **Resource Efficient**: Images streamed as generated

### Caching Strategy
- GradCAM images saved to disk for reuse
- Base64 data included in responses for immediate display
- File serving with proper HTTP headers

### Error Handling
- Graceful degradation if GradCAM service unavailable
- Clear error messages in streaming responses
- Fallback to analysis without GradCAM

## 🔒 Security Features

### Path Validation
- Prevent directory traversal attacks
- Validate file paths within GradCAM directory
- Secure file serving with access controls

### File Management
- Automatic cleanup of old GradCAM files (can be implemented)
- Size limits on GradCAM images
- MIME type validation

## 🎯 Integration with OpenWebUI

### Markdown Image Display
GradCAM images are embedded using markdown format:
```markdown
![GradCAM for filename](data:image/png;base64,{base64_data})
```

### Real-time Streaming
OpenWebUI can display streamed content including:
- Progressive text updates
- Inline images as they're generated
- Formatted medical analysis results

## 🚀 Future Enhancements

### Planned Features
1. **Interactive GradCAM**: Click-to-zoom functionality
2. **Comparison Views**: Side-by-side original vs. GradCAM
3. **Multiple Visualizations**: Different layer activations
4. **Animation**: Time-lapse of analysis process
5. **Export Options**: PDF reports with embedded GradCAM

### API Extensions
1. **Custom GradCAM Parameters**: Layer selection, transparency
2. **Batch Processing**: Multiple images in single request
3. **Analysis History**: Track GradCAM generations over time
4. **Quality Metrics**: GradCAM clarity and confidence scores

## 📊 Monitoring and Analytics

### Available Metrics
- GradCAM generation success rate
- Average generation time
- Storage usage for GradCAM images
- Popular visualization requests

### Logging
- Detailed logs for GradCAM generation process
- Error tracking and debugging information
- Performance metrics and optimization insights

This comprehensive GradCAM integration provides a powerful visual explanation system for medical AI decisions, enhancing trust and interpretability in automated pneumonia detection. 