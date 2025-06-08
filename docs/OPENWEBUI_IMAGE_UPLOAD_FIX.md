# OpenWebUI Image Upload Fix Documentation

## Overview

This document outlines the comprehensive fix implemented to resolve 400 Bad Request errors in OpenWebUI when uploading images. Based on web search analysis, the root causes were identified and addressed.

## Root Causes Identified

### 1. Missing File Upload Endpoint
- **Issue**: OpenWebUI expects `/api/v1/files/` for file uploads
- **Solution**: Added complete file upload endpoint with validation

### 2. Incorrect Message Content Format  
- **Issue**: Images should be in `image_url` format within content arrays
- **Solution**: Enhanced message parsing to handle OpenWebUI format

### 3. File Reference Handling
- **Issue**: OpenWebUI uses file IDs in `files` array after upload
- **Solution**: Added file storage and reference resolution

### 4. Base64 Image Support
- **Issue**: Missing support for inline base64 images
- **Solution**: Added base64 decoding in message content

## Implementation Details

### New Endpoints Added

#### 1. File Upload: `POST /api/v1/files/`
```json
{
  "id": "file-abc123",
  "object": "file", 
  "bytes": 2048,
  "created_at": 1640995200,
  "filename": "chest_xray.jpg",
  "purpose": "vision"
}
```

#### 2. File List: `GET /api/v1/files/`
```json
{
  "object": "list",
  "data": [...]
}
```

#### 3. OpenWebUI Chat: `POST /api/chat/completions`
Handles OpenWebUI-specific message format with files array.

### Enhanced Message Format Support

#### OpenWebUI File Reference Format:
```json
{
  "model": "openmed-pneumonia-agent",
  "messages": [
    {
      "role": "user", 
      "content": "Analyze this X-ray"
    }
  ],
  "files": [
    {"type": "file", "id": "file-abc123"}
  ]
}
```

#### OpenWebUI Base64 Format:
```json
{
  "model": "openmed-pneumonia-agent",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this image"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."
          }
        }
      ]
    }
  ]
}
```

### Error Handling & Logging

When 400 Bad Request errors occur, the system now logs:
- Complete request payload in JSON format
- File validation errors
- Base64 decoding errors  
- File reference resolution errors
- Available files in storage

### Supported Image Formats
- JPEG (`image/jpeg`, `image/jpg`)
- PNG (`image/png`) 
- GIF (`image/gif`)
- WebP (`image/webp`)
- DICOM (`application/dicom`)

### File Size Limits
- Maximum: 10MB per file
- Files stored in memory with automatic cleanup

## Testing the Fix

### 1. Test File Upload
```bash
curl -X POST "http://localhost:8000/api/v1/files/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

Expected Response:
```json
{
  "id": "file-1234567890-1234",
  "object": "file",
  "bytes": 45678,
  "created_at": 1640995200,
  "filename": "chest_xray.jpg", 
  "purpose": "vision"
}
```

### 2. Test Chat with File Reference
```bash
curl -X POST "http://localhost:8000/api/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openmed-pneumonia-agent",
    "messages": [
      {"role": "user", "content": "Analyze this chest X-ray for pneumonia"}
    ],
    "files": [
      {"type": "file", "id": "file-1234567890-1234"}
    ]
  }'
```

### 3. Test Chat with Base64 Image
```bash
curl -X POST "http://localhost:8000/api/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openmed-pneumonia-agent", 
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What do you see in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
            }
          }
        ]
      }
    ]
  }'
```

### 4. Test Error Logging
Send invalid request to see 400 error payload logging:
```bash
curl -X POST "http://localhost:8000/api/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openmed-pneumonia-agent",
    "messages": [{"role": "user", "content": "test"}],
    "files": [{"type": "file", "id": "nonexistent-file-id"}]
  }'
```

Expected Log Output:
```
================================================================================
🚨 400 BAD REQUEST ERROR - PAYLOAD DEBUG INFO [File Upload]
================================================================================
Error Message: Referenced file ID 'nonexistent-file-id' not found in storage
----------------------------------------
Request Payload (JSON format):
{
  "request_type": "JSON_WITH_FILE_REFS",
  "model": "openmed-pneumonia-agent",
  "messages_count": 1,
  "file_id": "nonexistent-file-id",
  "available_files": []
}
================================================================================
```

## OpenWebUI Configuration

To use with OpenWebUI:

1. **API URL**: Set to `http://your-server:8000`
2. **Models**: Use medical models like `openmed-pneumonia-agent`
3. **File Upload**: OpenWebUI will automatically use `/api/v1/files/` endpoint
4. **Chat**: Uses `/api/chat/completions` with file references

## Benefits of This Fix

### ✅ **Resolves 400 Bad Request Errors**
- Proper file upload endpoint
- Correct message format handling
- File reference resolution

### ✅ **Enhanced Debugging**
- Comprehensive error logging
- Payload inspection for 400 errors
- Clear error messages

### ✅ **OpenWebUI Compatibility**
- Supports all OpenWebUI image upload methods
- File ID reference system
- Base64 inline images

### ✅ **Medical AI Integration**
- Works with existing pneumonia detection
- ResNet50 analysis pipeline
- GradCAM visualization

## Migration Notes

- **Backwards Compatible**: Existing endpoints continue to work
- **Memory Storage**: Files stored in memory (restart clears storage)
- **File Cleanup**: Temporary files automatically cleaned up
- **Debug Endpoints**: Added for troubleshooting

This fix ensures OpenWebUI image uploads work seamlessly with the OpenMed backend while providing comprehensive error logging for debugging any remaining issues. 