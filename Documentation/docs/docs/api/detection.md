# 🎯 Object Detection & OCR Service

Performs advanced scene analysis with detection and text extraction.

## Endpoint

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/object/detect/` | Detects objects and extracts text where applicable. |

## Models

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Object Detection** | YOLOv8n | Detects and localizes objects |
| **Text Recognition** | EasyOCR | Reads text in cropped areas |

## Output Example

```json
{
  "detections": [
    {
      "label": "car",
      "confidence": 0.94,
      "bbox": [120, 45, 300, 220],
      "detected_text": "AB-123-CD"
    }
  ]
}
