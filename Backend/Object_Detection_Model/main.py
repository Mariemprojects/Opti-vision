from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import easyocr
import numpy as np
import logging
import torch


router = APIRouter(prefix="/object")


# Set up logging for better error visibility
logging.basicConfig(level=logging.INFO)

# --- Model Loading ---
# Load YOLO model.
try:
    logging.info("Loading YOLOv8n model...")
    yolo_model = YOLO("yolov8n.pt")
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Could not load the YOLO model. Error: {e}")

# Load EasyOCR reader.
# I set the languages to 'en' (English).
try:
    logging.info("Loading EasyOCR reader...")
    ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    logging.info("EasyOCR reader loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Could not load the EasyOCR reader. Error: {e}")



@router.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Performs object detection and text recognition on an uploaded image.
    """
    try:
        # Read the image file and convert to a PIL Image object
        image_bytes = await file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)

        # Run the YOLO model on the image
        # 'conf=0.25' is the confidence threshold to filter out weak detections
        results = yolo_model.predict(image_np, conf=0.25)

        # Prepare a list to hold all detected objects and their details
        detected_objects = []

        # Iterate through the detection results
        # 'results[0]' represents the results for the first image in the batch
        if results and results[0].boxes:
            for box in results[0].boxes:
                # Extract the bounding box coordinates, label, and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = yolo_model.names[class_id]

                # Default text is None, it will only be set if OCR finds something
                detected_text = None

                # Perform OCR on specific objects like 'car', 'bus', 'truck', etc.
                # You can customize this list to suit your needs
                if label in ['car', 'bus', 'truck', 'motorcycle']:
                    # Crop the region of interest for OCR
                    cropped_image = pil_image.crop((x1, y1, x2, y2))

                    # Use EasyOCR to recognize text in the cropped region
                    # `detail=0` returns only the recognized text strings
                    ocr_results = ocr_reader.readtext(np.array(cropped_image), detail=0)

                    if ocr_results:
                        # Join all recognized text into a single string
                        detected_text = " ".join(ocr_results)

                # Add the object's details to our list
                detected_objects.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "text": detected_text
                })

        # Return a JSON response with the list of detected objects
        return {"detections": detected_objects}

    except Exception as e:
        # Log the full error for debugging
        logging.error(f"Error during image processing: {e}", exc_info=True)
        # Return a user-friendly error message
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {e}"
        )
