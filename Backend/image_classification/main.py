import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from fastapi import APIRouter, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware  <-- REMOVED: CORS belongs in main app.py
from PIL import Image
from io import BytesIO

# Load the trained Keras model
MODEL_FILE_PATH = "cifar10_image_classifier.h5"
try:

    model = tf.keras.models.load_model(MODEL_FILE_PATH)
    print(f"Successfully loaded model from {MODEL_FILE_PATH}")
except Exception as e:
    raise RuntimeError(f"Could not load the model from {MODEL_FILE_PATH}. Please make sure the file exists and is named correctly. Error: {e}")

# Define the class labels in the correct order
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']


router = APIRouter(prefix="/image")


@router.get("/")
async def get_router_status():
    """
    Optional: Check to see if the router is reachable at /image/.
    """
    return {"message": "Image classification API is running!"}


@router.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classifies an uploaded image from the CIFAR-10 dataset.
    Final endpoint URL will be: /image/classify
    """
    try:
        # Read the image file and convert it to a PIL Image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')

        # Resize the image to 32x32 pixels, as required by the model
        image = image.resize((32, 32))

        # Convert the image to a NumPy array and normalize
        img_array = np.array(image) / 255.0

        # The model expects a batch dimension, so we add one
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class index and confidence
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        # Get the predicted label from the list of class labels
        predicted_label = class_labels[predicted_class_index]

        # Return the JSON response
        return {
            "predicted_label": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Detailed Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

