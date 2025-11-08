# 🧠 Opti-Vision

AI-Powered Visual Intelligence Platform

# 🌍 About the Project

Opti-Vision is an innovative AI-based platform designed to process and understand visual data through three intelligent microservices:

- 🖼️ Image Classification

- 🎯 Object Detection & OCR

- 📄 Document Analysis

It bridges the gap between human and computer vision by offering powerful, accurate, and real-time insights from images and documents.

# 👩‍💻 About the Developer

👋 Hi, I’m Mariam Chtioui,
a Master’s student in Computer Science (Research Track) passionate about Artificial Intelligence and Data Science.

I created Opti-Vision as part of my exploration into AI-driven vision systems, blending research and real-world application.


# ⚙️ Features

- ✅ Image Classification: Recognizes objects using a trained CIFAR-10 CNN model.
- ✅ Object Detection + OCR: Detects objects and reads text (like license plates) using YOLOv8 and EasyOCR.
- ✅ Document Analysis: Extracts text and structured data from printed and handwritten documents.
- ✅ FastAPI Backend: Built for high-performance and scalability.
- ✅ Modern Frontend: Clean web interface to visualize AI results.
- ✅ Modular Architecture: Each service runs independently for flexibility and maintenance.

# 🧩 Tech Stack
- Category:	Technology
- Backend:	Python, FastAPI
- AI/ML Models:	TensorFlow, Keras, YOLOv8, EasyOCR, DocTR, Camelot
- Frontend:	HTML, CSS, JavaScript
- Documentation:	MkDocs Material
- Tools:	PyCharm, VS Code, GitHub

# 🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/YourUsername/Opti-Vision.git
cd Opti-Vision

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Backend
uvicorn main:app --reload

4️⃣ Open in Browser

Go to 👉 http://127.0.0.1:8000

# 🧠 AI Services Overview

This system includes three independent microservices deployed with FastAPI, each handling a distinct Computer Vision task.

## 🖼️ 1. CIFAR-10 Image Classifier Service (/image)

- Purpose: Classify images into one of 10 predefined categories.

- Endpoint
Method	Path	Description
POST	/image/classify/	Classifies an uploaded image.
Model Details

- Model: Custom-trained CNN

- Dataset: CIFAR-10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

- Input: 32×32 RGB image

- Output: JSON with predicted_label and confidence

<code>
Example Response
{
  "predicted_label": "cat",
  "confidence": 0.87
}
</code>

## 🎯 2. Object Detection and OCR Service (/object)

- Purpose: Detect multiple objects in an image and extract text from specific ones.

- Endpoint
Method	Path	Description
POST	/object/detect/	Detects and performs OCR on uploaded image.
Components
- Component	Technology	Role
- Object Detection	YOLOv8n	Detects and localizes objects.
- Text Recognition	EasyOCR	Extracts text from detected regions (e.g., license plates).

<code>
Example Response
{
  "detections": [
    {
      "label": "car",
      "confidence": 0.93,
      "bbox": [120, 80, 230, 190],
      "detected_text": "178 TN 2025"
    }
  ]
}
</code>

## 📄 3. Enhanced Document Analysis Service (/document)

- Purpose: Extract structured data from scanned or handwritten documents.

- Endpoint
Method	Path	Description
POST	/document/analyze/enhanced	Processes document image and extracts text + entities.
Pipeline

- OCR Extraction – Uses Pytesseract for printed text.

- Document Type Detection – Identifies if it’s an invoice, letter, or handwritten note.

- Handwritten Recognition – Uses TrOCR for handwritten text.

- Language Detection – Combines langid, langdetect, and heuristics.

- Entity Extraction – Extracts key entities like invoice number, dates, and amounts.

<code> Example Response
{
  "raw_text": "Invoice #2025, Total: $320.00",
  "language": "en",
  "entities": {
    "INVOICE_NUMBER": "2025",
    "AMOUNT": "$320.00"
  },
  "summary": "Printed invoice document"
}
</code>

# 💡 Use Cases

- 🔹 Smart Content Classification
- 🔹 Traffic and License Plate Recognition
- 🔹 Automated Document Processing
- 🔹 AI Research & Prototyping
- 🔹 Real-time Vision Intelligence

# 📘 Documentation

Full documentation built with MkDocs Material is available in the /docs folder.
You can deploy it easily with:

mkdocs serve


Then open 👉 http://127.0.0.1:8000

# ⭐ Support

If you like this project, please star 🌟 the repository — it motivates me to keep building and improving Opti-Vision!
