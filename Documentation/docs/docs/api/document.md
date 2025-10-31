# 📄 Enhanced Document Analysis Service

The **Document Analysis Service** is the most advanced module of the AI Vision system.  
It provides comprehensive **OCR**, **language detection**, and **structured data extraction** for various document types — invoices, letters, and handwritten notes.

---

## ⚙️ Endpoint

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/document/analyze/enhanced` | Processes a document image (invoice, letter, etc.), extracts text, detects document type, and extracts structured data. |

---


## 🧰 Technologies Used
-**FastAPI** – for serving the service endpoint.

-**Pytesseract** – for traditional OCR extraction.

-**HuggingFace TrOCR** – for handwritten text recognition.

-**Regex** – for pattern-based extraction (dates, invoice numbers).

-**LangID + LangDetect** – for multilingual text support.

-**PIL / OpenCV** – for image preprocessing and enhancement.

---

## 🧠 Pipeline Overview

The system uses a **multi-stage pipeline** to intelligently analyze and interpret document images.

```mermaid
flowchart TD
    A[📤 Upload Document Image] --> B[🧩 OCR Stage (Pytesseract)]
    B --> C[🧾 Document Typing (invoice, letter, handwritten)]
    C --> D{Document Type}
    D -->|Printed| E[Pytesseract for printed text]
    D -->|Handwritten| F[TrOCR (Handwriting Recognition)]
    E --> G[🌐 Language Detection (langid + langdetect)]
    F --> G
    G --> H[🔍 Entity Extraction (Regex & Heuristics)]
    H --> I[📊 Structured JSON Output]

````
## Exemple Output
```json
{
  "raw_text": "Invoice #INV-4567\nDate: 2024-06-15\nAmount: 320.00 USD\n...",
  "language": {
    "detected": "en",
    "confidence": 0.98
  },
  "entities": {
    "INVOICE_NUMBER": "INV-4567",
    "DATE": "2024-06-15",
    "AMOUNT": "320.00 USD"
  },
  "line_items": [
    { "item": "Product A", "quantity": 2, "price": 50 },
    { "item": "Product B", "quantity": 3, "price": 40 }
  ],
  "summary": {
    "document_type": "invoice",
    "contains_handwriting": false
  }
}
