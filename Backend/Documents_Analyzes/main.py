from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from transformers import (
    LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
    TrOCRProcessor, VisionEncoderDecoderModel,
    pipeline
)
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
from typing import Dict, List, Any
import re
import json
import pytesseract
import cv2
from langdetect import detect, LangDetectException, detect_langs
import langid

router = APIRouter(prefix="/document")

print("Loading enhanced document analysis system with improved language detection...")

# Initialize models
try:
    # LayoutLMv3 for document understanding
    layout_model_name = "microsoft/layoutlmv3-base"
    processor = LayoutLMv3Processor.from_pretrained(layout_model_name)
    model = LayoutLMv3ForTokenClassification.from_pretrained(layout_model_name)
    print(f"✅ Loaded LayoutLMv3 model: {layout_model_name}")
except Exception as e:
    print(f"❌ Error loading LayoutLMv3 model: {e}")
    processor = None
    model = None

try:
    # TrOCR for handwritten text recognition
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    print("✅ Loaded TrOCR model for handwritten text")
except Exception as e:
    print(f"❌ Error loading TrOCR model: {e}")
    trocr_processor = None
    trocr_model = None

# Improved language detection setup
language_detector = None
try:
    # Try multiple language detection approaches
    print("✅ Using ensemble language detection (langid + fallbacks)")
except Exception as e:
    print(f"❌ Error setting up language detection: {e}")


def improved_language_detection(text):
    """Enhanced language detection with multiple fallbacks"""
    if not text or len(text.strip()) < 5:
        return {"language": "unknown", "confidence": 0.0, "method": "insufficient_text"}

    text_sample = text[:1000]  # Use first 1000 chars for efficiency

    try:
        # Method 1: langid (fast and reliable for common languages)
        lang, confidence = langid.classify(text_sample)
        if confidence > 0.8:
            return {
                "language": lang,
                "confidence": float(confidence),
                "method": "langid"
            }
    except Exception as e:
        print(f"langid error: {e}")

    try:
        # Method 2: langdetect with multiple results
        from langdetect import detect_langs
        languages = detect_langs(text_sample)
        if languages:
            best_lang = languages[0]
            if best_lang.prob > 0.6:
                return {
                    "language": best_lang.lang,
                    "confidence": float(best_lang.prob),
                    "method": "langdetect",
                    "all_detections": [{"lang": l.lang, "prob": float(l.prob)} for l in languages[:3]]
                }
    except Exception as e:
        print(f"langdetect error: {e}")

    try:
        # Method 3: Simple heuristic based on common words
        english_words = set(['the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they',
                             'this', 'have', 'from', 'one', 'had', 'word', 'but', 'not', 'what', 'all',
                             'were', 'when', 'your', 'can', 'said', 'there', 'use', 'each', 'which',
                             'how', 'their', 'will', 'other', 'about', 'out', 'many', 'then', 'them',
                             'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'time',
                             'has', 'two', 'more', 'write', 'see', 'number', 'way', 'could', 'people',
                             'my', 'than', 'first', 'water', 'been', 'call', 'who', 'oil', 'its',
                             'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made'])

        french_words = set(['le', 'de', 'un', 'à', 'être', 'et', 'en', 'avoir', 'que', 'pour',
                            'dans', 'ce', 'il', 'qui', 'ne', 'sur', 'se', 'pas', 'plus', 'pouvoir',
                            'par', 'je', 'avec', 'tout', 'faire', 'son', 'mettre', 'autre', 'on',
                            'mais', 'nous', 'comme', 'ou', 'si', 'leur', 'y', 'dire', 'elle',
                            'devoir', 'avant', 'deux', 'même', 'prendre', 'aussi', 'celui', 'donner',
                            'bien', 'où', 'fois', 'vous', 'encore', 'nouveau', 'aller', 'cela',
                            'entre', 'premier', 'vouloir', 'déjà', 'grand', 'mon', 'me', 'moins'])

        spanish_words = set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del',
                             'las', 'un', 'por', 'con', 'no', 'una', 'su', 'para', 'es', 'al',
                             'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí',
                             'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también',
                             'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante',
                             'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante',
                             'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo'])

        german_words = set(['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
                            'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als',
                            'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'daß', 'sie', 'nach',
                            'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie', 'einem', 'über',
                            'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber', 'vor', 'bis',
                            'mehr', 'durch', 'man', 'sein', 'wurde', 'dessen', 'zeit', 'unter'])

        words = re.findall(r'\b[a-z]+\b', text_sample.lower())
        word_counts = {}

        for word_set, lang_name in [(english_words, 'en'), (french_words, 'fr'),
                                    (spanish_words, 'es'), (german_words, 'de')]:
            matches = len([w for w in words if w in word_set])
            if len(words) > 0:
                word_counts[lang_name] = matches / len(words)

        if word_counts:
            best_lang = max(word_counts.items(), key=lambda x: x[1])
            if best_lang[1] > 0.1:  # At least 10% matching words
                return {
                    "language": best_lang[0],
                    "confidence": float(best_lang[1]),
                    "method": "word_heuristic"
                }
    except Exception as e:
        print(f"Word heuristic error: {e}")

    return {"language": "unknown", "confidence": 0.0, "method": "all_failed"}


def preprocess_image_for_handwriting(image):
    """
    Preprocess image to improve handwritten text recognition while maintaining RGB
    """
    try:
        # Convert PIL to OpenCV (keep RGB)
        img_array = np.array(image)

        # Convert to grayscale for processing
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply filters to enhance handwritten text
        denoised = cv2.medianBlur(gray, 3)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)

        # Convert back to RGB (3 channels) for TrOCR
        rgb_enhanced = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2RGB)

        # Convert back to PIL
        enhanced_image = Image.fromarray(rgb_enhanced)
        return enhanced_image

    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return image


def extract_text_with_trocr(image):
    """Extract text using TrOCR (specialized for handwritten text)"""
    try:
        if trocr_processor is None or trocr_model is None:
            return "TrOCR model not available"

        # Ensure image is RGB (3 channels)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess image for TrOCR
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values

        # Generate text
        generated_ids = trocr_model.generate(
            pixel_values,
            max_length=128,  # Increased for longer text
            num_beams=4,
            early_stopping=True
        )
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text.strip()
    except Exception as e:
        print(f"TrOCR processing error: {str(e)}")
        return f"TrOCR error: {str(e)}"


def extract_text_with_tesseract(image, is_handwritten=False):
    """Extract text using Tesseract OCR"""
    try:
        if is_handwritten:
            # Try different PSM modes for handwriting
            best_text = ""
            for psm in [6, 8, 13]:
                config = f'--oem 3 --psm {psm} -l eng'  # Specify English
                text = pytesseract.image_to_string(image, config=config)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
            return best_text.strip()
        else:
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
    except Exception as e:
        return f"Tesseract error: {str(e)}"


def detect_document_type(image, ocr_text):
    """Detect if document is handwritten, invoice, form, etc."""
    text_lower = ocr_text.lower()

    # Check for invoice indicators
    invoice_indicators = ['invoice', 'bill', 'total', 'amount', 'date', '#', 'subtotal', 'tax', 'due', 'balance']
    is_invoice = any(indicator in text_lower for indicator in invoice_indicators)

    # Check for letter indicators (like your example)
    letter_indicators = ['dear', 'sincerely', 'yours', 'love', 'regards', 'letter', 'dad', 'mom', 'family']
    is_letter = any(indicator in text_lower for indicator in letter_indicators)

    # Check for handwritten characteristics
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]

    if not lines:
        return {
            'is_invoice': False,
            'is_handwritten': False,
            'is_letter': False,
            'line_count': 0,
            'avg_line_length': 0
        }

    avg_line_length = sum(len(line) for line in lines) / len(lines)
    line_lengths = [len(line) for line in lines]
    line_length_variance = np.var(line_lengths) if len(lines) > 1 else 0

    # Handwritten text detection heuristic
    is_handwritten = (avg_line_length < 25 and line_length_variance > 30) or \
                     any(word in text_lower for word in ['handwritten', 'signature', 'dear', 'sincerely'])

    return {
        'is_invoice': bool(is_invoice),
        'is_handwritten': bool(is_handwritten),
        'is_letter': bool(is_letter),
        'line_count': int(len(lines)),
        'avg_line_length': float(avg_line_length)
    }


def extract_invoice_specific_info(ocr_text):
    """Specialized extraction for invoice documents"""
    results = []

    # Enhanced patterns for multilingual support
    invoice_patterns = [
        r'(?:invoice|facture|rechnung)\s*#?\s*:?\s*([a-zA-Z0-9-]+)',
        r'(?:inv\.?|fact\.?)\s*#?\s*:?\s*([a-zA-Z0-9-]+)',
        r'#\s*([a-zA-Z0-9-]+)',
    ]

    for pattern in invoice_patterns:
        matches = re.findall(pattern, ocr_text, re.IGNORECASE)
        for match in matches:
            results.append({"entity": "INVOICE_NUMBER", "text": match, "confidence": "high"})

    # Multilingual date patterns
    date_patterns = [
        r'(?:date|datum|fecha)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:date|datum|fecha)\s*:?\s*(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    ]

    for pattern in date_patterns:
        dates = re.findall(pattern, ocr_text, re.IGNORECASE)
        for date in dates:
            results.append({"entity": "DATE", "text": date, "confidence": "medium"})

    # Currency patterns (multiple currencies)
    currency_patterns = [
        r'(?:total|summe|total|montant)\s*:?\s*[€$£]?\s*(\d+\.?\d*)',
        r'(?:amount|betrag|montant)\s*:?\s*[€$£]?\s*(\d+\.?\d*)',
        r'[€$£]\s*(\d+\.?\d*)',
    ]

    for pattern in currency_patterns:
        amounts = re.findall(pattern, ocr_text, re.IGNORECASE)
        for amount in amounts:
            results.append({"entity": "AMOUNT", "text": f"${amount}", "confidence": "high"})

    # Contact information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, ocr_text)
    for email in emails:
        results.append({"entity": "EMAIL", "text": email, "confidence": "high"})

    # International phone patterns
    phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}'
    phones = re.findall(phone_pattern, ocr_text)
    for phone in phones:
        results.append({"entity": "PHONE", "text": phone, "confidence": "high"})

    return results


def extract_line_items_enhanced(ocr_text):
    """Enhanced line item extraction for invoices"""
    lines = ocr_text.split('\n')
    line_items = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) < 3:
            continue

        item = {
            "line_number": i + 1,
            "text": line,
            "confidence": "medium"
        }

        # Enhanced pattern matching for line items
        patterns = [
            r'(\d+)\s+([€$£]?\d+\.?\d*)\s+([€$£]?\d+\.?\d*)',  # qty price total
            r'(.+?)\s+(\d+)\s+([€$£]?\d+\.?\d*)',  # description qty price
            r'([€$£]\d+\.?\d*).*?([€$£]\d+\.?\d*)',  # price and total
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    item.update({
                        "quantity": groups[0],
                        "unit_price": groups[1],
                        "amount": groups[2],
                        "confidence": "high"
                    })
                elif len(groups) == 2:
                    item.update({
                        "amount": groups[0],
                        "total": groups[1],
                        "confidence": "high"
                    })
                break

        line_items.append(item)

    return line_items


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@router.post("/analyze/enhanced")
async def analyze_document_enhanced(file: UploadFile = File(...)):
    """Enhanced analysis with improved language detection"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        print(f"Processing document: {file.filename}")

        # Read and process image
        image_data = await file.read()
        original_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Step 1: Initial analysis to determine approach
        initial_ocr = extract_text_with_tesseract(original_image)
        doc_type = detect_document_type(original_image, initial_ocr)
        language_info = improved_language_detection(initial_ocr)

        print(f"Document type: {doc_type}")
        print(f"Language detected: {language_info}")

        # Step 2: Choose OCR method based on document type
        final_ocr = ""
        ocr_method = ""

        if doc_type['is_handwritten'] and trocr_processor:
            print("Using TrOCR for handwritten text")
            try:
                processed_image = preprocess_image_for_handwriting(original_image)
                final_ocr = extract_text_with_trocr(processed_image)
                ocr_method = "TrOCR (handwriting optimized)"
                print("TrOCR completed successfully")
            except Exception as e:
                print(f"TrOCR failed, falling back to Tesseract: {e}")
                final_ocr = extract_text_with_tesseract(original_image, True)
                ocr_method = "Tesseract (fallback)"
        else:
            print("Using Tesseract OCR")
            final_ocr = extract_text_with_tesseract(original_image, doc_type['is_handwritten'])
            ocr_method = "Tesseract"

        # If OCR failed, use initial OCR as fallback
        if not final_ocr or len(final_ocr.strip()) < 10:
            print("Using initial OCR as fallback")
            final_ocr = initial_ocr
            ocr_method += " (fallback to initial)"

        # Step 3: Enhanced language detection with final text
        final_language_info = improved_language_detection(final_ocr)

        # Step 4: Extract information
        entities = extract_invoice_specific_info(final_ocr)
        line_items = extract_line_items_enhanced(final_ocr)

        # Step 5: Create comprehensive response
        response = convert_numpy_types({
            "filename": file.filename,
            "document_type": doc_type,
            "language": final_language_info,
            "raw_text": final_ocr,
            "entities": entities,
            "line_items": line_items,
            "summary": {
                "characters_extracted": len(final_ocr),
                "lines_extracted": len(final_ocr.split('\n')),
                "entities_found": len(entities),
                "line_items_found": len(line_items),
                "is_handwritten": doc_type['is_handwritten'],
                "is_invoice": doc_type['is_invoice'],
                "is_letter": doc_type['is_letter']
            },
            "processing_info": {
                "ocr_method": ocr_method,
                "handwriting_enhancement_applied": doc_type['is_handwritten'],
                "invoice_specific_processing": doc_type['is_invoice'],
                "language_detection_method": final_language_info.get('method', 'unknown'),
                "language_detection_confidence": final_language_info.get('confidence', 0.0)
            },
            "success": True,
            "message": "Document processed successfully with enhanced OCR and language detection"
        })

        return JSONResponse(content=response)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/health")
async def health_check():
    models_loaded = {
        "layoutlmv3": processor is not None,
        "trocr": trocr_processor is not None,
        "language_detection": "ensemble_method"
    }

    return {
        "status": "healthy",
        "version": "3.1.0",
        "models_loaded": models_loaded,
        "features": [
            "TrOCR for handwritten text",
            "Ensemble language detection (langid + fallbacks)",
            "Enhanced invoice processing",
            "Smart OCR method selection",
            "Improved English detection for handwritten letters"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(router, host="0.0.0.0", port=8000)