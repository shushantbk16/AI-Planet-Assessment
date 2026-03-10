"""
OCR Tool — Vision LLM primary, EasyOCR fallback
================================================
Primary: Groq llama-3.2-11b-vision-preview
  • Natively understands math notation, superscripts, fractions, Greek letters
  • Prompts the model to extract text in standard math notation
  
Fallback: EasyOCR + OpenCV preprocessing
  • Used if the Groq API is unavailable or image cannot be base64-encoded
"""
import os
import base64
import io
import easyocr
import numpy as np
from PIL import Image
import cv2
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Confidence threshold for triggering HITL review
CONFIDENCE_THRESHOLD = 0.75

# EasyOCR fallback reader (lazy init — only used if vision LLM fails)
_easyocr_reader = None
def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'])
    return _easyocr_reader


def _image_to_base64(image_file) -> str:
    """Convert any image file-like object to a base64 data URL."""
    if not hasattr(image_file, "read"):
        with open(image_file, "rb") as f:
            raw = f.read()
    else:
        image_file.seek(0)
        raw = image_file.read()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_via_vision_llm(image_file) -> tuple[str, float]:
    """
    Use Groq vision model to extract math text from an image.
    Returns (text, confidence) where confidence is 0.95 on success, 0.0 on error.
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        data_url = _image_to_base64(image_file)

        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are an expert mathematician and OCR assistant. Extract ALL mathematical text, "
                                "equations, and text from this image EXACTLY as written.\n\n"
                                "CRITICAL RULES:\n"
                                "1. Convert superscripts into rigorous standard notation (e.g., a² → a^2, a³ → a^3, n² → n^2, e^x).\n"
                                "2. Convert subscripts using underscore (e.g., a₁ → a_1).\n"
                                "3. Transcribe complex structures like summations rigorously (e.g. sum from n=1 to 21 of 3/((4n-1)(4n+3))).\n"
                                "4. Transcribe large fractions accurately with parentheses around numerators and denominators if needed: (A) / (B).\n"
                                "5. If there are multiple parts (like question and multiple choice answers A, B, C, D), include ALL of them.\n"
                                "6. Do not try to solve the problem, just extract the text.\n"
                                "7. Output ONLY the extracted text, absolutely no conversational filler or markdown formatting headers."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        # Clean up markdown code blocks if the model accidentally includes them
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
            
        return text.strip(), 0.95          # Vision LLM is very reliable

    except Exception as e:
        return "", 0.0


def _extract_via_easyocr(image_file) -> tuple[str, float]:
    """EasyOCR + OpenCV fallback. Returns (text, mean_confidence)."""
    try:
        if hasattr(image_file, "seek"):
            image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)

        gray   = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blur   = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        reader  = _get_reader()
        results = reader.readtext(thresh)

        if not results:
            return "", 0.0

        texts = [t for (_, t, _) in results]
        probs = [p for (_, _, p) in results]
        return " ".join(texts).strip(), float(sum(probs) / len(probs))

    except Exception as e:
        return f"Error extracting text: {e}", 0.0


def extract_text_from_image(image_file) -> tuple[str, float]:
    """
    Extract math text from an image.
    
    Strategy:
      1. Try Vision LLM (Groq) — best for math superscripts / fractions
      2. Fall back to EasyOCR if LLM fails
      
    Returns:
        (extracted_text: str, confidence: float [0–1])
    """
    text, conf = _extract_via_vision_llm(image_file)
    if text:
        return text, conf
    # Fallback
    return _extract_via_easyocr(image_file)


def is_low_confidence(confidence: float) -> bool:
    return confidence < CONFIDENCE_THRESHOLD


if __name__ == "__main__":
    print("OCR Tool ready.")
    print("Primary: Groq vision LLM (llama-3.2-11b-vision-preview)")
    print("Fallback: EasyOCR + OpenCV")
    print("Confidence threshold:", CONFIDENCE_THRESHOLD)