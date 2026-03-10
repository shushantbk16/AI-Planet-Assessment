import os
import base64
import io
from PIL import Image
from groq import Groq
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
CONFIDENCE_THRESHOLD = 0.75



def _image_to_base64(image_file) ->str:
    if not hasattr(image_file, 'read'):
        with open(image_file, 'rb') as f:
            raw = f.read()
    else:
        image_file.seek(0)
        raw = image_file.read()
    encoded = base64.b64encode(raw).decode('utf-8')
    return f'data:image/jpeg;base64,{encoded}'


def _extract_via_vision_llm(image_file) ->tuple[str, float]:
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        data_url = _image_to_base64(image_file)
        response = client.chat.completions.create(model=
            'llama-3.2-11b-vision-preview', messages=[{'role': 'user',
            'content': [{'type': 'image_url', 'image_url': {'url': data_url
            }}, {'type': 'text', 'text':
            """You are an expert mathematician and OCR assistant. Extract ALL mathematical text, equations, and text from this image EXACTLY as written.

CRITICAL RULES:
1. Convert superscripts into rigorous standard notation (e.g., a² → a^2, a³ → a^3, n² → n^2, e^x).
2. Convert subscripts using underscore (e.g., a₁ → a_1).
3. Transcribe complex structures like summations rigorously (e.g. sum from n=1 to 21 of 3/((4n-1)(4n+3))).
4. Transcribe large fractions accurately with parentheses around numerators and denominators if needed: (A) / (B).
5. If there are multiple parts (like question and multiple choice answers A, B, C, D), include ALL of them.
6. Do not try to solve the problem, just extract the text.
7. Output ONLY the extracted text, absolutely no conversational filler or markdown formatting headers."""
            }]}], max_tokens=1024, temperature=0.0)
        text = response.choices[0].message.content.strip()
        if text.startswith('```'):
            text = '\n'.join(text.split('\n')[1:])
        if text.endswith('```'):
            text = '\n'.join(text.split('\n')[:-1])
        return text.strip(), 0.95
    except Exception as e:
        return '', 0.0


def extract_text_from_image(image_file) ->tuple[str, float]:
    return _extract_via_vision_llm(image_file)


def is_low_confidence(confidence: float) ->bool:
    return confidence < CONFIDENCE_THRESHOLD


if __name__ == '__main__':
    print('OCR Tool ready.')
    print('Primary: Groq vision LLM (llama-3.2-11b-vision-preview)')
    print('Confidence threshold:', CONFIDENCE_THRESHOLD)
