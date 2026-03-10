from faster_whisper import WhisperModel
import re

# Initialize model globally to avoid reloading on every call
# "base" is a good balance of accuracy and speed on CPU
model_size = "base"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# ── Math phrase normalization map ────────────────────────────────────────────
# Maps spoken math phrases → standard math notation
# Applied after ASR transcription to handle JEE-style dictation
MATH_PHRASE_MAP = [
    # Powers / exponents
    (r'\braised to the power of\b', '^'),
    (r'\braised to the power\b', '^'),
    (r'\braised to\b', '^'),
    (r'\bto the power of\b', '^'),
    (r'\bsquared\b', '^2'),
    (r'\bcubed\b', '^3'),
    # Roots
    (r'\bsquare root of\b', 'sqrt('),
    (r'\bcube root of\b', 'cbrt('),
    # Fractions
    (r'\bdivided by\b', '/'),
    (r'\bover\b', '/'),
    # Common operations
    (r'\btimes\b', '*'),
    (r'\bmultiplied by\b', '*'),
    (r'\bplus\b', '+'),
    (r'\bminus\b', '-'),
    # Greek / math terms
    (r'\bpi\b', 'π'),
    (r'\btheta\b', 'θ'),
    (r'\balpha\b', 'α'),
    (r'\bbeta\b', 'β'),
    (r'\blambda\b', 'λ'),
    # limit notation
    (r'\bapproaches\b', '→'),
    (r'\btends to\b', '→'),
    (r'\binfinity\b', '∞'),
    (r'\bintegral of\b', '∫'),
    (r'\bderivative of\b', 'd/dx'),
]

def normalize_math_phrases(text: str) -> str:
    """Replace spoken math phrases with math notation."""
    result = text.lower()
    for pattern, replacement in MATH_PHRASE_MAP:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def transcribe_audio(audio_file_path: str):
    """
    Transcribe an audio file and normalize math phrases.

    Returns:
        tuple: (transcript: str, is_unclear: bool)
            is_unclear=True when confidence is low (triggers HITL)
    """
    try:
        segments, info = model.transcribe(audio_file_path, beam_size=5)
        segment_list = list(segments)

        if not segment_list:
            return "", True   # Empty transcript → unclear

        raw_text = " ".join([s.text for s in segment_list]).strip()

        # Normalize spoken math to notation
        normalized = normalize_math_phrases(raw_text)

        # Derive overall confidence from segment avg_logprob
        avg_logprob = sum(s.avg_logprob for s in segment_list) / len(segment_list)
        # avg_logprob is in log space; > -0.5 is generally confident
        is_unclear = avg_logprob < -0.5

        return normalized, is_unclear

    except Exception as e:
        return f"Error transcribing audio: {str(e)}", True


if __name__ == "__main__":
    test = "find the square root of x raised to 2 plus 4 as x approaches infinity"
    norm = normalize_math_phrases(test)
    print(f"Original: {test}")
    print(f"Normalized: {norm}")