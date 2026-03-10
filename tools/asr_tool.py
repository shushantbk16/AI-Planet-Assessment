from faster_whisper import WhisperModel
import re
model_size = 'base'
model = WhisperModel(model_size, device='cpu', compute_type='int8')
MATH_PHRASE_MAP = [('\\braised to the power of\\b', '^'), (
    '\\braised to the power\\b', '^'), ('\\braised to\\b', '^'), (
    '\\bto the power of\\b', '^'), ('\\bsquared\\b', '^2'), ('\\bcubed\\b',
    '^3'), ('\\bsquare root of\\b', 'sqrt('), ('\\bcube root of\\b',
    'cbrt('), ('\\bdivided by\\b', '/'), ('\\bover\\b', '/'), (
    '\\btimes\\b', '*'), ('\\bmultiplied by\\b', '*'), ('\\bplus\\b', '+'),
    ('\\bminus\\b', '-'), ('\\bpi\\b', 'π'), ('\\btheta\\b', 'θ'), (
    '\\balpha\\b', 'α'), ('\\bbeta\\b', 'β'), ('\\blambda\\b', 'λ'), (
    '\\bapproaches\\b', '→'), ('\\btends to\\b', '→'), ('\\binfinity\\b',
    '∞'), ('\\bintegral of\\b', '∫'), ('\\bderivative of\\b', 'd/dx')]


def normalize_math_phrases(text: str) ->str:
    result = text.lower()
    for pattern, replacement in MATH_PHRASE_MAP:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def transcribe_audio(audio_file_path: str):
    try:
        segments, info = model.transcribe(audio_file_path, beam_size=5)
        segment_list = list(segments)
        if not segment_list:
            return '', True
        raw_text = ' '.join([s.text for s in segment_list]).strip()
        normalized = normalize_math_phrases(raw_text)
        avg_logprob = sum(s.avg_logprob for s in segment_list) / len(
            segment_list)
        is_unclear = avg_logprob < -0.5
        return normalized, is_unclear
    except Exception as e:
        return f'Error transcribing audio: {str(e)}', True


if __name__ == '__main__':
    test = (
        'find the square root of x raised to 2 plus 4 as x approaches infinity'
        )
    norm = normalize_math_phrases(test)
    print(f'Original: {test}')
    print(f'Normalized: {norm}')
