from langdetect import detect

def detect_language(text: str) -> str:
    """
    Detect language code from text using langdetect.
    Returns ISO 639-1 language code like 'en', 'hi', 'es'.
    """
    try:
        return detect(text)
    except:
        return "unknown"
