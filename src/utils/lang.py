from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

SUPPORTED_UI_LANGS = ["en", "hi", "es", "fr", "de", "ar", "zh", "ja", "pt", "ru"]

def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def pick_ui_lang(user_pref: str | None) -> str:
    if user_pref and user_pref in SUPPORTED_UI_LANGS:
        return user_pref
    return "en"
