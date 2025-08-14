import os
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

LANG_CODE_MAP = {
    "en":"en_XX","hi":"hi_IN","es":"es_XX","fr":"fr_XX","de":"de_DE",
    "ar":"ar_AR","zh":"zh_CN","ja":"ja_XX","pt":"pt_XX","ru":"ru_RU"
}

@lru_cache(maxsize=1)
def _load_mbart():
    model_name = os.getenv("TRANSLATION_MODEL", "facebook/mbart-large-50-many-to-many-mmt")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, model

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text or src_lang == tgt_lang:
        return text
    tok, model = _load_mbart()
    tok.src_lang = LANG_CODE_MAP.get(src_lang, "en_XX")
    inputs = tok(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids(LANG_CODE_MAP.get(tgt_lang, "en_XX")),
            max_new_tokens=512
        )
    return tok.batch_decode(generated, skip_special_tokens=True)[0]
