import os
from typing import List, Dict
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QAInput(BaseModel):
    query: str
    passages: List[Dict]  # {text, source_lang, metadata}

# Load once (can switch to "google/mt5-base")
_MODEL_NAME = os.getenv("GENERATION_MODEL", "google/mt5-small")
_TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME, legacy=False)

# _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
_MODEL = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)

def generate_answer(inputs: QAInput, answer_lang: str) -> str:
    """
    Generate an answer using ONLY local mT5, grounded on provided passages.
    """
    context = "\n\n".join([f"[Doc {i+1}] {p['text']}" for i, p in enumerate(inputs.passages)])

    prompt = f"""
Answer the question using ONLY the given context.
If the answer is not in the context, say you don't know.
Keep cultural names as they are, and cite sources as [Doc X].
Question: {inputs.query}

Context:
{context}

The answer must be in ISO language code: {answer_lang}
"""
    enc = _TOKENIZER(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = _MODEL.generate(**enc, max_new_tokens=512)
    return _TOKENIZER.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # return _TOKENIZER.decode(out[0], skip_special_tokens=True)
