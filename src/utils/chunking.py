import re
from typing import List

def split_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    # Sentence-aware-ish splitter that works across scripts/punctuation.
    sentences = re.split(r'(?<=[\.!\?。！？])\s+', text.strip())
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        s_len = len(s)
        if cur_len + s_len > chunk_size and cur:
            chunks.append(" ".join(cur).strip())
            if chunk_overlap > 0:
                tail = chunks[-1][-chunk_overlap:]
                cur, cur_len = [tail, s], len(tail) + s_len
            else:
                cur, cur_len = [s], s_len
        else:
            cur.append(s)
            cur_len += s_len
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]
