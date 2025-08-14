import os, glob, uuid
from dotenv import load_dotenv
from .embeddings import Embeddings
from .vectorstore import VectorStore
from .utils.chunking import split_into_chunks
from .utils.lang import detect_lang
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

load_dotenv()

def _iter_files(input_dir: str):
    for pat in ("*.txt", "*.md"):
        yield from glob.glob(os.path.join(input_dir, pat))

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def build_index(input_dir: str, persist_dir: str, embedding_model: str, chunk_size: int, chunk_overlap: int):
    os.makedirs(persist_dir, exist_ok=True)
    vs = VectorStore(persist_dir)
    emb = Embeddings(embedding_model)

    for path in _iter_files(input_dir):
        text = _read(path)
        src_lang = detect_lang(text[:8000])
        chunks = split_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            continue
        ids = [str(uuid.uuid4()) for _ in chunks]
        # embs = emb.encode(chunks)
        embs = emb.encode(chunks)
        embs_list = [e.tolist() for e in embs]  # Convert each to list
        vs.add(ids=ids, embeddings=embs_list, metadatas=metadatas, documents=chunks)

        metadatas = [{"source": os.path.basename(path), "lang": src_lang, "chunk_index": i} for i,_ in enumerate(chunks)]
        vs.add(ids=ids, embeddings=embs, metadatas=metadatas, documents=chunks)
        print(f"Indexed {len(chunks)} chunks from {path} (lang={src_lang})")

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
    build_index(
        input_dir=cfg["data"]["input_dir"],
        persist_dir=cfg["data"]["persist_dir"],
        embedding_model=cfg["models"]["embedding"],
        chunk_size=cfg["retrieval"]["chunk_size"],
        chunk_overlap=cfg["retrieval"]["chunk_overlap"],
    )
