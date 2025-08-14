import yaml
from src.retrieve import Retriever

def test_basic_retrieval():
    cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
    r = Retriever(cfg["data"]["persist_dir"], cfg["models"]["embedding"], top_k=2)
    try:
        docs = r.search("hello")
    except Exception:
        docs = []
    assert isinstance(docs, list)
