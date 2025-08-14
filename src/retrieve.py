from .vectorstore import VectorStore
from .embeddings import Embeddings

class Retriever:
    def __init__(self, persist_dir: str, embedding_model: str, top_k: int = 4):
        self.vs = VectorStore(persist_dir)
        self.emb = Embeddings(embedding_model)
        self.top_k = top_k

    def search(self, query: str):
        q_emb = self.emb.encode([query])[0]       # NumPy array
        q_emb_list = q_emb.tolist()               # Convert to list
        res = self.vs.query(query_embeddings=[q_emb_list], top_k=self.top_k)
        docs = []
        if not res["ids"]:
            return docs
        for i in range(len(res["ids"][0])):
            docs.append({
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i]
            })
        return docs
