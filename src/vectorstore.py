import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_dir: str):
        # Persistent client (stores on disk)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space":"cosine"}
        )

    def reset(self):
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space":"cosine"}
        )

    def add(self, ids, embeddings, metadatas, documents):
        self.collection.add(
            ids=ids, embeddings=embeddings,
            metadatas=metadatas, documents=documents
        )

    def query(self, query_embeddings, top_k=4):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
