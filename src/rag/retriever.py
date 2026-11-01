"""
Retrieves relevant chunks from the FAISS index.
Enforces document precedence (current policy > agent > help > faq > marketing).
"""
import pickle
from dataclasses import dataclass
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PATHS, EMBEDDING_MODEL, TOP_K, DOC_TYPE_PRECEDENCE
from src.rag.ingest import KBChunk
from src.rag.index import INDEX_FILE, META_FILE

@dataclass
class RetrievedChunk:
    chunk: KBChunk
    score: float

class Retriever:
    def __init__(self):
        if not INDEX_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError("Index not found. Run `python -m src.rag.index` first.")
            
        # Load FAISS and metadata
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(META_FILE, "rb") as f:
            self.chunks: List[KBChunk] = pickle.load(f)
            
        # Load the exact same embedding model used for indexing
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Map doc_types to their priority (0 is highest)
        self.precedence_map = {dt: i for i, dt in enumerate(DOC_TYPE_PRECEDENCE)}

    def search(self, query: str, k: int = TOP_K) -> List[RetrievedChunk]:
        """
        Search the index, retrieve top-k chunks, and apply precedence rules.
        """
        # 1. Embed the query
        query_vector = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        
        # 2. Search FAISS (get double the candidates so we can filter down later)
        distances, indices = self.index.search(query_vector, k * 2)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: # FAISS returns -1 if not enough results
                continue
                
            chunk = self.chunks[idx]
            
            # Drop superseded policies immediately
            if chunk.status == "superseded":
                continue
                
            # FAISS L2 distance: lower is closer/better. 
            # We convert it to a similarity score (higher is better) just for display.
            score = 1.0 / (1.0 + float(dist))
            
            results.append(RetrievedChunk(chunk=chunk, score=score))
            
        # 3. Apply Precedence Sorting
        # Sort primarily by doc_type priority (policy first), then by similarity score.
        results.sort(
            key=lambda x: (
                self.precedence_map.get(x.chunk.doc_type, 99), # 99 = unknown types go last
                -x.score  # Negative because we want highest score first
            )
        )
        
        # Return only the top K after sorting
        return results[:k]

# Singleton instance for easy importing
_retriever = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever