"""
Builds and saves the FAISS index.
"""
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from src.config import PATHS, EMBEDDING_MODEL
from src.rag.ingest import process_kb_files, KBChunk

INDEX_FILE = PATHS.index / "kb.faiss"
META_FILE = PATHS.index / "kb_meta.pkl"

def build_index():
    """Processes KB, generates embeddings, and saves FAISS + metadata to disk."""
    print("Processing KB files...")
    chunks = process_kb_files()
    if not chunks:
        raise ValueError("No chunks were extracted from the KB.")
    print(f"Extracted {len(chunks)} chunks.")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Extract just the text for embedding
    texts = [chunk.text for chunk in chunks]
    
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # FAISS requires float32
    embeddings = np.array(embeddings, dtype=np.float32)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, str(INDEX_FILE))
    
    # Save metadata chunks (parallel array)
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)
        
    print(f"Saved index to {INDEX_FILE} and metadata to {META_FILE}")

if __name__ == "__main__":
    build_index()