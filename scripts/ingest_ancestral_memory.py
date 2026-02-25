import os
import glob
import json
import uuid
import numpy as np
import faiss
from datetime import datetime
from pathlib import Path

# Real local embedding using SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    # Using a fast, lightweight local model (downloads automatically if not present)
    model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    model = None
    print("WARNING: sentence-transformers not installed. Fallback empty embeddings.")

def get_embedding(text: str) -> np.ndarray:
    if model is None:
        return np.zeros(384, dtype=np.float32)
    # Generate real embedding
    vec = model.encode(text)
    # Normalize for inner product (cosine similarity)
    vec = vec / np.linalg.norm(vec)
    return vec.astype(np.float32)

def chunk_markdown(content: str, max_words=200) -> list[str]:
    paragraphs = content.split("\n\n")
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk.split()) + len(p.split()) < max_words:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c.strip()) > 10]

def ingest_directory(dir_path: str, memory_db_path: str):
    # Setup FAISS flat index
    dimension = 384  # Updated to match all-MiniLM-L6-v2
    index_file = os.path.join(memory_db_path, "tonesoul_cognitive.index")
    meta_file = os.path.join(memory_db_path, "tonesoul_metadata.jsonl")
    
    os.makedirs(memory_db_path, exist_ok=True)
    
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatIP(dimension) # Inner Product matching

    vectors = []
    doc_count = 0
    
    search_pattern = os.path.join(dir_path, "**", "*.md")
    
    with open(meta_file, 'a', encoding='utf-8') as meta_out:
        for filepath in glob.glob(search_pattern, recursive=True):
            print(f"Ingesting {filepath}...")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = chunk_markdown(content)
                for chunk in chunks:
                    doc_id = str(uuid.uuid4())
                    vector = get_embedding(chunk)
                    vectors.append(vector)
                    
                    # Store exact timestamp for time-decay hybrid RAG algorithm
                    meta = {
                        "id": doc_id,
                        "source_file": filepath,
                        "content": chunk,
                        "ingested_at": datetime.utcnow().isoformat()
                    }
                    meta_out.write(json.dumps(meta) + "\n")
                    doc_count += 1
                    
            except Exception as e:
                print(f"Failed to parse {filepath}: {e}")

    if vectors:
        vector_matrix = np.array(vectors, dtype=np.float32)
        index.add(vector_matrix)
        faiss.write_index(index, index_file)
        
    print(f"✅ Inserted {doc_count} memory chunks into FAISS db at {memory_db_path}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="temp", help="Directory to parse (e.g. temp)")
    parser.add_argument("--db-path", type=str, default="./memory_base", help="Path to the faiss storage directory")
    args = parser.parse_args()
    
    ingest_directory(args.source, args.db_path)
