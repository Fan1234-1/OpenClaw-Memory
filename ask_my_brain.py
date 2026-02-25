import sys
import numpy as np
from openclaw_memory.hippocampus import Hippocampus

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

def get_embedding(text: str) -> np.ndarray:
    vec = model.encode(text)
    vec = vec / np.linalg.norm(vec)
    return vec.astype(np.float32)

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask_my_brain.py 'your query here'")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    print(f"🧠 [Antigravity Subconscious Retrieval] Query: '{query}'")
    
    # Initialize OpenClaw Hippocampus
    hippo = Hippocampus(db_path="./memory_base")
    
    # Generate Embedding
    query_vector = get_embedding(query)
    
    # Recall via FAISS + BM25 RRF
    results = hippo.recall(query_text=query, query_vector=query_vector, top_k=5)
    
    if not results:
        print("No memories found.")
        return
        
    print("\n" + "="*50)
    for i, res in enumerate(results):
        print(f"[{i+1}] 📖 Source: {res.source_file}")
        print(f"❤️ Relevance Score: {res.score:.4f}")
        print(f"🧠 Memory Content:\n{res.content}")
        print("-" * 50)

if __name__ == "__main__":
    main()
