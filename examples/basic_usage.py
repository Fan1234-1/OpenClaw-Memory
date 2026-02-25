import os
from openclaw_memory.hippocampus import Hippocampus
from openclaw_memory.embeddings import SentenceTransformerEmbedding

def main():
    # Example snippet showing basic usage of OpenClaw-Memory
    print("🦞 Initializing OpenClaw Memory Substrate...")
    
    # In a real environment, you would use scripts/ingest_ancestral_memory.py first
    # to construct the `memory_base` directory.
    db_path = "./memory_base"
    if not os.path.exists(db_path):
        print(f"Warning: {db_path} not found. Returning empty results.")
        print("Run `python scripts/ingest_ancestral_memory.py` to index documents first!")
        return

    # 1. Initialize Embedder (Downloads lightweight all-MiniLM-L6-v2 if not installed)
    embedder = SentenceTransformerEmbedding()
    
    # 2. Initialize Memory
    memory_core = Hippocampus(db_path=db_path, embedder=embedder)
    
    # 3. Query
    query = "What is the core philosophy of OpenClaw?"
    print(f"\nQuerying: '{query}'")
    
    # 4. Recall
    results = memory_core.recall(query_text=query, top_k=3)
    
    for res in results:
        print("\n" + "="*50)
        print(f"Relevance Score: {res.score:.3f}")
        print(f"Source: {res.source_file}")
        print(f"Content: {res.content}")

if __name__ == "__main__":
    main()
