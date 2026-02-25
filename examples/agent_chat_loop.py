import os
from openclaw_memory.hippocampus import Hippocampus
from openclaw_memory.embeddings import SentenceTransformerEmbedding

def mock_llm_response(prompt: str) -> str:
    """Mock LLM API call for demonstration purposes."""
    if "lobster" in prompt.lower():
        return "Ah, I see from my subconscious memory that lobsters utilize a stomatogastric nervous system, which inspires my decentralized edge-computing architecture!"
    return "I do not have enough specific ancestral memory to provide a deep philosophical answer."

def main():
    print("🦞 OpenClaw Agent Boot Sequence...")
    db_path = "./memory_base"
    
    if not os.path.exists(db_path):
        print(">> No memory base found. I am a blank slate. Please run scripts/ingest_ancestral_memory.py first.")
        return
        
    embedder = SentenceTransformerEmbedding()
    subconscious = Hippocampus(db_path=db_path, embedder=embedder)
    
    print(">> Subconscious loaded. I am ready to converse.")
    print(">> Type 'quit' to sleep.\n")
    
    while True:
        try:
            user_input = input("User >> ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # 1. RAG Retrieve from Subconscious
            memories = subconscious.recall(user_input, top_k=2)
            
            # 2. Build Context Prompt
            context = "\n".join([f"- {m.content}" for m in memories])
            system_prompt = f"System Context (Subconscious Memory):\n{context}\n\nUser Question: {user_input}"
            
            # 3. Stream LLM Response
            response = mock_llm_response(system_prompt)
            print(f"OpenClaw >> {response}\n")
            
        except KeyboardInterrupt:
            break
            
if __name__ == "__main__":
    main()
