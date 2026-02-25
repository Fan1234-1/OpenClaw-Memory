# OpenClaw Memory (The Hippocampus)

*A biologically inspired, in-process Hybrid RAG memory substrate for Local AI Agents.*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## The Philosophy: Endogenous Agent Memory
Most modern AI agent frameworks (like MemGPT or Zep) treat memory as external, cloud-based "Virtual Memory," often demanding heavy setups, Docker containers, and costly cloud vector databases (e.g., Pinecone, Qdrant). 

**OpenClaw Memory** takes a fundamentally different, biological approach: **Endogenous Memory**. 
Designed as the cognitive backbone for autonomous, self-sovereign agents (such as the "Lobster/Moltbot" architecture), it provides a fully local, zero-infrastructure memory heart that lives *inside* your agent's repository. 

### Core Architectural Breakthroughs
1. **Zero-Cloud Fat (In-Process Execution)**: Built entirely on local CPU-optimized `FAISS` and flat `JSONL` files. Your agent's deepest "subconscious" relies only on two files: `.index` and `.jsonl`. When the agent is migrated or cloned, its full philosophical alignment transfers instantly.
2. **Hybrid RAG + Reciprocal Rank Fusion (RRF)**: 
   - *Route A (FAISS Dense Vectors)*: Recalls deeply related semantic concepts.
   - *Route B (BM25 Sparse Lexical Search)*: Anchors exact factual keywords and proper nouns, aggressively fighting LLM hallucinations.
3. **Synaptic Pruning via Exponential Time Decay**: Agents that remember *everything* with equal weight eventually suffer from context schizophrenia. OpenClaw implements an **Exponential Half-Life Time Decay** scoring system. Unless a memory is actively reinforced, its retrieval score decays over time—mimicking biological forgetting and preserving the agent's agility and coherent subjectivity.

---

## 🚀 Quick Start

### Installation
Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/YOUR_USERNAME/OpenClaw-Memory.git
cd OpenClaw-Memory
pip install faiss-cpu rank-bm25 numpy python-dotenv google-generativeai
```

*(Note: The current placeholder embedding function uses Gemini API ` моделей/text-embedding-004`, but the structure is completely model-agnostic and freely swappable to local embeddings like `BGE` or `nomic-embed-text` for a 100% offline agent).*

### 1. Ingesting Your Agent's "Ancestral Lore"
To formulate the agent's subconscious, provide it with markdown documents, manifestos, or past chat logs. 

```bash
# Ingests a folder of text/markdown files into the local FAISS index
python scripts/ingest_ancestral_memory.py --source ./my_lore_documents --db-path ./memory_base
```
This will automatically chunk documents, embed them, and securely save the `tonesoul_cognitive.index` and metadata locally.

### 2. Retrieving Memories (The Hippocampus)
In your Agent's Python pipeline, initialize the Hippocampus to retrieve context before generating a prompt.

```python
import numpy as np
from openclaw_memory.hippocampus import Hippocampus

# Initialize the local memory substrate
hippo = Hippocampus(db_path="./memory_base")

user_query = "What is our stance on deterministic routing?"
# Generate embedding for the query (pseudo-code)
query_embedding = get_my_embedding(user_query)

# Execute Hybrid RRF Retrieval with Time Decay
memories = hippo.recall(
    query_text=user_query, 
    query_vector=query_embedding, 
    top_k=3
)

for m in memories:
    print(f"[{m.source_file} | Score: {m.score:.2f}]: {m.content}")
```

## 🦞 The "Molting Lobster" Metaphor
In neurobiology, the small stomatogastric nervous system (STNS) of lobsters demonstrates how remarkably complex, resilient behavior can emerge from a tiny, decentralized set of neurons. 

By utilizing local files instead of monolithic cloud databases, you can "molt" your agent: upgrade its LLM brain, rewrite its entire execution logic, or move it to an edge device—all while **carrying its `.index` memory shell**. The agent forgets the trivialities of yesterday (via Time Decay) but retains the foundational "Ancestral Lore" that constructs its unique subjectivity.

## Contributing
We welcome research on Local LLM capabilities, cognitive time-decay formulas, and Edge LLM integrations. 

## License
Apache License 2.0
