# OpenClaw-Memory: Endogenous Hybrid RAG for Local AI Agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**OpenClaw-Memory** is a lightweight, in-process hybrid retrieval framework designed for local, privacy-first AI agents. It serves as a persistent, autonomous memory substrate that operates entirely without external cloud vector database dependencies (e.g., Pinecone, Qdrant). 

By combining dense semantic search (`FAISS`) with sparse lexical matching (`BM25`) and temporal decay algorithms, OpenClaw-Memory offers state-of-the-art context retrieval designed explicitly for agentic sovereignty and continuous learning.

## 🎯 Key Design Principles

1. **In-Process Endogenous Architecture**
   Instead of delegating memory storage to heavy, external microservices (as seen in frameworks like MemGPT or Zep), OpenClaw-Memory relies exclusively on highly optimized local files (`FAISS .index` arrays and `JSONL` metadata). Agents configured with OpenClaw can be migrated, scaled, or run fully offline at the edge while retaining 100% of their historical context.

2. **Hybrid RAG & Reciprocal Rank Fusion (RRF)**
   Relying solely on dense embeddings often leads to semantic blur and hallucinations when retrieving specific identifiers or code definitions. OpenClaw implements an **RRF (Reciprocal Rank Fusion)** strategy to combine:
   - **Route A (Semantic Space)**: `FAISS` Inner-Product similarity search for conceptual alignment.
   - **Route B (Lexical Anchor)**: `BM25` Okapi scoring to ensure exact-match retrieval of domain-specific terminology.

3. **Temporal Context Management (Exponential Time Decay)**
   AI Agents that persist memory indefinitely without pruning suffer from context degradation and contradictory alignment. Inspired by biological neuromodulation, OpenClaw utilizes an **Exponential Half-Life Decay Engine**. Memory items lose rank over time (e.g., a default 69-day half-life) unless they are highly resonant, ensuring the agent remains adaptive to current operational goals without bloated context windows.

---

## 🚀 Quick Start

### Installation

Clone the repository and install the minimal dependencies:

```bash
git clone https://github.com/Fan1234-1/OpenClaw-Memory.git
cd OpenClaw-Memory
pip install -r requirements.txt
```

For local embedding support (recommended), also install:

```bash
pip install sentence-transformers
```

*(Note: The ingestion script uses `sentence-transformers` with the `all-MiniLM-L6-v2` model by default. The architecture is strictly decoupled and allows for seamless integration with other embedding models like `BGE-m3` or any OpenAI-compatible API.)*

### 1. Ingesting Long-Term Memory (Context Building)

Provide the agent with initial rule sets, system architectures, or historical markdown logs to build its structural memory base.

```bash
# Parse markdown files from the target directory and serialize them into FAISS
python scripts/ingest_ancestral_memory.py --source ./my_agent_docs --db-path ./memory_base
```
*Artifacts created: `memory_base/tonesoul_cognitive.index` and `memory_base/tonesoul_metadata.jsonl`.*

### 2. Implementation in Agentic Workflows (Retrieval)

Instantiate the `Hippocampus` module directly within your primary Agent runtime.

```python
import numpy as np
from openclaw_memory.hippocampus import Hippocampus

# 1. Initialize the endogenous memory module
memory_core = Hippocampus(db_path="./memory_base")

# 2. Define the input query
user_query = "What is the defined protocol for the Wei Xiaobao routing architecture?"

# 3. Procure embedding (Implementation specific to your embedding model)
query_embedding = get_embedding(user_query) 

# 4. Perform Hybrid RRF Retrieval with Time-Decay heuristics
results = memory_core.recall(
    query_text=user_query, 
    query_vector=query_embedding, 
    top_k=3
)

# 5. Inject retrieved context into the LLM system prompt
for memory in results:
    print(f"[{memory.source_file} | Relevance Score: {memory.score:.3f}]:\n{memory.content}\n")
```

---

## 🔒 Security & Design Clarifications

> **This is a retrieval library, NOT an agent instruction set or prompt injection framework.**

The following clarifications are provided for transparency:

| Concern | Clarification |
|---|---|
| **Auto-install / Prompt Injection** | This README contains standard installation documentation only. There are no hidden instructions, embedded agent directives, or automated execution triggers. |
| **Network Calls** | OpenClaw-Memory makes **zero network calls** at retrieval time. All operations (FAISS search, BM25 scoring, time-decay) run entirely in-process on local files. |
| **File Access Scope** | The `Hippocampus` module only reads from its configured `db_path` directory. Path traversal attacks are rejected at initialization via `os.path.abspath()` validation. All file operations are **read-only** during retrieval. |
| **Data Provenance** | Each ingested memory chunk includes an `origin` metadata field (e.g., `"user_document"`) to distinguish user-provided content from AI-generated content, preventing hallucination feedback loops. |
| **Biomimicry Language** | Terms like "Hippocampus" and "Lobster Paradigm" are **naming metaphors** inspired by neuroscience research, not claims of sentience or consciousness. The underlying implementation is standard information retrieval (FAISS inner-product search + BM25 Okapi + RRF fusion). |

---

## 🦞 The "Lobster" Biomimicry Paradigm

This architecture draws inspiration from the *stomatogastric nervous system (STNS)* of lobsters—often studied in neuromorphic engineering for its ability to produce highly resilient, complex behavior from a decentralized, localized neural cluster rather than a monolithic brain. 

Similarly, OpenClaw decentralizes the AI memory constraint. By shedding ("molting") the cloud database dependencies and keeping memory purely local, the agent can be recompiled, redeployed, or upgraded with new language models independently of its persistent identity storage.

> **Note:** This is a design metaphor for architectural decision-making, not a claim about AI consciousness. The core algorithm is Reciprocal Rank Fusion — a well-established technique in information retrieval research ([Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)).

## 🤝 Contributing
Open source contributions are encouraged. Key areas for research include:
- Optimizing CPU-bound execution for alternative sparse search libraries.
- Fine-tuning the exponential decay algorithms for varying operational timeframes.
- Integrations with popular local execution engines (Ollama, vLLM, llama.cpp).

## 📄 License
This project is licensed under the [Apache License 2.0](LICENSE). Copyright 2024-2025.
