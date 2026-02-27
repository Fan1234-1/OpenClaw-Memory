# OpenClaw-Memory

Lightweight local hybrid memory retrieval for AI agents.

- Baseline retrieval: FAISS + BM25 + RRF + time decay
- No external vector DB required
- Works offline with local files (`.index` + `.jsonl`)

## 1-Minute Quick Start

```bash
git clone https://github.com/Fan1234-1/OpenClaw-Memory.git
cd OpenClaw-Memory
pip install -r requirements.txt
```

### Ingest memory from markdown files

```bash
python scripts/ingest_ancestral_memory.py --source ./my_docs --db-path ./memory_base
```

### Query memory (no model download required)

`ask_my_brain.py` now uses a deterministic local hash embedder by default, so first-time users can run immediately:

```bash
python ask_my_brain.py "what did we decide about deployment rollback?"
```

## Two Profiles: OpenClaw vs ToneSoul

### 1) OpenClaw baseline (for general users)

```bash
python ask_my_brain.py --profile openclaw "rollback policy"
```

### 2) ToneSoul mode (tension-aware)

```bash
python ask_my_brain.py --profile tonesoul --learn "high-risk deploy should degrade safely first" --tension 0.82 --tag safety
python ask_my_brain.py --profile tonesoul "deployment policy" --query-tension 0.8 --top-k 3
```

### Quick difference check

```bash
python ask_my_brain.py --why-tonesoul
```

## What Is Different in ToneSoul

| Capability | OpenClaw Baseline | ToneSoul Extension |
|---|---|---|
| Core retrieval | FAISS + BM25 + RRF | Same |
| Metadata | id/source/content/time | + `kind` + `tension` + `tags` + `wave` |
| Ranking | Relevance + time decay | + tension resonance + wave resonance |
| Philosophy | pure retrieval | make disagreement/pressure visible |

Tension resonance formula:

```text
resonance = 1 - abs(query_tension - memory_tension)
final_score = rrf_score * (1 + 0.20 * clamp(resonance, 0, 1))
```

Tension conflict formula (for high delta retrieval):

```text
delta = abs(query_tension - memory_tension)
final_score = rrf_score * (1 + 0.20 * clamp(delta, 0, 1))
```

Wave resonance formula:

```text
distance = mean(abs(query_wave[i] - memory_wave[i])) over shared dimensions
resonance = 1 - distance
final_score = score_after_tension * (1 + 0.25 * clamp(resonance, 0, 1))
```

## Common CLI

```bash
# ingest one memory
python ask_my_brain.py --learn "we should add canary rollback" --tension 0.75 --tag release

# ingest from file (split by blank lines)
python ask_my_brain.py --memory-file docs/TENSION_MEMORY_UPGRADE_2026.md --tension 0.80 --tag architecture

# query as json
python ask_my_brain.py "rollback" --top-k 5 --json
```

## Structured Memory CLI (AI-Meaningful)

```bash
# store memory kind + wave vector
python ask_my_brain.py --profile tonesoul --learn "guardian blocked unsafe release" \
  --kind incident --tension 0.90 \
  --wave-uncertainty 0.82 --wave-divergence 0.88 --wave-risk 0.95 --wave-revision 0.70 \
  --tag safety

# query with wave target and inspect metadata
python ask_my_brain.py --profile tonesoul "release decision" \
  --query-tension 0.85 \
  --query-wave-mode resonance \
  --query-wave-uncertainty 0.80 --query-wave-divergence 0.85 --query-wave-risk 0.95 --query-wave-revision 0.75 \
  --with-meta --top-k 3

# retrieve "useful contradiction" memories (high tension delta)
python ask_my_brain.py --profile tonesoul "release decision" \
  --query-tension 0.85 --query-tension-mode conflict --query-wave-mode conflict \
  --with-meta --top-k 3
```

## How To Validate In OpenClaw-Memory

```bash
# 1) Unit tests (includes wave resonance and persistence checks)
python -m pytest -q

# 2) In-process validation scenario (no external model/download)
python ask_my_brain.py --validate-structured

# 3) Manual A/B sanity check
python ask_my_brain.py --profile openclaw "deployment decision memory" --top-k 2
python ask_my_brain.py --profile tonesoul "deployment decision memory" \
  --query-tension 0.9 \
  --query-wave-uncertainty 0.9 --query-wave-divergence 0.9 --query-wave-risk 0.95 --query-wave-revision 0.85 \
  --with-meta --friction-report --top-k 2
```

Engineering blueprint:
- `docs/ENGINEERING_BLUEPRINT_FRICTION_GOVERNANCE.md`

## Security Notes

- Retrieval-time network calls: none
- Memory DB path traversal is rejected
- Metadata includes `origin` for provenance tracking

## Optional Local Embeddings

If you want semantic embeddings from `sentence-transformers`, install:

```bash
pip install sentence-transformers
```

## References (2025-2026 memory systems)

- Mem0 paper (2025): https://arxiv.org/abs/2504.19413
- MemOS paper (2025): https://arxiv.org/abs/2507.03724
- SeCom paper (2025): https://arxiv.org/abs/2502.05589
- LoCoMo-Plus benchmark (2026): https://arxiv.org/abs/2602.10715

- Mem0 repo: https://github.com/mem0ai/mem0
- Letta repo: https://github.com/letta-ai/letta
- LangMem repo: https://github.com/langchain-ai/langmem
- MemOS repo: https://github.com/MemTensor/MemOS

## License

Apache-2.0
