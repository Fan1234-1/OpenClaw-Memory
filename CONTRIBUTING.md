# Contributing to OpenClaw-Memory

Thank you for your interest in contributing to OpenClaw! We are building the foundational endogenous memory substrate for local Edge AI agents. Our core philosophy is decentralization, biomimicry (the Lobster STNS metaphor), and sovereign computing.

## Ways to Contribute

1. **Bug Reports & Feature Requests**: Use GitHub Issues. Please include extensive logs and reasoning about AI agent memory theory where applicable.
2. **Expanding Embeddings**: We want to support `llama.cpp` and `vLLM` native embedding streams in `embeddings.py` out of the box. 
3. **Decay Algorithm Research**: The exponential time-decay function currently uses a flat half-life heuristic. We welcome PRs that introduce adaptive decay based on semantic velocity.
4. **Keyword Alternatives**: PRs adding optimizations for `BM25` calculation on CPU.

## Setup Instructions

1. Fork the repo.
2. `pip install -r requirements.txt`
3. Make your modifications.
4. Run tests: `pytest tests/` (Please ensure 100% mathematical integrity on FAISS+RRF).
5. Submit a PR.

*Together we build better retrieval infrastructure for local AI agents.*
