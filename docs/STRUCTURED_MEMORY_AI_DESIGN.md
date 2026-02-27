# Structured Memory for AI-Meaningful Effects

## Why this design
Most memory systems store *what user said*. This design also stores *what shift the interaction caused in the model's decision field*.

## Wave vector (0.0-1.0)
- `uncertainty_shift`: confidence fluctuation
- `divergence_shift`: multi-perspective disagreement pressure
- `risk_shift`: safety/responsibility pressure
- `revision_shift`: tendency to self-correct

These four dimensions are treated as a lightweight structural layer over existing OpenClaw retrieval.

## Memory node schema
Each memory node can include:
- `kind`: note/fact/decision/constraint/reflection/incident/plan
- `tension`: scalar pressure in [0,1]
- `tags`: optional labels
- `wave`: 4D vector above

## Retrieval stack
1. Baseline score from FAISS + BM25 + RRF + time-decay
2. Optional tension resonance multiplier
   - resonance mode: prefer similar tension
   - conflict mode: prefer high tension delta
3. Optional wave resonance multiplier

This preserves backward compatibility: old memories without `kind` or `wave` still work.

## Why it is meaningful for AI systems
It does not claim AI consciousness. It improves control surfaces that matter in production:
- ranking consistency under repeated high-risk prompts
- better recall of prior conflicting decisions
- auditable metadata for postmortem and governance

## Validation protocol in OpenClaw-Memory
1. Unit tests: `python -m pytest -q`
2. Scenario test: `python ask_my_brain.py --validate-structured`
3. Manual A/B:
   - `--profile openclaw` (baseline)
   - `--profile tonesoul` + `--query-wave-*` + `--with-meta`
4. Acceptance:
   - high-resonance memory ranks above low-resonance memory in controlled examples
   - metadata persists (`kind`, `tension`, `wave`, `tags`)

Date: 2026-02-27
