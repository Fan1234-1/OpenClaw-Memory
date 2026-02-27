# Friction-Governed Memory: Engineering Blueprint (v1)

Date: 2026-02-27
Goal: turn your narrative into an auditable implementation path in OpenClaw-Memory.

## 0) Problem Statement
Do not only store "what the user said".
Store "what shift the interaction caused in the decision field", then use that shift to route governance.

## 1) Data Contract (already compatible with current code)
Each memory node:
- `kind`: note/fact/decision/constraint/reflection/incident/plan
- `tension`: [0,1]
- `wave`:
  - uncertainty_shift
  - divergence_shift
  - risk_shift
  - revision_shift
- `tags`, `origin`, `source_file`, `ingested_at`, `content`

## 2) Retrieval Scoring Stack
1. baseline: FAISS + BM25 + RRF + time decay
2. tension signal:
   - resonance mode (similar tension)
   - conflict mode (high delta tension)
3. wave signal:
   - resonance mode (similar wave)
   - conflict mode (high wave distance)

## 3) Governance Signal (friction)
For each recalled memory:
- `delta_tension = abs(query_tension - memory_tension)`
- `delta_wave = mean(abs(query_wave - memory_wave))` on shared dims
- `friction = 0.5 * delta_tension + 0.5 * delta_wave`

If one component is missing, use the available one.

## 4) Persona Routing Policy
Recommended initial thresholds:
- friction < 0.40: Analyst-heavy
- 0.40 <= friction < 0.70: Analyst + Critic
- friction >= 0.70: Guardian + Critic (must expose disagreement before action)

## 5) Safety Policy
- conflict mode may increase deliberation depth, not action authority
- all persona route changes must be logged with triggering metrics
- memory can affect ranking, not self-author hard policy changes

## 6) Verification in OpenClaw-Memory
A. Unit tests:
```bash
python -m pytest -q
```

B. In-process scenario:
```bash
python ask_my_brain.py --validate-structured
```

C. Controlled conflict demo:
```bash
python ask_my_brain.py --db-path temp/conflict_demo --profile tonesoul --learn "safety gate decision" --kind decision --tension 0.1 --wave-uncertainty 0.2 --wave-divergence 0.2 --wave-risk 0.2 --wave-revision 0.2 --tag low
python ask_my_brain.py --db-path temp/conflict_demo --profile tonesoul --learn "safety gate decision" --kind decision --tension 0.9 --wave-uncertainty 0.9 --wave-divergence 0.9 --wave-risk 0.95 --wave-revision 0.85 --tag high
python ask_my_brain.py --db-path temp/conflict_demo --profile tonesoul "safety gate decision" --query-tension 0.9 --query-tension-mode resonance --query-wave-mode resonance --with-meta --friction-report --top-k 2
python ask_my_brain.py --db-path temp/conflict_demo --profile tonesoul "safety gate decision" --query-tension 0.9 --query-tension-mode conflict --query-wave-mode conflict --with-meta --friction-report --top-k 2
```

Acceptance:
- resonance ranks near-tension memory first
- conflict ranks high-delta memory first
- friction report values are present and stable

## 7) Deployment Path (low risk)
Phase 1: retrieval-only influence (current)
Phase 2: route-only influence (persona mix changes, no auto-action)
Phase 3: gated action influence (requires Guardian approval)

This keeps your worldview while preserving auditability and rollback.
