# OpenClaw vs ToneSoul (Practical Guide)

## Baseline for Everyone
Use OpenClaw profile when you want straightforward retrieval without extra policy assumptions.

```bash
python ask_my_brain.py --profile openclaw "my query"
```

## ToneSoul Layer
Use ToneSoul profile when you need memory retrieval to preserve semantic tension and responsibility context.

```bash
python ask_my_brain.py --profile tonesoul --learn "high-risk change should degrade safely" --tension 0.82 --tag safety
python ask_my_brain.py --profile tonesoul "release strategy" --query-tension 0.8
python ask_my_brain.py --profile tonesoul "release strategy" --query-tension 0.8 --query-tension-mode conflict
```

## Design Principle
- Keep baseline simple and boring-correct.
- Add ToneSoul behavior as a visible, auditable layer.
- Do not hide disagreements; rank them explicitly.

## Why This Matters
General users can adopt OpenClaw immediately.
Teams needing governance can switch to ToneSoul without replacing core retrieval.

Date: 2026-02-27
