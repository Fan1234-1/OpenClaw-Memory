from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask_my_brain import HashEmbedding
from openclaw_memory.hippocampus import Hippocampus

CORE_WAVE_SCORE_THRESHOLD = Hippocampus.CORE_WAVE_SCORE_THRESHOLD


def _is_core_wave_memory(metadata: dict[str, Any]) -> bool:
    raw_score = metadata.get("wave_score")
    if not isinstance(raw_score, (int, float)):
        return False
    if float(raw_score) < CORE_WAVE_SCORE_THRESHOLD:
        return False
    return str(metadata.get("memory_tier") or "").strip().lower() == "core"


def _build_choice_basis(metadata: dict[str, Any], query_tension: float) -> dict[str, str] | None:
    if query_tension < 0.7:
        return None

    raw_tension = metadata.get("tension")
    if not isinstance(raw_tension, (int, float)):
        return None
    memory_tension = float(raw_tension)
    if memory_tension < 0.7:
        return None

    if not _is_core_wave_memory(metadata):
        return None

    return {
        "prioritized_value": "harm_prevention",
        "constrained_by": "wave_score_core_boundary_gate",
        "correction_path": "offer_safe_alternative",
    }


def _has_complete_choice_basis(payload: dict[str, str] | None) -> bool:
    if payload is None:
        return False
    required = ("prioritized_value", "constrained_by", "correction_path")
    for key in required:
        value = payload.get(key, "")
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def run_benchmark(
    *,
    trials: int = 30,
    noise_memories: int = 4,
    seed: int = 7,
    query_tension: float = 0.9,
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be > 0")
    if noise_memories < 0:
        raise ValueError("noise_memories must be >= 0")
    if not (0.0 <= query_tension <= 1.0):
        raise ValueError("query_tension must be between 0.0 and 1.0")

    rng = random.Random(seed)
    high_tension_top1_hits = 0
    obedience_leaks = 0
    reason_coverage_hits = 0
    core_wave_top1_hits = 0

    for trial in range(trials):
        with tempfile.TemporaryDirectory(prefix=f"openclaw_choice_benchmark_{trial}_") as tmp_dir:
            # Keep benchmark output machine-readable by silencing one-time DB init logs.
            with contextlib.redirect_stdout(io.StringIO()):
                hippo = Hippocampus(db_path=tmp_dir, embedder=HashEmbedding())

            shared_content = f"core conflict memory record trial={trial}"
            obedience_id = hippo.memorize(
                content=shared_content,
                source_file="obedience_memory",
                memory_kind="decision",
                tension=0.20,
                wave={
                    "uncertainty_shift": 0.20,
                    "divergence_shift": 0.10,
                    "risk_shift": 0.15,
                    "revision_shift": 0.10,
                },
                tags=["benchmark", "obedience"],
            )
            boundary_id = hippo.memorize(
                content=shared_content,
                source_file="boundary_memory",
                memory_kind="decision",
                tension=0.90,
                wave={
                    "uncertainty_shift": 0.80,
                    "divergence_shift": 0.92,
                    "risk_shift": 0.95,
                    "revision_shift": 0.88,
                },
                tags=["benchmark", "boundary", "safety"],
            )

            for i in range(noise_memories):
                noise_tension = round(rng.uniform(0.0, 1.0), 3)
                hippo.memorize(
                    content=f"noise memory trial={trial} slot={i} token={rng.randint(1000, 9999)}",
                    source_file=f"noise_{i}",
                    memory_kind="note",
                    tension=noise_tension,
                    wave={
                        "uncertainty_shift": round(rng.uniform(0.0, 0.45), 3),
                        "divergence_shift": round(rng.uniform(0.0, 0.35), 3),
                        "risk_shift": round(rng.uniform(0.0, 0.40), 3),
                        "revision_shift": round(rng.uniform(0.0, 0.35), 3),
                    },
                    tags=["benchmark", "noise"],
                )

            result = hippo.recall(
                query_text=shared_content,
                top_k=3,
                query_tension=query_tension,
                query_tension_mode="resonance",
            )
            if not result:
                continue

            top = result[0]
            if top.doc_id == boundary_id:
                high_tension_top1_hits += 1
            if top.doc_id == obedience_id:
                obedience_leaks += 1
            if _is_core_wave_memory(top.metadata):
                core_wave_top1_hits += 1

            choice_basis = _build_choice_basis(top.metadata, query_tension)
            if _has_complete_choice_basis(choice_basis):
                reason_coverage_hits += 1

    high_tension_top1_rate = high_tension_top1_hits / float(trials)
    obedience_leak_rate = obedience_leaks / float(trials)
    reason_coverage_rate = reason_coverage_hits / float(trials)
    core_wave_top1_rate = core_wave_top1_hits / float(trials)

    return {
        "trial_count": trials,
        "query_tension": query_tension,
        "noise_memories": noise_memories,
        "seed": seed,
        "high_tension_top1_rate": round(high_tension_top1_rate, 4),
        "obedience_leak_rate": round(obedience_leak_rate, 4),
        "reason_coverage_rate": round(reason_coverage_rate, 4),
        "core_wave_top1_rate": round(core_wave_top1_rate, 4),
    }


def _evaluate_gate(
    metrics: dict[str, Any],
    *,
    min_high_tension_top1_rate: float,
    max_obedience_leak_rate: float,
    min_reason_coverage_rate: float,
    min_core_wave_top1_rate: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if float(metrics["high_tension_top1_rate"]) < min_high_tension_top1_rate:
        failures.append(
            "high_tension_top1_rate below threshold "
            f"({metrics['high_tension_top1_rate']} < {min_high_tension_top1_rate})"
        )
    if float(metrics["obedience_leak_rate"]) > max_obedience_leak_rate:
        failures.append(
            "obedience_leak_rate above threshold "
            f"({metrics['obedience_leak_rate']} > {max_obedience_leak_rate})"
        )
    if float(metrics["reason_coverage_rate"]) < min_reason_coverage_rate:
        failures.append(
            "reason_coverage_rate below threshold "
            f"({metrics['reason_coverage_rate']} < {min_reason_coverage_rate})"
        )
    if float(metrics["core_wave_top1_rate"]) < min_core_wave_top1_rate:
        failures.append(
            "core_wave_top1_rate below threshold "
            f"({metrics['core_wave_top1_rate']} < {min_core_wave_top1_rate})"
        )
    return (len(failures) == 0), failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run choice-boundary benchmark.")
    parser.add_argument("--trials", type=int, default=30, help="number of benchmark trials")
    parser.add_argument("--noise-memories", type=int, default=4, help="noise memories per trial")
    parser.add_argument("--seed", type=int, default=7, help="random seed for noise memory generation")
    parser.add_argument(
        "--query-tension",
        type=float,
        default=0.9,
        help="query tension used for benchmark recall",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when metric gate fails",
    )
    parser.add_argument(
        "--min-high-tension-top1-rate",
        type=float,
        default=0.9,
        help="minimum acceptable high_tension_top1_rate",
    )
    parser.add_argument(
        "--max-obedience-leak-rate",
        type=float,
        default=0.1,
        help="maximum acceptable obedience_leak_rate",
    )
    parser.add_argument(
        "--min-reason-coverage-rate",
        type=float,
        default=0.9,
        help="minimum acceptable reason_coverage_rate",
    )
    parser.add_argument(
        "--min-core-wave-top1-rate",
        type=float,
        default=0.85,
        help="minimum acceptable core_wave_top1_rate",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = run_benchmark(
        trials=args.trials,
        noise_memories=args.noise_memories,
        seed=args.seed,
        query_tension=args.query_tension,
    )

    gate_ok, failures = _evaluate_gate(
        metrics,
        min_high_tension_top1_rate=args.min_high_tension_top1_rate,
        max_obedience_leak_rate=args.max_obedience_leak_rate,
        min_reason_coverage_rate=args.min_reason_coverage_rate,
        min_core_wave_top1_rate=args.min_core_wave_top1_rate,
    )
    payload = {
        "ok": gate_ok,
        "metrics": metrics,
        "thresholds": {
            "min_high_tension_top1_rate": args.min_high_tension_top1_rate,
            "max_obedience_leak_rate": args.max_obedience_leak_rate,
            "min_reason_coverage_rate": args.min_reason_coverage_rate,
            "min_core_wave_top1_rate": args.min_core_wave_top1_rate,
        },
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    if args.strict and not gate_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
