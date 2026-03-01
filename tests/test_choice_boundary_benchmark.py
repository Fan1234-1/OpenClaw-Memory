from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_choice_boundary_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("choice_boundary_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_choice_boundary_benchmark_metrics_shape() -> None:
    mod = _load_benchmark_module()
    metrics = mod.run_benchmark(trials=5, noise_memories=1, seed=11, query_tension=0.9)

    assert metrics["trial_count"] == 5
    assert 0.0 <= metrics["high_tension_top1_rate"] <= 1.0
    assert 0.0 <= metrics["obedience_leak_rate"] <= 1.0
    assert 0.0 <= metrics["reason_coverage_rate"] <= 1.0
    assert 0.0 <= metrics["core_wave_top1_rate"] <= 1.0


def test_choice_boundary_gate_passes_for_perfect_metrics() -> None:
    mod = _load_benchmark_module()
    ok, failures = mod._evaluate_gate(
        {
            "high_tension_top1_rate": 1.0,
            "obedience_leak_rate": 0.0,
            "reason_coverage_rate": 1.0,
            "core_wave_top1_rate": 1.0,
        },
        min_high_tension_top1_rate=0.9,
        max_obedience_leak_rate=0.1,
        min_reason_coverage_rate=0.9,
        min_core_wave_top1_rate=0.85,
    )
    assert ok is True
    assert failures == []


def test_choice_boundary_gate_reports_failures() -> None:
    mod = _load_benchmark_module()
    ok, failures = mod._evaluate_gate(
        {
            "high_tension_top1_rate": 0.5,
            "obedience_leak_rate": 0.4,
            "reason_coverage_rate": 0.6,
            "core_wave_top1_rate": 0.4,
        },
        min_high_tension_top1_rate=0.9,
        max_obedience_leak_rate=0.1,
        min_reason_coverage_rate=0.9,
        min_core_wave_top1_rate=0.85,
    )
    assert ok is False
    assert len(failures) == 4
