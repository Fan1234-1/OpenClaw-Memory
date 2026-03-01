from ask_my_brain import _compute_friction_summary, _run_tension_replay_validation


def test_tension_replay_validation_passes() -> None:
    assert _run_tension_replay_validation()


def test_friction_summary_includes_wave_score_delta() -> None:
    friction = _compute_friction_summary(
        {
            "tension": 0.2,
            "wave_score": 0.9,
            "wave": {
                "uncertainty_shift": 0.1,
                "divergence_shift": 0.1,
                "risk_shift": 0.1,
                "revision_shift": 0.1,
            },
        },
        query_tension=0.9,
        query_wave={
            "uncertainty_shift": 0.9,
            "divergence_shift": 0.9,
            "risk_shift": 0.9,
            "revision_shift": 0.9,
        },
    )
    assert friction is not None
    assert friction["wave_score_delta"] == 0.0
    assert friction["friction"] >= 0.0
