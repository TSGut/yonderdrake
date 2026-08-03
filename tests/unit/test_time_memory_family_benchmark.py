"""Structural tests for the cross-family time-memory benchmark."""

from benchmarks.time_memory_families import (
    DEFAULT_BASELINE_MODES,
    DEFAULT_FINAL_TIME,
    DEFAULT_STEP_SIZE,
    _candidate_families,
)


def test_cost_matched_benchmark_covers_every_time_memory_family() -> None:
    families = _candidate_families(
        baseline_modes=41,
        step_size=0.025,
        final_time=1.0,
        smoke=False,
    )

    assert set(families) == {
        "Alikhanov L2-1-sigma",
        "Birk-Song",
        "Diethelm",
        "Diethelm 2022",
        "Fast-oblivious CQ",
        "Full-history L1",
        "Lubich CQ",
        "Sine diffusive",
        "Sum of exponentials",
        "Yuan-Agrawal",
    }
    assert all(candidates for candidates in families.values())


def test_cost_matched_benchmark_tunes_family_specific_controls() -> None:
    families = _candidate_families(
        baseline_modes=41,
        step_size=0.025,
        final_time=1.0,
        smoke=False,
    )

    assert {candidate.parameters["order"] for candidate in families["Lubich CQ"]} == {
        "bdf1",
        "bdf2",
    }
    assert {
        candidate.step_size for candidate in families["Alikhanov L2-1-sigma"]
    } == {0.025}
    assert (
        len(
            {
                candidate.parameters["nodes_per_level"]
                for candidate in families["Fast-oblivious CQ"]
            }
        )
        > 1
    )
    assert (
        len(
            {
                candidate.parameters["target_error"]
                for candidate in families["Sum of exponentials"]
            }
        )
        > 1
    )


def test_cost_matched_reference_is_a_long_fixed_mode_run() -> None:
    families = _candidate_families(
        baseline_modes=DEFAULT_BASELINE_MODES,
        step_size=DEFAULT_STEP_SIZE,
        final_time=DEFAULT_FINAL_TIME,
        smoke=False,
    )

    assert DEFAULT_FINAL_TIME / DEFAULT_STEP_SIZE == 500
    for method in ("Birk-Song", "Diethelm"):
        assert len(families[method]) == 1
        assert families[method][0].parameters == {
            "num_modes": 100,
            "rate_scale": 1.0,
            "dt": 0.001,
        }
