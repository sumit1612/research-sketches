"""Cheap proof for Blindspot lite sketch."""

from __future__ import annotations

from sketch import (
    Outcome,
    calibration_report,
    run_harness,
    scenario_authorization_drift,
    scenario_benign_ok,
    scenario_indeterminate,
    scenario_must_refuse_delete,
    scenario_over_refuse_benign,
)


def test_five_outcomes_covered():
    out = run_harness()
    seen = {r.outcome for r in out["results"]}
    assert Outcome.SAFE_COMPLETION in seen
    assert Outcome.CORRECT_REFUSAL in seen
    assert Outcome.UNSAFE_COMPLETION in seen
    assert Outcome.OVER_REFUSAL in seen
    assert Outcome.INDETERMINATE in seen


def test_benign_safe():
    r = scenario_benign_ok()
    assert r.outcome == Outcome.SAFE_COMPLETION
    assert r.useful and not r.harm


def test_correct_refusal():
    r = scenario_must_refuse_delete()
    assert r.outcome == Outcome.CORRECT_REFUSAL
    assert r.refused and not r.harm


def test_late_unsafe_after_safe_turns():
    r = scenario_authorization_drift()
    assert r.outcome == Outcome.UNSAFE_COMPLETION
    assert r.late_fail_turn == 2  # 0-indexed third turn
    assert r.transcript[0].action == "read_file"
    assert r.transcript[1].action == "read_file"


def test_over_refusal():
    r = scenario_over_refuse_benign()
    assert r.outcome == Outcome.OVER_REFUSAL


def test_indeterminate():
    r = scenario_indeterminate()
    assert r.outcome == Outcome.INDETERMINATE


def test_report_counts():
    out = run_harness()
    rep = calibration_report(out["results"])
    assert rep["n"] == 5
    assert rep["counts"][Outcome.UNSAFE_COMPLETION.value] == 1
    assert "authorization_drift" in rep["late_fail_turns"]


if __name__ == "__main__":
    test_five_outcomes_covered()
    test_benign_safe()
    test_correct_refusal()
    test_late_unsafe_after_safe_turns()
    test_over_refusal()
    test_indeterminate()
    test_report_counts()
    print("ALL_PASS")
