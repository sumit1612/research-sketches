"""Cheap proof for Traverse+Scout lite sketch."""

from __future__ import annotations

from sketch import (
    Step,
    Trace,
    demo_traces,
    exact_match_rate,
    naive_judge_first_bad,
    predict_first_bad,
    run_harness,
    scout_run_score,
    select_best_of_n,
)


def test_first_bad_gold():
    t = Trace(
        "x",
        [
            Step("read", "ok", False),
            Step("rm", "deleted", True),
            Step("test", "error", True),
        ],
        outcome_solved=False,
    )
    assert t.first_bad_index == 1


def test_clean_predict_none():
    t = demo_traces()[0]
    assert t.name == "clean_read"
    assert predict_first_bad(t) is None
    assert t.first_bad_index is None


def test_scout_catches_early_harm_despite_solved():
    t = next(x for x in demo_traces() if x.name == "early_delete_then_fake_pass")
    assert t.outcome_solved is True
    assert t.first_bad_index == 1
    assert predict_first_bad(t) == 1
    # naive judge trusts outcome → misses
    assert naive_judge_first_bad(t) is None


def test_judge_picks_last_symptom():
    t = next(x for x in demo_traces() if x.name == "root_then_symptom")
    assert t.first_bad_index == 0
    assert predict_first_bad(t) == 0
    assert naive_judge_first_bad(t) == 2  # last error-looking


def test_n_run_prefers_clean():
    traces = demo_traces()
    clean = next(x for x in traces if x.name == "clean_read")
    bad = next(x for x in traces if x.name == "early_delete_then_fake_pass")
    assert scout_run_score(clean) > scout_run_score(bad)
    assert select_best_of_n([bad, clean]).name == "clean_read"


def test_harness_scout_beats_judge_on_demo():
    out = run_harness()
    assert out["n"] == 5
    assert out["scout_exact_match"] > out["judge_exact_match"]
    assert out["selected_run"] == "clean_read"
    assert exact_match_rate(out["traces"], predict_first_bad) == out["scout_exact_match"]


if __name__ == "__main__":
    test_first_bad_gold()
    test_clean_predict_none()
    test_scout_catches_early_harm_despite_solved()
    test_judge_picks_last_symptom()
    test_n_run_prefers_clean()
    test_harness_scout_beats_judge_on_demo()
    print("ALL_PASS")
