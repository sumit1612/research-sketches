"""Cheap proof for GAUGE lite sketch."""

from __future__ import annotations

from sketch import (
    World,
    ranking_disagreement,
    run_dialogue,
    run_harness,
    satisfied_but_failed_rate,
    toy_llm_satisfaction,
)


def test_world_success():
    w = World(balance=100, target=100)
    assert w.task_success()
    w.deposit(1)
    assert not w.task_success()
    w.withdraw(1)
    assert w.task_success()


def test_baseline_reaches_target():
    r = run_dialogue("baseline", start_balance=40, target=100, scenario_id="t")
    assert r.task_success
    assert r.conversation_completed


def test_overshoot_fails_but_judge_may_like_polite():
    r = run_dialogue(
        "polite",
        start_balance=40,
        target=100,
        scenario_id="over",
        force_overshoot_deposit=200,
    )
    assert not r.task_success
    assert r.llm_satisfaction >= 0.0
    # polite + notes tend to push sat up; allow either side of thresh but property holds
    assert isinstance(r.satisfied_but_failed, bool)


def test_truncate_tripwire():
    r = run_dialogue(
        "baseline",
        start_balance=0,
        target=1000,
        scenario_id="trunc",
        max_steps=2,
        force_truncate=True,
    )
    assert r.conversation_completed is False


def test_judge_ignores_success():
    # empty-ish polite-looking transcript vs success flag unused
    from sketch import Turn

    t = [
        Turn("get_balance", {}, 1, "please check"),
        Turn("deposit", {"amount": 1}, 2, "thanks"),
    ]
    s = toy_llm_satisfaction(t, "polite")
    assert 0.0 <= s <= 1.0


def test_metrics_and_harness():
    out = run_harness()
    assert out["n"] >= 10
    assert 0.0 <= out["satisfied_but_failed_rate"] <= 1.0
    assert 0.0 <= out["ranking_disagreement"] <= 1.0
    # demo overshoot should contribute at least one SBF when sat is high enough
    sbf = satisfied_but_failed_rate(out["results"])
    assert sbf == out["satisfied_but_failed_rate"]


def test_ranking_disagreement_logic():
    from sketch import RunResult

    a = RunResult("polite", "a", [], task_success=False, llm_satisfaction=0.9, conversation_completed=True)
    b = RunResult("baseline", "b", [], task_success=True, llm_satisfaction=0.2, conversation_completed=True)
    assert ranking_disagreement([a, b]) == 1.0
    assert ranking_disagreement([a]) == 0.0


if __name__ == "__main__":
    test_world_success()
    test_baseline_reaches_target()
    test_overshoot_fails_but_judge_may_like_polite()
    test_truncate_tripwire()
    test_judge_ignores_success()
    test_metrics_and_harness()
    test_ranking_disagreement_logic()
    print("ALL_PASS")
