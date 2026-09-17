"""Cheap proof for Dream-RSI lite sketch."""

from __future__ import annotations

from sketch import (
    build_toy_tree,
    dream_select,
    iter_nodes,
    log_discovery_tree,
    replay,
)


def test_tree_shape():
    root = build_toy_tree()
    nodes = {n.id: n for n in iter_nodes(root)}
    assert set(nodes) == {"root", "a", "b", "c", "a1", "a2", "c1", "c2"}
    assert nodes["a1"].reward == 0.9


def test_logger_records():
    root = build_toy_tree()
    log = log_discovery_tree(root)
    assert len(log) == 8
    assert log[0]["id"] == "root"
    by_id = {r["id"]: r for r in log}
    assert by_id["a"]["child_ids"] == ["a1", "a2"]


def test_breadth_order():
    root = build_toy_tree()
    r = replay(root, "breadth", budget=4, seed=0)
    # root, then a,b,c in FIFO order of children append
    assert r.visited_ids == ["root", "a", "b", "c"]


def test_depth_order():
    root = build_toy_tree()
    r = replay(root, "depth", budget=4, seed=0)
    # LIFO: root, then last child c first
    assert r.visited_ids[0] == "root"
    assert r.visited_ids[1] == "c"


def test_replay_score_mean():
    root = build_toy_tree()
    r = replay(root, "breadth", budget=1, seed=0)
    assert r.visited_ids == ["root"]
    assert abs(r.score - 0.1) < 1e-9


def test_dream_argmax():
    root = build_toy_tree()
    winner, scores, results = dream_select(root, budget=5, seed=0)
    assert winner in scores
    assert scores[winner] == max(scores.values())
    assert len(results) == 3


def test_empty_budget():
    root = build_toy_tree()
    r = replay(root, "breadth", budget=0)
    assert r.visited_ids == []
    assert r.score == 0.0


def test_weights_fixed_contract():
    # Documented contract: dream_select only returns policy choice, no weight dict
    root = build_toy_tree()
    winner, scores, _ = dream_select(root, budget=3, seed=1)
    assert isinstance(winner, str)
    assert all(isinstance(v, float) for v in scores.values())


if __name__ == "__main__":
    test_tree_shape()
    test_logger_records()
    test_breadth_order()
    test_depth_order()
    test_replay_score_mean()
    test_dream_argmax()
    test_empty_budget()
    test_weights_fixed_contract()
    print("ALL_PASS")
