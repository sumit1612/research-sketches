"""Dream-RSI lite: discovery-tree logger + replay scoring of exploration policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Literal

PolicyName = Literal["breadth", "depth", "eps_greedy"]


@dataclass
class Node:
    id: str
    reward: float
    children: list[Node] = field(default_factory=list)
    parent_id: str | None = None


@dataclass
class ReplayResult:
    policy: PolicyName
    visited_ids: list[str]
    score: float
    budget: int


def build_toy_tree() -> Node:
    r"""
    Fixed discovery tree (realized search space). Rewards are pre-stored —
    replay never re-executes nodes.

         root(0.1)
        /    |    \
      a(0.2) b(0.5) c(0.3)
      / \          / \
    a1  a2       c1  c2
   (0.9)(0.4)   (0.6)(0.8)
    """
    a1 = Node("a1", 0.9)
    a2 = Node("a2", 0.4)
    c1 = Node("c1", 0.6)
    c2 = Node("c2", 0.8)
    a = Node("a", 0.2, children=[a1, a2])
    b = Node("b", 0.5, children=[])
    c = Node("c", 0.3, children=[c1, c2])
    root = Node("root", 0.1, children=[a, b, c])
    for child in (a, b, c):
        child.parent_id = root.id
    a1.parent_id = a2.parent_id = a.id
    c1.parent_id = c2.parent_id = c.id
    return root


def iter_nodes(root: Node) -> list[Node]:
    out: list[Node] = []
    stack = [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(reversed(n.children))
    return out


def log_discovery_tree(root: Node) -> list[dict[str, object]]:
    """Structured logger: one record per node (handoff: discovery-tree logger)."""
    records: list[dict[str, object]] = []
    for n in iter_nodes(root):
        records.append(
            {
                "id": n.id,
                "parent_id": n.parent_id,
                "reward": n.reward,
                "child_ids": [c.id for c in n.children],
            }
        )
    return records


def _choose_breadth(frontier: deque[Node], _rng_state: list[int]) -> Node:
    return frontier.popleft()


def _choose_depth(frontier: deque[Node], _rng_state: list[int]) -> Node:
    return frontier.pop()


def _choose_eps_greedy(
    frontier: deque[Node],
    rng_state: list[int],
    eps: float = 0.2,
) -> Node:
    """Deterministic PRNG via LCG on rng_state[0] so tests are stable."""
    rng_state[0] = (1103515245 * rng_state[0] + 12345) % (2**31)
    u = rng_state[0] / float(2**31)
    items = list(frontier)
    if u < eps:
        idx = rng_state[0] % len(items)
        chosen = items[idx]
    else:
        chosen = max(items, key=lambda n: (n.reward, n.id))
    frontier.remove(chosen)
    return chosen


POLICY_CHOOSERS: dict[PolicyName, Callable[..., Node]] = {
    "breadth": _choose_breadth,
    "depth": _choose_depth,
    "eps_greedy": _choose_eps_greedy,
}


def replay(
    root: Node,
    policy: PolicyName,
    budget: int,
    seed: int = 0,
) -> ReplayResult:
    if budget < 1:
        return ReplayResult(policy=policy, visited_ids=[], score=0.0, budget=budget)

    frontier: deque[Node] = deque([root])
    visited: list[Node] = []
    rng_state = [seed]
    choose = POLICY_CHOOSERS[policy]

    for _ in range(budget):
        if not frontier:
            break
        if policy == "eps_greedy":
            node = choose(frontier, rng_state)
        else:
            node = choose(frontier, rng_state)
        visited.append(node)
        for child in node.children:
            frontier.append(child)

    score = sum(n.reward for n in visited) / len(visited) if visited else 0.0
    return ReplayResult(
        policy=policy,
        visited_ids=[n.id for n in visited],
        score=score,
        budget=budget,
    )


def dream_select(
    root: Node,
    budget: int = 5,
    seed: int = 0,
    policies: tuple[PolicyName, ...] = ("breadth", "depth", "eps_greedy"),
) -> tuple[PolicyName, dict[PolicyName, float], list[ReplayResult]]:
    """
    Score policies by replay; pick argmax.
    Agent weights stay fixed — we only choose an exploration schedule.
    """
    results = [replay(root, p, budget=budget, seed=seed) for p in policies]
    scores: dict[PolicyName, float] = {r.policy: r.score for r in results}
    # stable tie-break: higher score, then policy name
    winner = max(policies, key=lambda p: (scores[p], p))
    return winner, scores, results


def main() -> None:
    root = build_toy_tree()
    log = log_discovery_tree(root)
    print(f"logged_nodes={len(log)}")
    winner, scores, results = dream_select(root, budget=5, seed=42)
    print(f"winner={winner}")
    for r in results:
        print(f"  {r.policy:12s} score={r.score:.3f} path={r.visited_ids}")
    print("agent_weights_fixed=True")


if __name__ == "__main__":
    main()
