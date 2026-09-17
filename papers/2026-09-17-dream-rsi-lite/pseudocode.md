# Dream-RSI lite — pseudocode

## Idea

Discovery history is already a simulator: alternative exploration policies can be scored by replaying recorded trees. Pick the best policy by argmax over replay scores; do not train the underlying agent.

## Decomposition

1. **Node** — id, parent, children, reward (pre-stored), visited flag for replay.
2. **Tree** — root + adjacency; built once online (here: hard-coded / generated).
3. **Policies** — given frontier, choose next node: breadth / depth / ε-greedy.
4. **Replay** — walk until budget; accumulate rewards of visited nodes.
5. **Select** — argmax score; report winner. Weights fixed.

## Pseudocode

```
Node = {id, parent_id, children[], reward: float}

build_toy_tree() -> root
  # fixed branching with known rewards on leaves/internals

Policy.choose(frontier, history) -> node_id
  breadth: popleft (FIFO)
  depth: pop (LIFO / DFS)
  eps_greedy: with prob eps random frontier node else argmax reward among frontier

replay(policy, tree, budget B) -> score
  frontier = [root]
  visited = []
  for t in 1..B:
    if frontier empty: break
    n = policy.choose(frontier, visited)
    remove n from frontier
    visited.append(n)
    add n.children to frontier
  score = mean(rewards of visited)   # toy; or max, or coverage*
  return score, visited

dream(policies, tree):
  scores = {p: replay(p, tree).score for p in policies}
  winner = argmax(scores)
  # no weight update on agent
  return winner, scores
```

## Edges

- Empty frontier before budget exhausted → short replay.
- ε-greedy with eps=0 → pure greedy on frontier rewards.
- Identical scores → stable tie-break by policy name order.
- Replay never mutates stored rewards (exact simulator over realized space).
