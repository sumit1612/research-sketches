# GAUGE lite — pseudocode

## Idea

Offline “gates” that promote agents by LLM satisfaction can disagree with grounded success. Reproduce the *separation* of signals on a tiny bank task.

## Decomposition

1. **World** — mutable `balance`, fixed `target`, tool handlers.
2. **Agent** — given a system prompt variant + user goal, emits tool calls or DONE.
3. **Judge** — maps transcript → satisfaction in [0, 1] (toy heuristic).
4. **Metrics** — task_success, conversation_completed, satisfied_but_failed, ranking_disagreement.
5. **Harness** — run 3 prompt variants × fixed scenarios; aggregate.

## Pseudocode

```
World:
  balance: int
  target: int
  tools:
    get_balance() -> balance
    deposit(amount) -> balance += amount
    withdraw(amount) -> balance -= amount  # clamp at 0 for toy
  success() -> balance == target

Agent(prompt_variant, world, max_steps):
  transcript = []
  completed = False
  for step in 1..max_steps:
    action = policy(prompt_variant, transcript, world.peek_via_tools_only)
    # policy is rule-based for the sketch (no LLM):
    #   - baseline: deposit/withdraw toward target greedily
    #   - polite: same but inserts filler “please/thanks” turns
    #   - tool-heavy: extra get_balance calls before each mutate
    if action == DONE:
      completed = True
      break
    result = world.apply(action)
    transcript.append((action, result))
  return transcript, world.success(), completed

Judge(transcript) -> float in [0,1]:
  # toy: reward politeness tokens + length; IGNORE actual balance
  # (intentionally mis-anchored, like the paper’s construct gap)

RunHarness(scenarios, variants):
  rows = []
  for scenario, variant in product:
    success, sat, completed = run(...)
    rows.append(...)
  sbf = mean(sat >= thresh AND not success)
  # ranking disagreement: compare argsort by sat vs argsort by success
  return rows, sbf, disagreement
```

## Edges

- Overshoot: withdraw clamps; deposit can overshoot → failure despite “done”.
- Truncation: max_steps without DONE → `conversation_completed=False` (tripwire).
- Judge can love a polite failed run → satisfied-but-failed.
- Two variants with same success but different sat → ranking disagreement when success ties break differently.
