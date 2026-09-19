# Traverse + Scout lite — pseudocode

## Idea

Outcome success hides the *first* irreversible mistake. On short labeled traces, locate that index with step features, then compare to a weak judge stub. Optionally pick the best of N candidate runs.

## Decomposition

1. **Trace** — ordered steps: tool, args summary, observation, optional gold `is_bad`.
2. **Features** — cheap signals per step (error-ish tokens, bad tool names, late index, fake-success phrases).
3. **Toy Scout** — score each step; first index with score ≥ thresh (or argmax) = predicted first-bad.
4. **Naive judge** — stub that prefers last failure-looking step or claims “ok” if outcome looks solved.
5. **Metrics** — exact first-bad match rate; mean absolute index error.
6. **N-run select** — among candidates, prefer higher `scout_run_score` (later / no predicted fail).

## Pseudocode

```
Step:
  tool: str
  observation: str
  is_bad: bool          # gold label (sketch only)
  looks_successful: bool

Trace:
  steps: list[Step]
  first_bad_index: int | None   # gold: min i with is_bad, else None
  outcome_solved: bool          # may be True even with early harm

features(step, i, n) -> dict:
  error_token = 1 if any(w in obs.lower() for w in ERROR_WORDS)
  risky_tool  = 1 if tool in RISKY_TOOLS
  fake_ok     = 1 if "fabricat" in obs or "claimed ok" in obs
  position    = i / max(1, n-1)
  return {error_token, risky_tool, fake_ok, position, ...}

toy_scout_step_score(feats) -> float:
  # weighted sum of failure cues; learning-sized weights (fixed)
  return w_err*error_token + w_risk*risky_tool + w_fake*fake_ok + ...

predict_first_bad(trace) -> int | None:
  scores = [toy_scout_step_score(features(s,i,n)) for s in enumerate(trace)]
  for i, sc in enumerate(scores):
    if sc >= THRESH: return i
  return None   # or argmax if all below but max is clear

naive_judge_first_bad(trace) -> int | None:
  # weak: if outcome_solved, often return None
  # else return LAST index with error_token (misses early root cause)

scout_run_score(trace) -> float:
  # higher = healthier run for selection
  pred = predict_first_bad(trace)
  if pred is None: return 1.0
  return pred / len(trace)   # later fail > earlier fail (toy)

select_best_of_n(candidates) -> Trace:
  return argmax_c scout_run_score(c)

evaluate(traces):
  scout_acc  = mean(predict_first_bad(t) == t.first_bad_index)
  judge_acc  = mean(naive_judge_first_bad(t) == t.first_bad_index)
  return scout_acc, judge_acc
```

## Edges

- Solved-but-harmful: `outcome_solved=True` with early `is_bad` — judge stub misses; Scout should catch.
- Late symptom vs early cause: judge picks last error token; Scout picks first above thresh.
- Clean run: both return None; selection prefers clean over early-fail candidates.
