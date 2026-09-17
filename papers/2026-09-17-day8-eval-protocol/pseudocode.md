# Pseudocode — Day 8 protocol decisions as code-shaped checklist

```
PROTOCOL = {
  model: "HuggingFaceTB/SmolLM2-360M-Instruct",
  positions: ["begin", "middle", "end"],
  distractors: ["random", "similar"],   # FROZEN
  primary_metric: "exact_match_strip_lower",
  secondary_metric: "normalized_punct_strip",  # log only
  decoding: {do_sample: false, max_new_tokens: 32},
  n_items: 50,            # or documented downsample
  matrix: "50 x 3 x 2 = 300",
  seeds: {dev_base: 1000, test_base: 9000},
}

FUNCTION paper_map_note():
  WRITE one paragraph each for Lost-in-the-Middle, RULER, NoLiMa
  FOR each paper: gist, what we borrow, what we do NOT claim

FUNCTION freeze_day8():
  ASSERT PROTOCOL.distractors == ["random", "similar"]
  ASSERT no other experimental knobs will change before pilot ends
  WRITE PROTOCOL to protocol.json and commit it

FUNCTION scorer_gate():
  RUN pytest on exact-match cases
  IF any fail: STOP (do not generate)

FUNCTION pilot():
  FOR position IN PROTOCOL.positions:
    FOR mode IN PROTOCOL.distractors:
      FOR i IN 0..n_items-1:
        ex = make_kv_example(..., seed=test_base + hash(position, mode, i))
        pred = greedy_generate(ex.prompt)
        LOG score(pred, ex.gold_value)

FUNCTION day14_memo():
  INCLUDE results table, runtime extrapolation, figure 1,
          error taxonomy on 20-30 misses, clean-env rerun of tiny subset
```
