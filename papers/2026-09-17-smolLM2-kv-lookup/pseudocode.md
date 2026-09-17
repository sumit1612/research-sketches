# Pseudocode — frozen SmolLM2 inference + KV lookup

## A. Frozen inference (P0 / Day 7)

```
CONFIG:
  model_id        = "HuggingFaceTB/SmolLM2-360M-Instruct"
  device          = "cpu"          # upgrade later only if timing forces it
  temperature     = 0
  do_sample       = False          # greedy
  max_new_tokens  = 32             # KV answers are short; keep this locked
  chat_template   = tokenizer's default apply_chat_template

FUNCTION timed_generate(prompt_text, n_runs):
  load tokenizer + model once
  FOR i IN 1..n_runs:
    messages = [{role: "user", content: prompt_text}]
    input_ids = apply_chat_template(messages, add_generation_prompt=True)
    t0 = now()
    output_ids = model.generate(input_ids,
                                max_new_tokens=max_new_tokens,
                                do_sample=False)
    dt = now() - t0
    new_tokens = len(output_ids) - len(input_ids)
    text = decode(new_tokens only)
    LOG i, dt, input_token_count, new_tokens, text
  RETURN logs
```

**Why:** one sitting proves the stack works and gives a runtime extrapolation for 300 gens.

## B. Build one KV example

```
FUNCTION make_kv_example(n_pairs, gold_position, distractor_mode, seed):
  # gold_position IN {begin, middle, end}
  # distractor_mode IN {random, similar}
  rng = RNG(seed)
  gold_key, gold_value = sample_unique_pair(rng)

  distractors = []
  WHILE len(distractors) < n_pairs - 1:
    k, v = sample_unique_pair(rng)
    IF distractor_mode == "similar":
      k = make_similar_key(gold_key, k)   # e.g. shared prefix / edit distance
    IF k != gold_key:
      distractors.append((k, v))

  pairs = distractors
  IF gold_position == "begin":
    pairs = [(gold_key, gold_value)] + pairs
  ELSE IF gold_position == "end":
    pairs = pairs + [(gold_key, gold_value)]
  ELSE:  # middle
    mid = len(pairs) // 2
    pairs = pairs[:mid] + [(gold_key, gold_value)] + pairs[mid:]

  context = join lines "key: {k} → value: {v}" for each pair
  question = "What is the value for key: {gold_key}?"
  prompt = context + "\n\n" + question
  RETURN {prompt, gold_value, gold_key, gold_position, distractor_mode, seed}
```

## C. Exact-match score (primary)

```
FUNCTION normalize(s):
  RETURN strip(lower(s))

FUNCTION exact_match(prediction, gold):
  # Primary metric for the pilot
  RETURN normalize(prediction) == normalize(gold)

FUNCTION normalized_bucket(prediction, gold):   # secondary, log only
  # e.g. strip punctuation / collapse spaces — DO NOT optimize on this in pilot
  RETURN normalize(strip_punct(prediction)) == normalize(strip_punct(gold))
```

## D. Mini pilot loop (after P0 works)

```
FUNCTION run_cell(position, distractor_mode, n_items, base_seed):
  results = []
  FOR i IN 0..n_items-1:
    ex = make_kv_example(..., gold_position=position,
                         distractor_mode=distractor_mode,
                         seed=base_seed + i)
    pred = timed_generate(ex.prompt, n_runs=1)[0].text
    results.append({
      **ex.meta,
      prediction: pred,
      exact: exact_match(pred, ex.gold_value),
      almost: normalized_bucket(pred, ex.gold_value),
    })
  RETURN results

# Frozen pilot shape from the execution plan:
# 50 items × 3 positions × 2 distractor modes = 300 generations
# Use separate seeds for "dev" vs "test" so you never peek.
```

## Still open (write into protocol before Day 11)

1. Context length N (how many pairs / tokens) that still finishes 10 greedy gens in a few minutes on your machine.
2. Confirm similar-vs-random is the only Day-8 factor (do not add length/model knobs mid-pilot).
3. Stay on CPU through Day 14 unless 10-example timing implies >24h for 300 gens.
