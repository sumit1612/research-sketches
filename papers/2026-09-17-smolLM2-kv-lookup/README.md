# SmolLM2 key–value lookup — P0 inference + harness sketch

Novice-friendly sketch for Sumit's LLM eval path (Notion: *AI research path — LLM evaluation & reliability*).

## Plain restatement

We want to know: **when a small chat model is given a long bag of key–value pairs, can it still look up the right value if the answer sits at the start, middle, or end?**

This folder is **Day 7–style scaffolding** (frozen inference) plus the **KV task skeleton** you will fill before the ~300-gen pilot. It is a learning sketch, not a paper reproduction yet.

## Why each piece exists

| Piece | Why |
| --- | --- |
| Frozen decoding | Same prompt must yield comparable answers across positions; `temperature=0` removes random drift. |
| Chat template | Instruct models expect a fixed message format; wrong format looks like a model failure. |
| Positions begin / middle / end | Classic “lost in the middle” stress: accuracy often drops when the gold pair is buried. |
| Similar vs random distractors | Similar keys make lookup harder than random noise; freeze this before the pilot. |
| Exact-match scorer | Primary metric for the pilot; simple, auditable, no LLM-as-judge. |

## Sources (verified)

- Liu et al., *Lost in the Middle* — https://arxiv.org/abs/2307.03172 · https://huggingface.co/papers/2307.03172 · code https://github.com/nelson-liu/lost-in-the-middle
- Model card: `HuggingFaceTB/SmolLM2-360M-Instruct` — https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct

## Files

- `pseudocode.md` — step-by-step method
- `infer_smollm2.py` — tiny CPU-first inference script (timed generations)
- `kv_lookup_sketch.py` — toy KV prompt builder + exact-match scorer (no model required for unit tests)
- `test_scorer.py` — failing-first scorer cases

## Run notes

```bash
pip install 'transformers>=4.42' torch accelerate
python infer_smollm2.py --n 10          # timed generations
python -m pytest test_scorer.py -q
python kv_lookup_sketch.py --demo      # builds one prompt, no model
```

CPU is fine for P0. Record `max_new_tokens`, template, and device in every log line.
