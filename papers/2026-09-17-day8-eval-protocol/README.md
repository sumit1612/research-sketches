# Day 8 protocol sketch — paper map + frozen pilot design

One-page protocol for the SmolLM2 key–value retrieval pilot (Notion due 29 Sep 2026).
Companion code sketch: `../2026-09-17-smolLM2-kv-lookup/`.

## Research question

How reliably does **SmolLM2-360M-Instruct** retrieve a value from a list of key–value pairs when the gold pair is at the **beginning, middle, or end**, under **random vs similar** distractors?

## Three-paper map (why each paper is here)

### 1. Lost in the Middle (Liu et al., 2023)
- **Link:** https://arxiv.org/abs/2307.03172 · https://huggingface.co/papers/2307.03172
- **Gist:** Long-context LMs often use the ends of the context better than the middle; multi-doc QA and **key–value retrieval** are the probes.
- **We borrow:** KV lookup task shape; position as a first-class factor; “U-shaped” accuracy intuition.
- **We do not claim yet:** same models, lengths, or absolute numbers as the paper.

### 2. RULER (Hsieh et al., 2024)
- **Link:** https://arxiv.org/abs/2404.06654 · https://huggingface.co/papers/2404.06654
- **Gist:** Advertised context window ≠ effective context; synthetic suites stress retrieval and aggregation beyond a single needle.
- **We borrow:** treat “can the model use N tokens?” as an empirical measurement, not a marketing number; keep the pilot small and measurable.
- **We do not claim yet:** full RULER task suite or multi-hop aggregation.

### 3. NoLiMa (Modarressi et al., 2025)
- **Link:** https://arxiv.org/abs/2502.05167 · code https://github.com/adobe-research/NoLiMa
- **Gist:** Needle-in-haystack can be gamed by **literal lexical overlap**; NoLiMa reduces that overlap so the model must do harder association.
- **We borrow:** awareness that exact string cues inflate scores; similar-key distractors are our cheap proxy for “harder than random noise.”
- **We do not claim yet:** NoLiMa’s latent-association needle set (out of scope for Day 8–14).

## Frozen pilot factors (do not change after Day 8)

| Factor | Levels | Notes |
| --- | --- | --- |
| Model | SmolLM2-360M-Instruct | Fixed; second model is a later phase |
| Task | Key → value lookup | One gold key asked at the end of the prompt |
| Gold position | begin / middle / end | 3 levels |
| Distractors | random / similar | **Frozen Day-8 question** |
| Metric (primary) | Exact match after strip + lower | |
| Metric (secondary) | Normalized / almost-right | Log only; do not tune on it |
| Decoding | Greedy (`do_sample=False`), fixed `max_new_tokens` | |
| Device | CPU first | GPU only if 10-gen timing implies >24h for 300 |

## Design matrix

- **Items:** 50 unique seeds (or fewer if slow — document the downsample)
- **Cells:** 3 positions × 2 distractor modes
- **Total generations:** 50 × 3 × 2 = **300**
- **Seeds:** separate `dev_seed_base` and `test_seed_base` (never peek at test while iterating prompts)

## Scoring harness checks (before any pilot)

Run `pytest` in the sibling sketch folder. Required cases:

1. Correct prediction → exact True  
2. Wrong value → exact False  
3. Malformed (“the answer is …”) → exact False  
4. Missing / empty → exact False  
5. Punctuation-only difference → secondary bucket may True, primary False  

## Logging fields (every generation)

`seed, position, distractor_mode, n_pairs, context_tokens, new_tokens, elapsed_sec, gold_value, prediction, exact, almost, model_id, max_new_tokens, device`

## NeurIPS-checklist style fields (fill as you go)

- [ ] Code + config that regenerates the table from a clean env  
- [ ] Compute: device, wall-clock, tokens in/out  
- [ ] Limitations: toy keys, small model, no NoLiMa needles  
- [ ] Negative results allowed: inconclusive is still a Day-14 success if explained  

## Definition of done (Day 14 / 29 Sep)

You can explain the setup, reproduce a tiny subset from a clean environment, show a results table (or documented downsample), one figure, and a short error taxonomy on 20–30 misses. Conference submission is a **week-6 gate**, not the September bar.

## Still unknown / decide before Day 11

1. Exact `n_pairs` / context length N for your laptop.  
2. Whether `pytorch-playground` stays the sandbox and a sibling eval repo holds logs (recommended).  
3. When (if ever) to leave CPU.
