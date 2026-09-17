# Dream-RSI lite — novice learning sketch

Tiny toy that logs a **discovery tree**, scores three hard-coded **exploration policies** by **replay**, and picks `argmax` — without a full meta-optimizer or changing agent weights.

## Paper / project (verified)

- **Title:** Dream-RSI: Recursive Self-Improvement through Evolving Worlds
- **Authors:** Tong Zheng, Xidong Wu, Zheng Zhang, et al. (Google / UMD / Google DeepMind / UVA)
- **arXiv:** https://arxiv.org/abs/2609.14858 (submitted 14 Sep 2026)
- **Site:** https://dream-rsi.com/
- **GitHub:** https://github.com/zhengkid/Dream-RSI
- **Abstract (restated):** RSI hinges on exploration. Fixed strategies do not adapt; online meta-optimization is expensive under delayed feedback. Dream-RSI keeps the coding agent fixed and adds a lightweight orchestration layer. Accumulated **discovery history** (trees) becomes a **replay simulator**: candidate exploration policies are “dreamt” offline at zero execution cost; the winner redeploys online and grows the world pool.

## What this sketch does

1. Build a small fixed discovery tree (nodes = attempts with pre-stored rewards).
2. Define three policies: **breadth**, **depth**, **ε-greedy**.
3. **Replay** each policy over the tree (walk / expand order only; rewards already on disk).
4. Score coverage / best-so-far; pick **argmax**. Agent “weights” stay fixed (no parameter updates).

## Verified vs guessed

| Claim | Status |
|-------|--------|
| Title, authors list on site, abstract framing | **Verified** (arXiv + dream-rsi.com) |
| History-as-exact-replay-simulator; agent unchanged | **Verified** (abstract + site) |
| Online explore → append tree → dream/replay → redeploy loop | **Verified** (site) |
| Domains (algo eng / math opt / GPU kernels) & reported speedups | **Verified on site** (not re-run; sketch does not claim those numbers) |
| Three hard-coded policies (breadth / depth / ε-greedy) | **Guessed / handoff** — pedagogical stand-ins |
| Toy tree + replay score = mean of visited node rewards | **Guessed** — paper’s real scoring/evaluator is richer |
| No full policy-development LLM rewriter | **Intentional omission** (handoff: no full meta-optimizer) |

## Deliverables

- `pseudocode.md`
- `sketch.py`
- `test_sketch.py`

## Links

- https://arxiv.org/abs/2609.14858
- https://dream-rsi.com/
- https://github.com/zhengkid/Dream-RSI

## Non-goals

Not a reimplementation of Dream-RSI. No LLM coding agent. No GitHub push from this sketch.
