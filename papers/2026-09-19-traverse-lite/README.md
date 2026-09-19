# Traverse lite — novice learning sketch

Tiny first-failure locator over short labeled tool-use traces. Features around each step predict the first bad index; compare against a naive LLM-as-judge heuristic stub. Optional N-run selection with a toy Scout score. Learning-sized — **not** full Traverse.

## Paper (verified)

- **Title:** Locating Hidden Failures Makes Long-Horizon Agents More Reliable
- **Authors:** Salman Rahman, Yubin Kim, Mihir Parmar, A. Ali Heydari, Genglin Liu, Simon A. Lee, Weizhi Zhang, Arian Hosseini, Ahmed A. Metwally, Yuzhe Yang, Baharan Mirzasoleiman, Xin Liu, Pavel Izmailov, Saadia Gabriel, Mark Malhotra, Shwetak Patel, Daniel McDuff, Hamid Palangi
- **arXiv:** https://arxiv.org/abs/2609.17930 (v1, submitted 15 Sep 2026)
- **Abstract (restated):** Outcome-only judging hides *where* long-horizon agents fail. The authors study **2518** trajectories (software engineering, computer use, science), classify **6967** mistakes into **78** failure types, and release human-verified annotations as **Traverse**. Failure often starts at a first mistake the agent neither recovers from nor self-catches. Six frontier judges struggle: even the strongest correctly identifies the first mistake in **fewer than a third** of runs. **Scout**, a **4B** verifier they trained, locates failure far better and transfers to unseen domains; used at test time to select among candidate runs, it raises task success without retraining the agent.

## What this sketch does

1. Build a handful of short **labeled** tool-use traces (each step has a binary `is_bad` label; one `first_bad_index`).
2. Extract simple **per-step features** (error tokens, tool name, step index, outcome lookalike).
3. **Toy Scout:** score each step → predict `argmax` / first-above-threshold as first-bad index.
4. **Naive judge stub:** heuristic that often picks the *last* error-looking step or “looks fine” — intentionally weak.
5. Optional **N-run selection:** among K candidate traces, pick the one with best toy Scout score (lowest predicted first-bad severity / latest first-bad).

## Verified vs guessed

| Claim | Status |
|-------|--------|
| Title, authors, abstract framing | **Verified** from arXiv abs + Atom API |
| 2518 trajectories | **Verified** in abstract |
| ~6967 mistakes / 78 failure types | **Verified** (abstract: 6967 / 78) |
| Frontier judges &lt;1/3 first-mistake accuracy | **Verified** (“fewer than a third”) |
| Scout ~4B verifier; test-time selection helps | **Verified** in abstract |
| HF dataset/model for this Traverse/Scout release | **Not found** on a quick HF search (handoff — may appear later) |
| Per-step feature schema, toy threshold Scout | **Guessed / handoff** — pedagogical stand-in |
| Naive LLM-as-judge heuristic stub | **Guessed** — paper uses real frontier judges |
| Exact Scout architecture / training data | **Not reimplemented** (intentional omission) |

## Deliverables

- `pseudocode.md` — idea → decomposition
- `sketch.py` — runnable toy
- `test_sketch.py` — cheap proof (no network)

## Links

- Paper: https://arxiv.org/abs/2609.17930
- PDF: https://arxiv.org/pdf/2609.17930
- DOI: https://doi.org/10.48550/arXiv.2609.17930

## Non-goals

Not a reimplementation of Traverse or Scout. No real LLM calls. No full failure taxonomy. No GitHub push from this sketch.
