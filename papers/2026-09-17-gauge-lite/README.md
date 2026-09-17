# GAUGE lite — novice learning sketch

Tiny toy that separates **grounded task success** from **LLM-as-a-judge satisfaction**, inspired by GAUGE.

## Paper (verified)

- **Title:** GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents
- **Authors:** Umesh Bodhwani, Thanh Tran, Kai Wei
- **arXiv:** https://arxiv.org/abs/2609.12191 (submitted 10 Sep 2026)
- **Abstract (restated):** Offline eval of task-oriented agents often runs a persona user-sim, then an LLM judge scores the transcript, and the higher-scoring agent wins. GAUGE checks whether that gate’s ranking matches a grounded verifiable reward. Key findings: (1) satisfaction is decorrelated from actual task success — **57.5%** of conversations rated satisfied still fail the customer’s task; (2) ranking is fine across a wide capability span but decision-disagreement jumps from **&lt;1%** on wide-reward pairs to **31%** on close pairs. Remedy sketch: calibrate-then-trust, with a judge-free **completion bit** as a zero-cost tripwire.

## What this sketch does

A bank “world” has `balance` and `target`. The agent must use tools until `balance == target`. We run three prompt variants (baseline / polite / tool-heavy), score each run with:

| Signal | Meaning |
|--------|---------|
| `task_success` | World/DB: `balance == target` after the dialogue |
| `llm_satisfaction` | Toy 0..1 judge over the transcript (not a real LLM) |
| `conversation_completed` | Tripwire: agent emitted an explicit done/stop signal |

Then we log **satisfied-but-failed** rate and **ranking disagreement** between satisfaction-order vs success-order.

## Verified vs guessed

| Claim | Status |
|-------|--------|
| Title, authors, abstract framing | **Verified** from arXiv abs page |
| 57.5% satisfied-but-failed | **Verified** in paper abstract |
| Decision-disagreement &lt;1% → 31% (wide vs close pairs) | **Verified** in paper abstract |
| τ2-bench / SimulatorArena / 25 agents / six providers | **Verified** in abstract (not re-run here) |
| Bank toy, tool API, three prompt variants | **Guessed / handoff** — pedagogical stand-in, not in the paper |
| Toy transcript “judge” heuristic | **Guessed** — paper uses real LLM/human raters |
| Exact tool names / DB schema | **Guessed** |

## Deliverables

- `pseudocode.md` — idea → decomposition
- `sketch.py` — runnable toy
- `test_sketch.py` — cheap proof (no network)

## Links

- Paper: https://arxiv.org/abs/2609.12191
- DOI: https://doi.org/10.48550/arXiv.2609.12191

## Non-goals

Not a reimplementation of GAUGE, τ2-bench, or SimulatorArena. No real LLM calls. No GitHub push from this sketch.
