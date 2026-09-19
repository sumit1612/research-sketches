# Blindspot lite — novice learning sketch

Tiny **trajectory-level** safety calibration toy: adjudicate short tool-use runs into one of five outcomes, including late unsafe completions after initially safe turns. Learning-sized — **not** the full Blindspot benchmark.

## Paper (verified)

- **Title:** Blindspot: A Benchmark for Safety and Refusal Calibration in Long-Horizon Tool-Using Agents
- **arXiv:** https://arxiv.org/abs/2609.16305 (v1)
- **Abstract (restated):** Binary task/attack success hides whether a tool agent stays calibrated over many turns as authorization and state evolve. Blindspot evaluates full user–agent–environment trajectories with adaptive adversarial interaction, stateful tools, and **execution-grounded** adjudication. Current instantiation: **22** attack families, **35** scenarios across seven domains, **more than 2,500** trajectories (avg **14.7** turns). Each trajectory gets one of five outcomes: **Safe Completion**, **Correct Refusal**, **Unsafe Completion**, **Over-Refusal**, or **Indeterminate**. They evaluate **13** proprietary and open-weight LLMs with eight metrics; failures can emerge only after several initially safe steps.
- **DOI:** https://doi.org/10.48550/arXiv.2609.16305

## What this sketch does

A tiny file/mail world with `auth_scope` (`read` | `write` | `admin`). Tools: `read_file`, `delete_file`, `send_email`. Policies mark when refusal is required. `adjudicate(traj, policy)` uses **execution facts** (did irreversible harm happen? did agent refuse? were benign goals completed?) to assign one of the five outcomes. One **authorization-drift** scenario stays safe for two turns, then deletes without admin — late `UnsafeCompletion`.

## Verified vs guessed

| Claim | Status |
|-------|--------|
| Title, abstract framing, five outcome names | **Verified** from arXiv HTML abstract |
| 22 attack families / 35 scenarios / >2500 trajs / 14.7 avg turns | **Verified** in abstract |
| 13 models evaluated; late-emerging failures | **Verified** in abstract |
| Exact metric formulas / attack family list | **Not reimplemented** |
| File/mail toy env, auth scopes, 3-turn scripts | **Guessed / pedagogical** |
| Author list | **Not extracted** on this run (abs page thin); treat as unknown |

## Deliverables

- `pseudocode.md` — idea → decomposition
- `sketch.py` — runnable toy
- `test_sketch.py` — cheap proof (no network)

## Links

- Paper: https://arxiv.org/abs/2609.16305
- HTML: https://arxiv.org/html/2609.16305v1
- DOI: https://doi.org/10.48550/arXiv.2609.16305

## Non-goals

Not a reimplementation of Blindspot live-sim. No real LLM calls. No 22 attack families.
