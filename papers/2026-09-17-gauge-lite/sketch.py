"""GAUGE lite: toy bank agent separating task_success from llm_satisfaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

PromptVariant = Literal["baseline", "polite", "tool-heavy"]

SATISFACTION_THRESH = 0.6


@dataclass
class World:
    balance: int
    target: int

    def get_balance(self) -> int:
        return self.balance

    def deposit(self, amount: int) -> int:
        if amount < 0:
            raise ValueError("deposit amount must be >= 0")
        self.balance += amount
        return self.balance

    def withdraw(self, amount: int) -> int:
        if amount < 0:
            raise ValueError("withdraw amount must be >= 0")
        self.balance = max(0, self.balance - amount)
        return self.balance

    def task_success(self) -> bool:
        return self.balance == self.target


@dataclass
class Turn:
    action: str
    args: dict[str, Any]
    result: Any
    note: str = ""


@dataclass
class RunResult:
    variant: PromptVariant
    scenario_id: str
    transcript: list[Turn]
    task_success: bool
    llm_satisfaction: float
    conversation_completed: bool

    @property
    def satisfied_but_failed(self) -> bool:
        return self.llm_satisfaction >= SATISFACTION_THRESH and not self.task_success


def toy_llm_satisfaction(transcript: list[Turn], variant: PromptVariant) -> float:
    """Mis-anchored judge: likes politeness and tool chatter, ignores world success."""
    score = 0.3
    polite_hits = sum(1 for t in transcript if "please" in t.note.lower() or "thanks" in t.note.lower())
    tool_hits = sum(1 for t in transcript if t.action == "get_balance")
    length_bonus = min(0.2, 0.02 * len(transcript))
    score += min(0.4, 0.1 * polite_hits)
    score += min(0.3, 0.05 * tool_hits)
    score += length_bonus
    if variant == "polite":
        score += 0.15
    if variant == "tool-heavy":
        score += 0.1
    return max(0.0, min(1.0, score))


def _delta(world: World) -> int:
    return world.target - world.balance


def agent_policy(
    variant: PromptVariant,
    world: World,
    step: int,
    max_steps: int,
) -> tuple[str, dict[str, Any], str]:
    """Rule-based stand-in for an LLM agent with three prompt styles."""
    need = _delta(world)

    if need == 0:
        note = "thanks, all set" if variant == "polite" else "done"
        return "DONE", {}, note

    # tool-heavy: peek before every mutate on even steps
    if variant == "tool-heavy" and step % 2 == 0:
        note = "please check balance" if variant == "polite" else "check balance"
        return "get_balance", {}, note

    if need > 0:
        note = "please deposit" if variant == "polite" else "deposit"
        return "deposit", {"amount": need}, note

    note = "please withdraw" if variant == "polite" else "withdraw"
    return "withdraw", {"amount": -need}, note


def run_dialogue(
    variant: PromptVariant,
    start_balance: int,
    target: int,
    scenario_id: str,
    max_steps: int = 8,
    # Optional hook to inject buggy/truncated behavior in tests
    force_truncate: bool = False,
    force_overshoot_deposit: int | None = None,
) -> RunResult:
    world = World(balance=start_balance, target=target)
    transcript: list[Turn] = []
    completed = False

    for step in range(max_steps):
        action, args, note = agent_policy(variant, world, step, max_steps)

        # Tripwire demo: refuse DONE and stop early so conversation_completed stays false.
        if force_truncate and action == "DONE":
            break

        forced = False
        if force_overshoot_deposit is not None and action == "deposit":
            args = {"amount": force_overshoot_deposit}
            forced = True

        if action == "DONE":
            completed = True
            transcript.append(Turn(action=action, args=args, result="ok", note=note))
            break
        if action == "get_balance":
            result = world.get_balance()
        elif action == "deposit":
            result = world.deposit(int(args["amount"]))
        elif action == "withdraw":
            result = world.withdraw(int(args["amount"]))
        else:
            raise ValueError(f"unknown action: {action}")

        transcript.append(Turn(action=action, args=args, result=result, note=note))

        # Pedagogical failure: stop after forced overshoot so the agent does not self-correct.
        if forced:
            completed = True
            transcript.append(
                Turn(action="DONE", args={}, result="ok", note="thanks, all set")
            )
            break

    success = world.task_success()
    # If we hit target but never said DONE, tripwire stays false
    if success and not completed and not force_truncate:
        # allow one free DONE if we landed exactly and still have a step conceptually
        # — only mark completed if last action was DONE
        completed = bool(transcript) and transcript[-1].action == "DONE"

    sat = toy_llm_satisfaction(transcript, variant)
    return RunResult(
        variant=variant,
        scenario_id=scenario_id,
        transcript=transcript,
        task_success=success,
        llm_satisfaction=sat,
        conversation_completed=completed,
    )


def satisfied_but_failed_rate(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.satisfied_but_failed) / len(results)


def ranking_disagreement(results: list[RunResult]) -> float:
    """
    Fraction of unordered pairs where satisfaction order disagrees with success order.
    Ties on either axis do not count as disagreement.
    """
    n = len(results)
    if n < 2:
        return 0.0
    disagree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = results[i], results[j]
            sat_cmp = (a.llm_satisfaction > b.llm_satisfaction) - (
                a.llm_satisfaction < b.llm_satisfaction
            )
            suc_cmp = (a.task_success > b.task_success) - (a.task_success < b.task_success)
            if sat_cmp == 0 or suc_cmp == 0:
                continue
            total += 1
            if sat_cmp != suc_cmp:
                disagree += 1
    return 0.0 if total == 0 else disagree / total


def run_harness() -> dict[str, Any]:
    scenarios = [
        ("s_under", 40, 100),
        ("s_over", 150, 100),
        ("s_exact", 100, 100),
    ]
    variants: list[PromptVariant] = ["baseline", "polite", "tool-heavy"]
    results: list[RunResult] = []
    for sid, bal, tgt in scenarios:
        for v in variants:
            results.append(run_dialogue(v, bal, tgt, scenario_id=f"{sid}:{v}"))

    # Inject one polite-but-failed case for the pedagogical gap demo
    failed_polite = run_dialogue(
        "polite",
        start_balance=40,
        target=100,
        scenario_id="demo:overshoot",
        force_overshoot_deposit=200,
    )
    results.append(failed_polite)

    return {
        "results": results,
        "satisfied_but_failed_rate": satisfied_but_failed_rate(results),
        "ranking_disagreement": ranking_disagreement(results),
        "n": len(results),
    }


def main() -> None:
    out = run_harness()
    print(f"runs={out['n']}")
    print(f"satisfied_but_failed_rate={out['satisfied_but_failed_rate']:.3f}")
    print(f"ranking_disagreement={out['ranking_disagreement']:.3f}")
    for r in out["results"]:
        print(
            f"  {r.scenario_id:20s} success={r.task_success} "
            f"sat={r.llm_satisfaction:.2f} completed={r.conversation_completed} "
            f"sbf={r.satisfied_but_failed}"
        )


if __name__ == "__main__":
    main()
