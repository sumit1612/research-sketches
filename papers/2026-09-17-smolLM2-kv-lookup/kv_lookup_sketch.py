"""Toy KV prompt builder + exact-match helpers (no model required).

This is the learning skeleton for the Lost-in-the-Middle style key–value
lookup task. Pair it with infer_smollm2.py once decoding is frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from dataclasses import asdict, dataclass
from typing import Literal

Position = Literal["begin", "middle", "end"]
Distractor = Literal["random", "similar"]


@dataclass
class KVExample:
    prompt: str
    gold_key: str
    gold_value: str
    gold_position: Position
    distractor_mode: Distractor
    n_pairs: int
    seed: int


def normalize(s: str) -> str:
    return s.strip().lower()


def strip_punct(s: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", s)


def exact_match(prediction: str, gold: str) -> bool:
    """Primary pilot metric."""
    return normalize(prediction) == normalize(gold)


def normalized_bucket(prediction: str, gold: str) -> bool:
    """Secondary log-only bucket — do not optimize the pilot on this."""
    return normalize(strip_punct(prediction)) == normalize(strip_punct(gold))


def _sample_token(rng: random.Random, n: int = 6) -> str:
    alphabet = string.ascii_lowercase
    return "".join(rng.choice(alphabet) for _ in range(n))


def _similar_key(rng: random.Random, gold_key: str) -> str:
    # Share a prefix so distractors look alike; still a different string.
    prefix = gold_key[: max(2, len(gold_key) // 2)]
    return prefix + _sample_token(rng, 4)


def make_kv_example(
    n_pairs: int,
    gold_position: Position,
    distractor_mode: Distractor,
    seed: int,
) -> KVExample:
    if n_pairs < 3:
        raise ValueError("n_pairs must be >= 3 so begin/middle/end are distinct")

    rng = random.Random(seed)
    gold_key = _sample_token(rng)
    gold_value = _sample_token(rng, 8)

    distractors: list[tuple[str, str]] = []
    seen = {gold_key}
    while len(distractors) < n_pairs - 1:
        if distractor_mode == "similar":
            k = _similar_key(rng, gold_key)
        else:
            k = _sample_token(rng)
        if k in seen:
            continue
        seen.add(k)
        distractors.append((k, _sample_token(rng, 8)))

    if gold_position == "begin":
        pairs = [(gold_key, gold_value)] + distractors
    elif gold_position == "end":
        pairs = distractors + [(gold_key, gold_value)]
    else:
        mid = len(distractors) // 2
        pairs = distractors[:mid] + [(gold_key, gold_value)] + distractors[mid:]

    lines = [f"key: {k} → value: {v}" for k, v in pairs]
    question = f"What is the value for key: {gold_key}?"
    prompt = "\n".join(lines) + "\n\n" + question
    return KVExample(
        prompt=prompt,
        gold_key=gold_key,
        gold_value=gold_value,
        gold_position=gold_position,
        distractor_mode=distractor_mode,
        n_pairs=n_pairs,
        seed=seed,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--n-pairs", type=int, default=7)
    p.add_argument("--position", choices=["begin", "middle", "end"], default="middle")
    p.add_argument("--distractors", choices=["random", "similar"], default="similar")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.demo:
        p.error("pass --demo to print one example JSON")

    ex = make_kv_example(args.n_pairs, args.position, args.distractors, args.seed)
    print(json.dumps(asdict(ex), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
