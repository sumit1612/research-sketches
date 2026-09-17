"""P0: timed greedy generations with SmolLM2-360M-Instruct (CPU-first).

Lock decoding for scoring runs: temperature unused, do_sample=False,
fixed max_new_tokens, default chat template.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any


def load_model(model_id: str, device: str):
    # Imports stay inside so `pytest` on the scorer does not need torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_once(
    tokenizer,
    model,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy; freeze for scoring
        )
    elapsed = time.perf_counter() - t0

    new_tokens = out[0, input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "elapsed_sec": round(elapsed, 3),
        "input_tokens": int(input_ids.shape[-1]),
        "new_tokens": int(new_tokens.shape[-1]),
        "text": text.strip(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=10, help="number of timed generations")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--prompt",
        default="Reply with exactly one word: ping",
        help="user message; keep short for timing smoke tests",
    )
    args = p.parse_args()

    print(json.dumps({"event": "load", "model": args.model, "device": args.device}))
    tokenizer, model = load_model(args.model, args.device)

    logs = []
    for i in range(1, args.n + 1):
        row = generate_once(
            tokenizer,
            model,
            args.prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        row["i"] = i
        logs.append(row)
        print(json.dumps({"event": "gen", **row}))

    total = sum(r["elapsed_sec"] for r in logs)
    mean = total / max(len(logs), 1)
    # Rough extrapolation for a 300-gen pilot on this machine.
    print(
        json.dumps(
            {
                "event": "summary",
                "n": len(logs),
                "mean_sec": round(mean, 3),
                "extrapolate_300_sec": round(mean * 300, 1),
                "extrapolate_300_hours": round(mean * 300 / 3600, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
