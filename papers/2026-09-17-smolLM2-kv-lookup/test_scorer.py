"""Scorer tests first: correct, wrong, malformed, missing."""

from kv_lookup_sketch import exact_match, make_kv_example, normalized_bucket


def test_exact_match_correct():
    assert exact_match("AbCdEfGh", "abcdefgh") is True


def test_exact_match_wrong():
    assert exact_match("nope", "abcdefgh") is False


def test_exact_match_malformed_extra_words():
    # Primary metric is strict after strip/lower — extra words fail.
    assert exact_match("the answer is abcdefgh", "abcdefgh") is False


def test_exact_match_missing_empty():
    assert exact_match("", "abcdefgh") is False


def test_normalized_bucket_allows_punct():
    assert normalized_bucket("abcdefgh!", "abcdefgh") is True


def test_make_kv_positions_place_gold():
    for pos in ("begin", "middle", "end"):
        ex = make_kv_example(7, pos, "random", seed=42)
        lines = [ln for ln in ex.prompt.splitlines() if ln.startswith("key:")]
        gold_line = f"key: {ex.gold_key} → value: {ex.gold_value}"
        idx = lines.index(gold_line)
        if pos == "begin":
            assert idx == 0
        elif pos == "end":
            assert idx == len(lines) - 1
        else:
            assert 0 < idx < len(lines) - 1
