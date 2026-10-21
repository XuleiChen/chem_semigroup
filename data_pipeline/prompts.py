"""
Step 4 — Prompt construction and tokenization

Core function:
    build_sample(tokenizer, cfg, sample) -> dict with input_ids / labels / meta

When run as __main__:
    • Tokenizes all train/val/test samples
    • Prints token-length distribution
    • Runs decode-check on 2 random samples
    • Saves data/{train,val,test}_tokenized.jsonl
"""

import argparse
import json
import logging
import random
import statistics
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# ── core ─────────────────────────────────────────────────────────────────────

def build_prompt_text(cfg: dict, source: str, k: int) -> str:
    template: str = cfg["prompt_template"]
    return template.format(source=source, k=k).rstrip("\n")


def build_sample(tokenizer, cfg: dict, sample: dict) -> dict:
    """
    Returns:
        input_ids : list[int]  — prompt + target + eos
        labels    : list[int]  — -100 * len(prompt) + target + eos
        n_prompt_tokens  : int
        n_target_tokens  : int  (target only, excluding eos)
        total_tokens     : int
    """
    user_msg = build_prompt_text(cfg, sample["source"], sample["k"])

    # ── prompt ids (includes <|im_start|>assistant\n prefix) ─────────────────
    # tokenize=False → string; then encode with add_special_tokens=False
    # (avoids BatchEncoding return type in transformers ≥ 5.x)
    prompt_str: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids: list[int] = tokenizer.encode(prompt_str, add_special_tokens=False)

    # ── target ids — no special tokens so we don't inject a stray BOS ────────
    target_enc = tokenizer(sample["target"], add_special_tokens=False)
    target_ids: list[int] = target_enc["input_ids"]

    # ── EOS for the assistant turn: <|im_end|> ────────────────────────────────
    # Verify token exists; Qwen2.5 chat template uses <|im_end|> (id 151645)
    im_end_id: Optional[int] = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        # fallback: use eos_token_id
        im_end_id = tokenizer.eos_token_id
    eos_ids: list[int] = [im_end_id]

    input_ids = prompt_ids + target_ids + eos_ids
    labels    = [-100] * len(prompt_ids) + target_ids + eos_ids

    # ── assertion: non-(-100) labels == target tokens + 1 (eos) ─────────────
    non_masked = [x for x in labels if x != -100]
    assert len(non_masked) == len(target_ids) + 1, (
        f"Label length mismatch: expected {len(target_ids)+1}, "
        f"got {len(non_masked)}"
    )
    assert len(input_ids) == len(labels), "input_ids / labels length mismatch"

    return {
        "reaction_id":    sample["reaction_id"],
        "t":              sample["t"],
        "k":              sample["k"],
        "input_ids":      input_ids,
        "labels":         labels,
        "n_prompt_tokens": len(prompt_ids),
        "n_target_tokens": len(target_ids),
        "total_tokens":    len(input_ids),
    }


# ── decode-check ─────────────────────────────────────────────────────────────

def decode_check(tokenizer, tokenized_samples: list[dict], n: int = 2, seed: int = 0):
    rng = random.Random(seed)
    chosen = rng.sample(tokenized_samples, min(n, len(tokenized_samples)))
    for i, s in enumerate(chosen):
        full_decoded = tokenizer.decode(s["input_ids"], skip_special_tokens=False)
        label_tokens = [t for t in s["labels"] if t != -100]
        label_decoded = tokenizer.decode(label_tokens, skip_special_tokens=False)
        log.info(
            "── Decode check %d/%d (reaction=%s t=%d k=%d) ──",
            i + 1, n, s["reaction_id"], s["t"], s["k"],
        )
        log.info("  full input_ids decoded:\n%s", full_decoded)
        log.info("  labels (non -100) decoded: %r", label_decoded)


# ── length statistics ─────────────────────────────────────────────────────────

def _pct(data: list[float], p: float) -> float:
    data_s = sorted(data)
    idx = (len(data_s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(data_s) - 1)
    return data_s[lo] + (data_s[hi] - data_s[lo]) * (idx - lo)


def print_token_stats(label: str, lengths: list[int]) -> None:
    log.info(
        "%s | n=%d  median=%.0f  p95=%.0f  max=%d",
        label,
        len(lengths),
        statistics.median(lengths),
        _pct(lengths, 95),
        max(lengths),
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict, debug: bool = False) -> None:
    from training.model_utils import load_tokenizer_and_model

    data_dir = Path(cfg["data_dir"])

    # load tokenizer only (no model weights needed here)
    tokenizer, _ = load_tokenizer_and_model(cfg, load_model=False)

    # confirm pad_token is set
    assert tokenizer.pad_token is not None, "tokenizer.pad_token is None"
    log.info(
        "Tokenizer loaded | vocab=%d  pad=%r  eos=%r  im_end=%r",
        tokenizer.vocab_size,
        tokenizer.pad_token,
        tokenizer.eos_token,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )

    splits_to_process = ["train", "val", "test"]
    train_val_target_lengths: list[int] = []   # for max_new_tokens computation

    for split_name in splits_to_process:
        src = data_dir / f"{split_name}_samples.jsonl"
        samples = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

        if debug:
            samples = samples[:50]

        tokenized = []
        for s in samples:
            tokenized.append(build_sample(tokenizer, cfg, s))

        total_lengths  = [t["total_tokens"]    for t in tokenized]
        target_lengths = [t["n_target_tokens"] for t in tokenized]
        prompt_lengths = [t["n_prompt_tokens"] for t in tokenized]

        print_token_stats(f"{split_name} total tokens ", total_lengths)
        print_token_stats(f"{split_name} prompt tokens", prompt_lengths)
        print_token_stats(f"{split_name} target tokens", target_lengths)

        if split_name in ("train", "val"):
            train_val_target_lengths.extend(target_lengths)

        # save
        out = data_dir / f"{split_name}_tokenized.jsonl"
        with open(out, "w") as f:
            for t in tokenized:
                f.write(json.dumps(t) + "\n")
        log.info("Saved %d tokenized samples → %s", len(tokenized), out)

    # ── max_new_tokens recommendation ─────────────────────────────────────────
    max_target = max(train_val_target_lengths)
    recommended_max_new = int(max_target * 1.3) + 1
    log.info(
        "max target token length (train+val): %d  →  recommended max_new_tokens: %d",
        max_target,
        recommended_max_new,
    )

    # ── decode check on train ─────────────────────────────────────────────────
    train_tok = [
        json.loads(l)
        for l in (data_dir / "train_tokenized.jsonl").read_text().splitlines()
        if l.strip()
    ]
    decode_check(tokenizer, train_tok, n=2, seed=cfg.get("seed", 42))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, debug=args.debug)
