"""
Step 6 — Inference

Supports:
  --mode zero_shot   : base model, no adapter
  --mode sft         : load best_adapter

For each test sample generates predictions for all configured splits.
Saves data/predictions.jsonl; supports resume (skips already-done keys).

Usage:
    python infer.py --config config.yaml --mode zero_shot
    python infer.py --config config.yaml --mode sft
    python infer.py --config config.yaml --mode sft --debug
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
import yaml

from data_pipeline.data import canon
from training.model_utils import load_tokenizer_and_model
from data_pipeline.prompts import build_prompt_text

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# ── SMILES extraction ─────────────────────────────────────────────────────────

def parse_smiles(text: str) -> str | None:
    """
    Strip code-block markers, whitespace; return first non-empty line.
    Returns None if nothing is left.
    """
    # remove markdown code fences
    text = re.sub(r"```[^\n]*\n?", "", text)
    text = text.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[0] if lines else None


# ── repetition detection ──────────────────────────────────────────────────────

def has_repetition(
    generated_text: str,
    target_n_tokens: int,
    n_generated_tokens: int,
    repeat_threshold: int,
    length_multiplier: float,
) -> bool:
    # fragment-level: any fragment repeated >= repeat_threshold times
    fragments = generated_text.split(".")
    for frag in set(fragments):
        if frag and fragments.count(frag) >= repeat_threshold:
            return True
    # length-level
    if target_n_tokens > 0 and n_generated_tokens > target_n_tokens * length_multiplier:
        return True
    return False


# ── single generation ─────────────────────────────────────────────────────────

def generate_one(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_ids: list[int],
    device: str,
) -> dict:
    """
    Generate once. Returns dict with:
        raw_text, clean_text, n_generated_tokens, terminated_by_eos, hit_max_new_tokens

    raw_text  — decoded with skip_special_tokens=False (for logging/diagnosis)
    clean_text — decoded with skip_special_tokens=True  (for SMILES parsing)
    """
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
        )
    new_tokens = out[0][len(prompt_ids):]
    n_gen = len(new_tokens)
    # Bug 1 fix: keep raw_text for logging, use clean_text (no special tokens) for parsing
    raw_text   = tokenizer.decode(new_tokens, skip_special_tokens=False)
    clean_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    terminated_by_eos = any(new_tokens[-1].item() == e for e in eos_ids) if n_gen > 0 else False
    hit_max = (n_gen >= max_new_tokens)

    return {
        "raw_text":            raw_text,
        "clean_text":          clean_text,
        "n_generated_tokens":  n_gen,
        "terminated_by_eos":   terminated_by_eos,
        "hit_max_new_tokens":  hit_max,
    }


def generate_with_retry(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_ids: list[int],
    device: str,
    target_n_tokens: int,
    cfg: dict,
) -> dict:
    """
    Generate, retry once at 2× max_new_tokens if truncated.
    Adds repetition_flag and generation_failure to the result dict.
    """
    result = generate_one(model, tokenizer, prompt_ids, max_new_tokens, eos_ids, device)

    if result["hit_max_new_tokens"]:
        log.debug("Truncated, retrying with 2× max_new_tokens=%d", max_new_tokens * 2)
        retry = generate_one(model, tokenizer, prompt_ids, max_new_tokens * 2, eos_ids, device)
        retry["truncated"] = retry["hit_max_new_tokens"]
        result = retry
    else:
        result["truncated"] = False

    # Bug 1 fix: parse from clean_text (special tokens stripped), log raw_text
    result["repetition_flag"] = has_repetition(
        result["clean_text"],
        target_n_tokens,
        result["n_generated_tokens"],
        cfg["repetition_repeat_threshold"],
        cfg["repetition_length_multiplier"],
    )

    parsed = parse_smiles(result["clean_text"])
    result["generation_failure"] = (parsed is None or parsed == "")
    result["parsed_smiles"] = parsed
    return result


# ── build prompt ids ──────────────────────────────────────────────────────────

def make_prompt_ids(tokenizer, cfg: dict, source: str, k: int) -> list[int]:
    user_msg = build_prompt_text(cfg, source, k)
    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(prompt_str, add_special_tokens=False)


# ── direct prediction ─────────────────────────────────────────────────────────

def predict_direct(
    model, tokenizer, cfg, sample: dict,
    max_new_tokens: int, eos_ids: list[int], device: str,
    target_n_tokens: int = 0,
) -> dict:
    """Single-step prediction: split = (k,)."""
    prompt_ids = make_prompt_ids(tokenizer, cfg, sample["source"], sample["k"])
    res = generate_with_retry(
        model, tokenizer, prompt_ids, max_new_tokens, eos_ids, device,
        target_n_tokens=target_n_tokens,
        cfg=cfg,
    )
    return {
        "split": [sample["k"]],
        "steps": [
            {
                "step_idx": 0,
                "source": sample["source"],
                "k_segment": sample["k"],
                "raw_text": res["raw_text"],
                "parsed_smiles": res["parsed_smiles"],
                "terminated_by_eos": res["terminated_by_eos"],
                "hit_max_new_tokens": res["hit_max_new_tokens"],
                "n_generated_tokens": res["n_generated_tokens"],
                "repetition_flag": res["repetition_flag"],
                "generation_failure": res["generation_failure"],
                "truncated": res["truncated"],
                "matches_true_intermediate": None,  # final step — compared in evaluate.py
            }
        ],
    }


# ── composed prediction ───────────────────────────────────────────────────────

def predict_composed(
    model, tokenizer, cfg, sample: dict,
    split: list[int], true_intermediates: list[str | None],
    max_new_tokens: int, eos_ids: list[int], device: str,
    target_n_tokens: int = 0,
) -> dict:
    """
    Chain of predictions. split = [k_1, k_2, ..., k_m], sum = k.
    true_intermediates[i] is the true state after cumulative steps split[:i+1],
    i.e. states[t + k_1 + ... + k_{i+1}] for i = 0..m-2 (intermediates only;
    the final target is checked in evaluate.py).
    """
    steps = []
    current_source = sample["source"]
    failed_at = None

    for seg_idx, k_seg in enumerate(split):
        prompt_ids = make_prompt_ids(tokenizer, cfg, current_source, k_seg)
        res = generate_with_retry(
            model, tokenizer, prompt_ids, max_new_tokens, eos_ids, device,
            target_n_tokens=target_n_tokens,
            cfg=cfg,
        )

        # check if this intermediate matches the true intermediate state
        true_inter = true_intermediates[seg_idx] if seg_idx < len(true_intermediates) else None
        if true_inter is not None and res["parsed_smiles"] is not None:
            matches_inter = (canon(res["parsed_smiles"]) == canon(true_inter))
        else:
            matches_inter = None

        step_record = {
            "step_idx": seg_idx,
            "source": current_source,
            "k_segment": k_seg,
            "raw_text": res["raw_text"],
            "parsed_smiles": res["parsed_smiles"],
            "terminated_by_eos": res["terminated_by_eos"],
            "hit_max_new_tokens": res["hit_max_new_tokens"],
            "n_generated_tokens": res["n_generated_tokens"],
            "repetition_flag": res["repetition_flag"],
            "generation_failure": res["generation_failure"],
            "truncated": res["truncated"],
            "matches_true_intermediate": matches_inter,
        }
        steps.append(step_record)

        # propagate or fail
        if res["generation_failure"] or res["parsed_smiles"] is None:
            failed_at = seg_idx
            break

        # canon-normalise before feeding to next step
        next_source = canon(res["parsed_smiles"])
        if next_source is None:
            failed_at = seg_idx
            break
        current_source = next_source

    return {
        "split": split,
        "failed_at_step": failed_at,
        "steps": steps,
    }


# ── split enumeration ─────────────────────────────────────────────────────────

def get_splits_for_k(k: int, cfg: dict) -> list[list[int]]:
    """
    Return all splits to evaluate for a given k.
    Uses cfg["eval_splits"] for k=2,3; for other k uses (k,) and (1,...,1).
    All segments must be <= train_k_max.
    """
    train_k_max: int = cfg["train_k_max"]
    eval_splits: dict = cfg.get("eval_splits", {})

    key = f"k{k}"
    if key in eval_splits:
        splits = [list(s) for s in eval_splits[key]]
    else:
        # default: direct + all-ones
        splits = [[k]]
        if k > 1:
            ones = [1] * k
            if ones not in splits:
                splits.append(ones)

    # Bug 2 fix: filter only composed splits; direct split [k] always kept
    splits = [
        s for s in splits
        if len(s) == 1 or all(seg <= train_k_max for seg in s)
    ]
    return splits


# ── true intermediates ───────────────────────────────────────────────────────

def get_true_intermediates(sample: dict, split: list[int], states: list[str]) -> list[str | None]:
    """
    For a composed split [k_1, ..., k_m], return the true intermediate states
    (not including the final target).  Length = m-1 (the state after each
    segment except the last).
    """
    t = sample["t"]
    intermediates = []
    cum = 0
    for k_seg in split[:-1]:   # all but last segment
        cum += k_seg
        idx = t + cum
        intermediates.append(states[idx] if idx < len(states) else None)
    return intermediates


# ── main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict, mode: str, debug: bool = False) -> None:
    data_dir   = Path(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load tokenizer + model ───────────────────────────────────────────────
    tokenizer, model = load_tokenizer_and_model(cfg, load_model=True)

    if mode == "sft":
        from peft import PeftModel
        adapter_path = output_dir / "best_adapter"
        log.info("Loading SFT adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    tokenizer.padding_side = "left"   # required for batch generation

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    log.info("Device: %s  mode: %s", device, mode)

    # ── eos tokens: explicitly include both <|im_end|> and <|endoftext|> ────
    im_end_id    = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = list({im_end_id, endoftext_id, tokenizer.eos_token_id} - {None})
    log.info("eos_ids: %s  (<|im_end|>=%s  <|endoftext|>=%s)",
             eos_ids, im_end_id, endoftext_id)

    # ── max_new_tokens: compute from train+val target lengths ────────────────
    max_target_len = 0
    for split_name in ("train", "val"):
        for line in (data_dir / f"{split_name}_tokenized.jsonl").read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                max_target_len = max(max_target_len, s["n_target_tokens"])
    max_new_tokens = int(max_target_len * 1.3) + 1
    log.info("max_new_tokens: %d  (max train+val target len: %d)", max_new_tokens, max_target_len)

    # ── load reactions for intermediate state lookup ──────────────────────────
    reactions = json.loads((data_dir / "reactions.json").read_text())
    id2states = {r["reaction_id"]: r["states"] for r in reactions}

    # ── load target token lengths for repetition detection (Bug 4 fix) ──────
    target_len_lookup: dict[tuple, int] = {}
    tok_path = data_dir / "test_tokenized.jsonl"
    if tok_path.exists():
        for line in tok_path.read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                target_len_lookup[(s["reaction_id"], s["t"], s["k"])] = s["n_target_tokens"]

    # ── load test samples ────────────────────────────────────────────────────
    test_samples = [
        json.loads(l)
        for l in (data_dir / "test_samples.jsonl").read_text().splitlines()
        if l.strip()
    ]
    if debug:
        test_samples = test_samples[:20]
        log.info("DEBUG: using %d test samples", len(test_samples))

    # ── resume: load already-done keys ───────────────────────────────────────
    out_path = output_dir / f"predictions_{mode}.jsonl"
    done_keys: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                key = f"{rec['reaction_id']}_{rec['t']}_{rec['k']}_{rec['split']}"
                done_keys.add(key)
        log.info("Resume: %d predictions already done", len(done_keys))

    # ── inference loop ────────────────────────────────────────────────────────
    out_f = open(out_path, "a")
    n_written = 0

    for sample in test_samples:
        k = sample["k"]
        splits = get_splits_for_k(k, cfg)
        states = id2states.get(sample["reaction_id"], [])

        target_n_tok = target_len_lookup.get(
            (sample["reaction_id"], sample["t"], sample["k"]), 0
        )

        for split in splits:
            key = f"{sample['reaction_id']}_{sample['t']}_{k}_{split}"
            if key in done_keys:
                continue

            if split == [k]:
                pred = predict_direct(
                    model, tokenizer, cfg, sample,
                    max_new_tokens, eos_ids, device,
                    target_n_tokens=target_n_tok,
                )
            else:
                true_inter = get_true_intermediates(sample, split, states)
                pred = predict_composed(
                    model, tokenizer, cfg, sample, split, true_inter,
                    max_new_tokens, eos_ids, device,
                    target_n_tokens=target_n_tok,
                )

            record = {
                "reaction_id":   sample["reaction_id"],
                "t":             sample["t"],
                "k":             sample["k"],
                "step_count":    sample["step_count"],
                "has_branching": sample["has_branching"],
                "source":        sample["source"],
                "target":        sample["target"],
                **pred,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_written += 1

    out_f.close()
    log.info("Done. Wrote %d new predictions → %s", n_written, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode",   choices=["zero_shot", "sft"], default="zero_shot")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, mode=args.mode, debug=args.debug)
