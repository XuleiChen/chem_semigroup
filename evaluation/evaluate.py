"""
Step 7 — Evaluation

Reads predictions_{mode}.jsonl, computes:
  7.1  Generation diagnostics + prediction correctness (two independent dims)
  7.2  Accuracy tables
  7.3  Joint consistency analysis (a/b/c/d/e)
  7.4  Stratification: in-distribution (k=2,3) vs extrapolation (k>train_k_max)
  7.5  Output files: per-sample CSV, summary CSVs, markdown report

Usage:
    python evaluate.py --config config.yaml --mode zero_shot
    python evaluate.py --config config.yaml --mode sft
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import yaml

from data_pipeline.data import canon


# ══════════════════════════════════════════════════════════════════════════════
# 7.1  Correctness classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_prediction(record: dict) -> dict:
    """
    Returns a flat dict with:
      - final_parsed_smiles, final_canon_smiles
      - invalid / valid_but_incorrect / correct   (mutually exclusive)
      - generation_failure, terminated_by_eos, hit_max_new_tokens,
        repetition_flag, truncated
      - failed_at_step (None for direct or successful composed)
      - step_intermediates: list of per-step intermediate-match results
    """
    steps = record["steps"]
    failed_at = record.get("failed_at_step")  # None for direct

    # ── pick the last meaningful step ────────────────────────────────────────
    last_step = steps[failed_at] if failed_at is not None else steps[-1]

    gen_diag = {
        "terminated_by_eos":  last_step["terminated_by_eos"],
        "hit_max_new_tokens": last_step["hit_max_new_tokens"],
        "n_generated_tokens": last_step["n_generated_tokens"],
        "repetition_flag":    last_step["repetition_flag"],
        "generation_failure": last_step["generation_failure"] or (failed_at is not None),
        "truncated":          last_step.get("truncated", False),
    }

    final_parsed = last_step["parsed_smiles"] if failed_at is None else None
    target_str   = record["target"]
    target_c     = canon(target_str)

    if gen_diag["generation_failure"] or final_parsed is None:
        final_canon = None
        invalid = True
        correct = False
        valid_but_incorrect = False
    else:
        final_canon = canon(final_parsed)
        if final_canon is None:
            invalid = True
            correct = False
            valid_but_incorrect = False
        elif final_canon == target_c:
            invalid = False
            correct = True
            valid_but_incorrect = False
        else:
            invalid = False
            correct = False
            valid_but_incorrect = True

    step_intermediates = [
        {
            "step_idx":                 s["step_idx"],
            "matches_true_intermediate": s["matches_true_intermediate"],
            "generation_failure":        s["generation_failure"],
        }
        for s in steps
    ]

    return {
        **gen_diag,
        "final_parsed_smiles":    final_parsed,
        "final_canon_smiles":     final_canon,
        "target_canon_smiles":    target_c,
        "invalid":                invalid,
        "correct":                correct,
        "valid_but_incorrect":    valid_but_incorrect,
        "failed_at_step":         failed_at,
        "step_intermediates":     step_intermediates,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    rows: list[dict],
    metric_fn,           # fn(rows) -> float
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Bootstrap CI. Resampling unit = reaction_id.
    Returns (lower, upper) for (1-alpha) CI.
    """
    reaction_ids = list({r["reaction_id"] for r in rows})
    if len(reaction_ids) < 2:
        pt = metric_fn(rows)
        return (pt, pt)

    id2rows: dict = defaultdict(list)
    for r in rows:
        id2rows[r["reaction_id"]].append(r)

    rng = random.Random(seed)
    estimates = []
    for _ in range(n):
        sampled_ids = rng.choices(reaction_ids, k=len(reaction_ids))
        sample_rows = [row for rid in sampled_ids for row in id2rows[rid]]
        v = metric_fn(sample_rows)
        if v is not None:
            estimates.append(v)

    if not estimates:
        return (None, None)
    estimates.sort()
    lo = estimates[int(math.floor(alpha / 2 * len(estimates)))]
    hi = estimates[min(int(math.ceil((1 - alpha / 2) * len(estimates))), len(estimates) - 1)]
    return (lo, hi)


def pct_str(v: float | None, lo: float | None = None, hi: float | None = None) -> str:
    if v is None:
        return "NA"
    s = f"{v:.1%}"
    if lo is not None and hi is not None:
        s += f" [{lo:.1%}, {hi:.1%}]"
    return s


# ══════════════════════════════════════════════════════════════════════════════
# 7.2  Accuracy metrics
# ══════════════════════════════════════════════════════════════════════════════

def accuracy_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "validity": None, "accuracy": None,
                "invalid_rate": None, "vbi_rate": None}
    valid  = sum(1 for r in rows if not r["invalid"])
    correct = sum(1 for r in rows if r["correct"])
    invalid = sum(1 for r in rows if r["invalid"])
    vbi    = sum(1 for r in rows if r["valid_but_incorrect"])
    return {
        "n":            n,
        "validity":     valid / n,
        "accuracy":     correct / n,
        "invalid_rate": invalid / n,
        "vbi_rate":     vbi / n,
    }


def accuracy_fn(rows):
    m = accuracy_metrics(rows)
    return m["accuracy"] if m["accuracy"] is not None else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 7.3  Joint consistency analysis
# ══════════════════════════════════════════════════════════════════════════════

def classify_pair(direct_row: dict, composed_row: dict) -> str:
    """Return one of a/b/c/d/e."""
    d_correct = direct_row["correct"]
    c_correct = composed_row["correct"]
    d_invalid = direct_row["invalid"]
    c_invalid = composed_row["invalid"]

    if d_correct and c_correct:
        return "a"
    if d_correct and not c_correct:
        return "b"
    if not d_correct and c_correct:
        return "c"
    # both wrong
    # consistent iff both valid AND same canonical SMILES
    if (not d_invalid) and (not c_invalid):
        dc = direct_row["final_canon_smiles"]
        cc = composed_row["final_canon_smiles"]
        if dc is not None and cc is not None and dc == cc:
            return "d"
    return "e"


def joint_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    cats = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}
    d_invalid = c_invalid = both_invalid = 0
    for d, c in pairs:
        cat = classify_pair(d, c)
        cats[cat] += 1
        if d["invalid"]:
            d_invalid += 1
        if c["invalid"]:
            c_invalid += 1
        if d["invalid"] and c["invalid"]:
            both_invalid += 1
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    a, b, c_, d_, e = cats["a"], cats["b"], cats["c"], cats["d"], cats["e"]
    abc = a + b + c_
    return {
        "n":                n,
        "a": a / n, "b": b / n, "c": c_ / n, "d": d_ / n, "e": e / n,
        "direct_acc":       (a + b) / n,
        "composed_acc":     (a + c_) / n,
        "consistency":      (a + d_) / n,
        "cond_consistency": (a / abc) if abc > 0 else None,
        "direct_invalid":   d_invalid / n,
        "composed_invalid": c_invalid / n,
        "both_invalid":     both_invalid / n,
    }


def joint_fn(pairs_flat):
    """pairs_flat: list of (direct_dict, composed_dict) stored as 2-element lists."""
    m = joint_metrics(pairs_flat)
    return m.get("consistency", 0.0) or 0.0


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def n_reactions(rows) -> int:
    return len({r["reaction_id"] for r in rows})


def annotate(n: int, min_cell: int) -> str:
    return " *" if n < min_cell else ""


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main(cfg: dict, mode: str) -> None:
    data_dir   = Path(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    pred_path  = output_dir / f"predictions_{mode}.jsonl"
    eval_dir   = output_dir / f"eval_{mode}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_k_max: int  = cfg["train_k_max"]
    extrap_bins       = cfg["extrap_bins"]          # [[4,5],[6,8]]
    boot_n: int       = cfg["bootstrap_n"]
    boot_seed: int    = cfg["bootstrap_seed"]
    min_cell: int     = cfg["min_cell_size"]

    # ── load & classify ──────────────────────────────────────────────────────
    raw_records = [
        json.loads(l)
        for l in pred_path.read_text().splitlines()
        if l.strip()
    ]

    flat: list[dict] = []
    for rec in raw_records:
        cl = classify_prediction(rec)
        flat.append({
            "reaction_id":    rec["reaction_id"],
            "t":              rec["t"],
            "k":              rec["k"],
            "step_count":     rec["step_count"],
            "has_branching":  rec["has_branching"],
            "split":          rec["split"],
            "split_str":      str(rec["split"]),
            "source":         rec["source"],
            "target":         rec["target"],
            **cl,
        })

    print(f"\nLoaded {len(flat)} predictions from {pred_path.name}\n")

    # ── 7.5a  per-sample CSV ─────────────────────────────────────────────────
    per_sample_path = eval_dir / "per_sample.csv"
    fieldnames = [
        "reaction_id", "t", "k", "step_count", "has_branching", "split_str",
        "source", "target", "target_canon_smiles",
        "final_parsed_smiles", "final_canon_smiles",
        "terminated_by_eos", "hit_max_new_tokens", "n_generated_tokens",
        "repetition_flag", "generation_failure", "truncated",
        "invalid", "valid_but_incorrect", "correct",
        "failed_at_step",
        "step_intermediates",
    ]
    with open(per_sample_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in flat:
            r = dict(row)
            r["step_intermediates"] = json.dumps(r["step_intermediates"])
            w.writerow(r)
    print(f"Per-sample CSV → {per_sample_path}")

    # ── helpers to build row groups ──────────────────────────────────────────
    def rows_for(k_vals, split=None, exclude_branching=False):
        return [
            r for r in flat
            if r["k"] in k_vals
            and (split is None or r["split"] == split)
            and (not exclude_branching or not r["has_branching"])
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # 7.2  Accuracy table
    # ══════════════════════════════════════════════════════════════════════════
    acc_rows = []

    def add_acc_row(label_k, label_split, rows):
        m = accuracy_metrics(rows)
        if m["n"] == 0:
            return
        ann = annotate(m["n"], min_cell)
        lo, hi = bootstrap_ci(rows, accuracy_fn, boot_n, boot_seed)
        acc_rows.append({
            "k":            label_k,
            "split":        label_split,
            "n_samples":    m["n"],
            "n_reactions":  n_reactions(rows),
            "validity":     pct_str(m["validity"]),
            "accuracy":     pct_str(m["accuracy"], lo, hi) + ann,
            "invalid_rate": pct_str(m["invalid_rate"]),
            "vbi_rate":     pct_str(m["vbi_rate"]),
        })

    # in-distribution: k=2,3
    for k in range(2, train_k_max + 1):
        k_rows = rows_for([k])
        splits_seen = sorted({tuple(r["split"]) for r in k_rows}, key=lambda s: (len(s), s))
        for sp in splits_seen:
            add_acc_row(str(k), str(list(sp)), [r for r in k_rows if r["split"] == list(sp)])
    # extrapolation — keep direct and composed splits separate
    for bin_ in extrap_bins:
        k_range = list(range(bin_[0], bin_[1] + 1))
        label_k = f"{bin_[0]}-{bin_[1]}"
        direct_rows = [r for r in flat if r["k"] in k_range and r["split"] == [r["k"]]]
        ones_rows   = [r for r in flat if r["k"] in k_range and r["split"] == [1] * r["k"]]
        if direct_rows:
            add_acc_row(label_k, "direct [k]",       direct_rows)
        if ones_rows:
            add_acc_row(label_k, "composed [1,…,1]", ones_rows)
    # k=1 single-step (from composed chains)
    k1_rows = [r for r in flat if r["k"] == 1]
    if k1_rows:
        add_acc_row("1", str([1]), k1_rows)

    acc_path = eval_dir / "accuracy.csv"
    with open(acc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(acc_rows[0].keys()) if acc_rows else [])
        w.writeheader()
        w.writerows(acc_rows)
    print(f"Accuracy CSV → {acc_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # 7.3  Joint consistency
    # ══════════════════════════════════════════════════════════════════════════
    # index by (reaction_id, t, k, split_str)
    idx: dict[tuple, dict] = {
        (r["reaction_id"], r["t"], r["k"], r["split_str"]): r
        for r in flat
    }

    cons_rows = []

    def add_cons_row(label_k, k_vals, composed_split, exclude_branching=True):
        """
        Build (direct, composed) pairs across all k in k_vals and append one row.

        k_vals         : list of k values to include
        composed_split : fixed list → every sample uses this composed split
                         None       → per-sample: direct=[k], composed=[1]*k
        Direct split is always [r["k"]] (per sample).
        """
        pairs: list[tuple[dict, dict]] = []
        reaction_ids_seen: set = set()

        for r in flat:
            if r["k"] not in k_vals:
                continue
            if exclude_branching and r["has_branching"]:
                continue
            c_split = [1] * r["k"] if composed_split is None else composed_split
            if r["split"] != [r["k"]]:          # keep only direct rows
                continue
            key_c = (r["reaction_id"], r["t"], r["k"], str(c_split))
            if key_c not in idx:
                continue
            pairs.append((r, idx[key_c]))
            reaction_ids_seen.add(r["reaction_id"])

        if not pairs:
            return

        m   = joint_metrics(pairs)
        ann = annotate(m["n"], min_cell)

        pair_records = [
            {"reaction_id": d["reaction_id"], "_d": d, "_c": c}
            for d, c in pairs
        ]

        def _boot_metric(fn_name):
            def metric(recs):
                ps = [(r["_d"], r["_c"]) for r in recs]
                return joint_metrics(ps).get(fn_name)   # may be None
            lo, hi = bootstrap_ci(pair_records, metric, boot_n, boot_seed)
            return lo, hi

        lo_da, hi_da = _boot_metric("direct_acc")
        lo_ca, hi_ca = _boot_metric("composed_acc")
        lo_co, hi_co = _boot_metric("consistency")
        lo_cc, hi_cc = _boot_metric("cond_consistency")

        if composed_split is None:
            comparison = "[k] vs [1,…,1]"
        else:
            comparison = f"[{k_vals[0]}] vs {composed_split}"

        cons_rows.append({
            "k":               label_k,
            "comparison":      comparison,
            "n_samples":       str(m["n"]) + ann,
            "n_reactions":     len(reaction_ids_seen),
            "direct_acc":      pct_str(m["direct_acc"], lo_da, hi_da),
            "composed_acc":    pct_str(m["composed_acc"], lo_ca, hi_ca),
            "consistency":     pct_str(m["consistency"], lo_co, hi_co),
            "cond_consistency":pct_str(m["cond_consistency"], lo_cc, hi_cc),
            "a": pct_str(m["a"]), "b": pct_str(m["b"]),
            "c": pct_str(m["c"]), "d": pct_str(m["d"]),
            "e": pct_str(m["e"]),
            "direct_invalid":   pct_str(m["direct_invalid"]),
            "composed_invalid": pct_str(m["composed_invalid"]),
        })

    # in-distribution comparisons (one k per call, fixed composed split)
    add_cons_row("2", [2], [1, 1])
    add_cons_row("3", [3], [1, 1, 1])
    add_cons_row("3", [3], [1, 2])
    add_cons_row("3", [3], [2, 1])

    # extrapolation: one call per bin, all k merged, composed=[1]*k per sample
    for bin_ in extrap_bins:
        k_range = list(range(bin_[0], bin_[1] + 1))
        add_cons_row(f"{bin_[0]}-{bin_[1]}", k_range, composed_split=None)

    cons_path = eval_dir / "consistency.csv"
    if cons_rows:
        with open(cons_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cons_rows[0].keys()))
            w.writeheader()
            w.writerows(cons_rows)
        print(f"Consistency CSV → {cons_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # 7.1  Generation diagnostics summary
    # ══════════════════════════════════════════════════════════════════════════
    diag_rows = []
    for k in sorted({r["k"] for r in flat}):
        k_rows = [r for r in flat if r["k"] == k]
        for sp_str in sorted({r["split_str"] for r in k_rows}):
            sp_rows = [r for r in k_rows if r["split_str"] == sp_str]
            n = len(sp_rows)
            ann = annotate(n, min_cell)
            diag_rows.append({
                "k":                     k,
                "split":                 sp_str,
                "n_samples":             str(n) + ann,
                "terminated_by_eos":     pct_str(sum(r["terminated_by_eos"] for r in sp_rows) / n),
                "hit_max_new_tokens":    pct_str(sum(r["hit_max_new_tokens"] for r in sp_rows) / n),
                "repetition_flag":       pct_str(sum(r["repetition_flag"] for r in sp_rows) / n),
                "generation_failure":    pct_str(sum(r["generation_failure"] for r in sp_rows) / n),
            })

    diag_path = eval_dir / "diagnostics.csv"
    if diag_rows:
        with open(diag_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            w.writeheader()
            w.writerows(diag_rows)
        print(f"Diagnostics CSV → {diag_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # 7.5c  Markdown report
    # ══════════════════════════════════════════════════════════════════════════
    md_lines = [
        f"# Evaluation Report — {mode}\n",
        "",
        "## Generation Diagnostics\n",
        "",
    ]

    def csv_to_md(rows: list[dict]) -> list[str]:
        if not rows:
            return ["_(no data)_\n"]
        keys = list(rows[0].keys())
        lines = ["| " + " | ".join(keys) + " |",
                 "| " + " | ".join("---" for _ in keys) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
        return [l + "\n" for l in lines]

    md_lines += csv_to_md(diag_rows)
    md_lines += ["\n## In-Distribution Accuracy (k = 2, 3)\n\n"]
    indist_acc = [r for r in acc_rows if r["k"] in ("2", "3")]
    md_lines += csv_to_md(indist_acc)
    md_lines += ["\n## In-Distribution Consistency (k = 2, 3)\n\n"]
    indist_cons = [r for r in cons_rows if r["k"] in ("2", "3")]
    md_lines += csv_to_md(indist_cons)
    md_lines += ["\n## Extrapolation Accuracy (k > train_k_max)\n\n"]
    extrap_acc = [r for r in acc_rows if r["k"] not in ("1", "2", "3")]
    md_lines += csv_to_md(extrap_acc)
    md_lines += ["\n## Extrapolation Consistency (k > train_k_max)\n\n"]
    extrap_cons = [r for r in cons_rows if r["k"] not in ("2", "3")]
    md_lines += csv_to_md(extrap_cons)

    md_path = eval_dir / "report.md"
    md_path.write_text("".join(md_lines))
    print(f"Markdown report → {md_path}")
    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode",   choices=["zero_shot", "sft"], default="zero_shot")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, mode=args.mode)
