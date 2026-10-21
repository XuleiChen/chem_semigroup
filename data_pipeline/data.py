"""
Step 1 — Preprocessing
  • Load ChatReaction-V456.jsonl
  • Canonicalise SMILES with RDKit (strip explicit H, sort fragments)
  • Round-trip validity check
  • Print length-distribution statistics
  • Save preprocessed reactions to data/reactions.json
"""

import argparse
import json
import logging
import os
import statistics
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit import RDLogger

# suppress RDKit warnings to keep output clean
RDLogger.DisableLog("rdApp.*")

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def to_bool(x) -> bool:
    """Accept Python bool or JSON string 'true'/'false' (case-insensitive)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        low = x.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
    raise ValueError(f"Cannot convert to bool: {x!r}")


def canon(smi: str, keep_explicit_h: bool = False) -> str | None:
    """
    Canonicalise a (possibly multi-molecule) SMILES string.

    Steps:
      1. Parse with RDKit (MolFromSmiles strips explicit H by default).
      2. Split on '.' to get individual molecule fragments.
      3. Canonicalise each fragment with MolToSmiles.
      4. Sort fragments and rejoin with '.'.

    Returns None if any fragment fails to parse.
    """
    if keep_explicit_h:
        # future experiment branch – parse with explicit H preserved
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, allHsExplicit=True)

    parts = smi.split(".")
    canon_parts = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        canon_parts.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canon_parts))


def _percentile(data: list[float], p: float) -> float:
    data_s = sorted(data)
    idx = (len(data_s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data_s) - 1)
    return data_s[lo] + (data_s[hi] - data_s[lo]) * (idx - lo)


def print_length_stats(label: str, lengths: list[int]) -> None:
    if not lengths:
        log.info("%s — no data", label)
        return
    log.info(
        "%s | n=%d  median=%.0f  p95=%.0f  max=%d",
        label,
        len(lengths),
        statistics.median(lengths),
        _percentile(lengths, 95),
        max(lengths),
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict, debug: bool = False) -> None:
    raw_path = Path(cfg["data_raw"])
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    keep_h = cfg.get("keep_explicit_h", False)

    # ── load ────────────────────────────────────────────────────────────────
    raw_records = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    log.info("Loaded %d raw records from %s", len(raw_records), raw_path)

    if debug:
        raw_records = raw_records[: cfg.get("debug_n_reactions", 10)]
        log.info("DEBUG mode: using first %d records", len(raw_records))

    # ── validate branching assertion (full dataset only) ────────────────────
    if not debug:
        n_branching = sum(
            to_bool(r["official_view"]["has_branching"]) for r in raw_records
        )
        assert n_branching == 3, (
            f"Expected exactly 3 has_branching=true reactions, got {n_branching}"
        )
        log.info("Assertion passed: exactly 3 has_branching reactions")

    # ── pre-conversion length stats ──────────────────────────────────────────
    raw_lengths_all: list[int] = []
    for r in raw_records:
        for s in r["official_path"]["states"]:
            raw_lengths_all.append(len(s))
    print_length_stats("Raw SMILES char length", raw_lengths_all)

    # ── canonicalise & validate ──────────────────────────────────────────────
    skipped_path_count = 0
    skipped_canon_fail = 0
    reactions: list[dict] = []

    canon_lengths_all: list[int] = []

    for r in raw_records:
        view = r["official_view"]
        path = r["official_path"]

        # skip non-linear paths
        if view["path_count"] != 1:
            log.warning(
                "reaction %s skipped: path_count=%s", r["sequence_idx"], view["path_count"]
            )
            skipped_path_count += 1
            continue

        states_raw = path["states"]
        step_count = path["step_count"]

        # canonicalise every state
        states_canon: list[str | None] = [canon(s, keep_h) for s in states_raw]

        # check for failures
        fail_indices = [i for i, c in enumerate(states_canon) if c is None]
        if fail_indices:
            log.warning(
                "reaction %s skipped: canon() failed at state indices %s",
                r["sequence_idx"],
                fail_indices,
            )
            skipped_canon_fail += 1
            continue

        # round-trip check: canon(raw) == canon(canon(raw))
        rt_failures = []
        for i, (raw_s, canon_s) in enumerate(zip(states_raw, states_canon)):
            recano = canon(canon_s, keep_h)
            if recano != canon_s:
                rt_failures.append(i)
        if rt_failures:
            log.warning(
                "reaction %s skipped: round-trip check failed at state indices %s",
                r["sequence_idx"],
                rt_failures,
            )
            skipped_canon_fail += 1
            continue

        for c in states_canon:
            canon_lengths_all.append(len(c))

        reactions.append(
            {
                "reaction_id": r["sequence_idx"],
                "step_count": step_count,
                "has_branching": to_bool(view["has_branching"]),
                "states": states_canon,
            }
        )

    # ── post-conversion length stats ─────────────────────────────────────────
    print_length_stats("Canon SMILES char length", canon_lengths_all)

    # ── summary ──────────────────────────────────────────────────────────────
    log.info(
        "Kept %d reactions | skipped path_count!=1: %d | skipped canon/roundtrip fail: %d",
        len(reactions),
        skipped_path_count,
        skipped_canon_fail,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = data_dir / "reactions.json"
    with open(out_path, "w") as f:
        json.dump(reactions, f, indent=2)
    log.info("Saved %d reactions → %s", len(reactions), out_path)


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, debug=args.debug)
