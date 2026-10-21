"""
Step 3 — Generate (t, k) samples
  • Train/val: k in {1..train_k_max}
  • Test:      all k
  • Save data/{train,val,test}_samples.jsonl
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def generate_samples(
    reactions: list[dict],
    reaction_ids: list,
    k_values: set[int] | None,   # None → all valid k
) -> list[dict]:
    """
    For each reaction in reaction_ids, enumerate all (t, k) with t+k <= T.
    k_values=None means keep every k.
    """
    id2rxn = {r["reaction_id"]: r for r in reactions}
    samples = []
    for rid in reaction_ids:
        rxn = id2rxn[rid]
        states = rxn["states"]
        T = rxn["step_count"]
        assert len(states) == T + 1, (
            f"reaction {rid}: expected {T+1} states, got {len(states)}"
        )
        for t in range(T):
            for k in range(1, T - t + 1):
                if k_values is not None and k not in k_values:
                    continue
                samples.append(
                    {
                        "reaction_id": rid,
                        "t": t,
                        "k": k,
                        "source": states[t],
                        "target": states[t + k],
                        "step_count": T,
                        "has_branching": rxn["has_branching"],
                    }
                )
    return samples


def main(cfg: dict, debug: bool = False) -> None:
    data_dir = Path(cfg["data_dir"])
    train_k_max: int = cfg["train_k_max"]

    reactions: list[dict] = json.loads((data_dir / "reactions.json").read_text())
    splits: dict = json.loads((data_dir / "splits.json").read_text())

    if debug:
        n = cfg.get("debug_n_reactions", 10)
        # keep only reactions present in the already-computed splits
        all_debug_ids = set(
            splits["train"][:n] + splits["val"][:2] + splits["test"][:3]
        )
        splits = {
            "train": [i for i in splits["train"] if i in all_debug_ids],
            "val":   [i for i in splits["val"]   if i in all_debug_ids],
            "test":  [i for i in splits["test"]  if i in all_debug_ids],
        }
        log.info("DEBUG mode: train=%d val=%d test=%d reactions",
                 len(splits["train"]), len(splits["val"]), len(splits["test"]))

    train_k = set(range(1, train_k_max + 1))

    datasets = {
        "train": generate_samples(reactions, splits["train"], k_values=train_k),
        "val":   generate_samples(reactions, splits["val"],   k_values=train_k),
        "test":  generate_samples(reactions, splits["test"],  k_values=None),
    }

    # ── print per-split, per-k statistics ───────────────────────────────────
    for split_name, samples in datasets.items():
        k_counts: dict[int, int] = defaultdict(int)
        for s in samples:
            k_counts[s["k"]] += 1
        log.info(
            "%s | total=%d  per-k: %s",
            split_name,
            len(samples),
            dict(sorted(k_counts.items())),
        )

    # ── extra stat: test k>=4, how many distinct reactions ──────────────────
    test_extrap = [s for s in datasets["test"] if s["k"] >= train_k_max + 1]
    extrap_rxns = len({s["reaction_id"] for s in test_extrap})
    k_extrap_counts: dict[int, int] = defaultdict(int)
    for s in test_extrap:
        k_extrap_counts[s["k"]] += 1
    log.info(
        "Test k>=%d | samples=%d from %d distinct reactions | per-k: %s",
        train_k_max + 1,
        len(test_extrap),
        extrap_rxns,
        dict(sorted(k_extrap_counts.items())),
    )

    # ── save ─────────────────────────────────────────────────────────────────
    for split_name, samples in datasets.items():
        out = data_dir / f"{split_name}_samples.jsonl"
        with open(out, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        log.info("Saved %d samples → %s", len(samples), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, debug=args.debug)
