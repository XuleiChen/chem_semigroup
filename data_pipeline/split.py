"""
Step 2 — Train / Val / Test split
  • Stratify by step_count, fixed random seed
  • Same reaction never appears in two splits
  • Save data/splits.json  (lists of reaction_id)
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def stratified_split(
    reactions: list[dict],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list, list, list]:
    """
    Split reaction_ids into train/val/test, stratified by step_count.
    Returns three lists of reaction_id.
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"
    r_train, r_val, r_test = ratios

    # group by step_count
    by_step: dict[int, list] = defaultdict(list)
    for rxn in reactions:
        by_step[rxn["step_count"]].append(rxn["reaction_id"])

    rng = random.Random(seed)

    train_ids, val_ids, test_ids = [], [], []

    for step, ids in sorted(by_step.items()):
        ids = ids[:]
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * r_test))
        n_val  = max(1, round(n * r_val))
        # ensure we don't over-allocate for tiny strata
        n_test = min(n_test, n)
        n_val  = min(n_val,  n - n_test)

        test_ids  += ids[:n_test]
        val_ids   += ids[n_test : n_test + n_val]
        train_ids += ids[n_test + n_val :]

    return train_ids, val_ids, test_ids


def main(cfg: dict, debug: bool = False) -> None:
    data_dir = Path(cfg["data_dir"])
    reactions: list[dict] = json.loads((data_dir / "reactions.json").read_text())

    if debug:
        reactions = reactions[: cfg.get("debug_n_reactions", 10)]
        log.info("DEBUG mode: using %d reactions", len(reactions))

    ratios = tuple(cfg["split_ratios"])          # [train, val, test]
    seed   = cfg["split_seed"]

    train_ids, val_ids, test_ids = stratified_split(reactions, ratios, seed)

    # ── sanity checks ────────────────────────────────────────────────────────
    all_ids = set(r["reaction_id"] for r in reactions)
    assert set(train_ids) | set(val_ids) | set(test_ids) == all_ids, \
        "some reactions are missing from the split"
    assert len(set(train_ids) & set(val_ids) & set(test_ids)) == 0, \
        "overlapping splits"
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(reactions), \
        "reaction count mismatch"

    total = len(reactions)
    log.info(
        "Split | train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
        len(train_ids), 100 * len(train_ids) / total,
        len(val_ids),   100 * len(val_ids)   / total,
        len(test_ids),  100 * len(test_ids)  / total,
    )

    # ── step_count distribution per split ───────────────────────────────────
    id2step = {r["reaction_id"]: r["step_count"] for r in reactions}

    def step_dist(ids):
        d = defaultdict(int)
        for rid in ids:
            d[id2step[rid]] += 1
        return dict(sorted(d.items()))

    log.info("Train step_count dist: %s", step_dist(train_ids))
    log.info("Val   step_count dist: %s", step_dist(val_ids))
    log.info("Test  step_count dist: %s", step_dist(test_ids))

    # ── save ─────────────────────────────────────────────────────────────────
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    out = data_dir / "splits.json"
    out.write_text(json.dumps(splits, indent=2))
    log.info("Saved splits → %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, debug=args.debug)
