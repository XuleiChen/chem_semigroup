"""
Diagnose invalid SMILES predictions from per_sample.csv.
Categorizes error types using RDKit error logging.

Usage:
    python diagnose_invalid.py
"""

import csv
import io
import sys
from collections import Counter
from rdkit import Chem, RDLogger

logger = RDLogger.logger()


def get_rdkit_error(smi: str) -> str:
    """Parse a SMILES string and capture the RDKit error message."""
    stderr_capture = io.StringIO()
    RDLogger.EnableLog("rdApp.error")

    import contextlib, os

    # RDKit logs to C stderr; capture via Python redirect
    old_stderr = sys.stderr
    sys.stderr = stderr_capture
    mol = Chem.MolFromSmiles(smi)
    sys.stderr = old_stderr

    RDLogger.DisableLog("rdApp.*")

    if mol is not None:
        return "VALID"

    err = stderr_capture.getvalue().strip()
    return err if err else "unknown_parse_error"


def categorize_error(raw_smi: str, rdkit_err: str) -> str:
    """Classify the error into a human-readable category."""
    if raw_smi is None or raw_smi.strip() == "":
        return "empty_output"

    # Check for natural language / non-SMILES content
    non_smi_keywords = ["the ", "step", "react", "predict", "after", "system",
                        "state", "result", "answer", "output", "smiles"]
    lower = raw_smi.lower()
    if any(kw in lower for kw in non_smi_keywords):
        return "natural_language"

    if "ring" in rdkit_err.lower() or "closure" in rdkit_err.lower():
        return "ring_closure_error"
    if "valence" in rdkit_err.lower():
        return "valence_error"
    if "bracket" in rdkit_err.lower() or "parenthes" in rdkit_err.lower():
        return "bracket_error"
    if "atom" in rdkit_err.lower():
        return "atom_error"
    if "aromatic" in rdkit_err.lower() or "kekulize" in rdkit_err.lower():
        return "aromaticity_error"
    if "charge" in rdkit_err.lower():
        return "charge_error"
    if rdkit_err == "unknown_parse_error":
        return "unknown_parse_error"

    return "other: " + rdkit_err[:80]


def main():
    csv_path = "outputs/eval_sft/per_sample.csv"

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter: only k=1 direct predictions that are invalid
    k1_invalid = [r for r in rows if r["k"] == "1" and r["invalid"] == "True"]
    k1_total = [r for r in rows if r["k"] == "1"]
    k1_valid = [r for r in rows if r["k"] == "1" and r["invalid"] == "False"]

    print(f"=== k=1 predictions ===")
    print(f"Total: {len(k1_total)}, Valid: {len(k1_valid)}, Invalid: {len(k1_invalid)}")
    print()

    # Analyze invalid predictions
    error_categories = Counter()
    examples_per_category = {}

    for r in k1_invalid:
        smi = r["final_parsed_smiles"]
        if smi == "" or smi == "None":
            smi = None

        if smi is None:
            cat = "empty_output"
            rdkit_err = ""
        else:
            # Try parsing each fragment
            parts = smi.split(".")
            fragment_errors = []
            for part in parts:
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    err = get_rdkit_error(part)
                    fragment_errors.append((part, err))

            if not fragment_errors:
                # All individual fragments parse OK but whole thing might fail
                err = get_rdkit_error(smi)
                cat = categorize_error(smi, err)
                rdkit_err = err
            else:
                # Use first failing fragment
                fail_frag, err = fragment_errors[0]
                cat = categorize_error(fail_frag, err)
                rdkit_err = err

        error_categories[cat] += 1
        if cat not in examples_per_category:
            examples_per_category[cat] = []
        if len(examples_per_category[cat]) < 3:
            examples_per_category[cat].append({
                "predicted": smi[:120] if smi else "None",
                "target": r["target"][:120],
                "rdkit_error": rdkit_err[:120] if rdkit_err else "",
            })

    # Print summary
    print(f"=== Error categories (k=1 invalid, n={len(k1_invalid)}) ===")
    print()
    for cat, count in error_categories.most_common():
        pct = 100 * count / len(k1_invalid)
        print(f"  {cat}: {count} ({pct:.1f}%)")
        for ex in examples_per_category[cat]:
            print(f"    pred: {ex['predicted']}")
            print(f"    target: {ex['target']}")
            if ex['rdkit_error']:
                print(f"    error: {ex['rdkit_error']}")
            print()

    # Also check: direct predictions for k=2,3
    print(f"\n=== Direct predictions (all k) ===")
    for k in ["1", "2", "3", "4", "5"]:
        direct = [r for r in rows if r["k"] == k and r["split_str"] == f"[{k}]"]
        if not direct:
            continue
        n = len(direct)
        n_invalid = sum(1 for r in direct if r["invalid"] == "True")
        n_correct = sum(1 for r in direct if r["correct"] == "True")
        print(f"  k={k}: total={n}, invalid={n_invalid} ({100*n_invalid/n:.1f}%), correct={n_correct} ({100*n_correct/n:.1f}%)")


if __name__ == "__main__":
    main()
