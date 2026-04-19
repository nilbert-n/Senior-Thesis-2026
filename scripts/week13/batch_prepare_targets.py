#!/usr/bin/env python3
"""
Week 13 batch pipeline — processes all targets listed in manifest_targets.yaml.

For each target:
  - Locates the input PDB/mmCIF in the inputs directory
  - Calls prepare_target() with chain/resrange overrides from the manifest
  - Writes outputs to Single_Chain/Week 13/outputs/<target>/
  - Writes a per-target validation report
  - Collects a batch summary JSON and flags any NEEDS_REVIEW targets

Usage:
  python batch_prepare_targets.py
  python batch_prepare_targets.py --manifest manifest_targets.yaml
  python batch_prepare_targets.py --inputs ../../Single_Chain/Week\ 13/inputs
  python batch_prepare_targets.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Resolve sibling paths relative to this script
SCRIPT_DIR  = Path(__file__).resolve().parent
WEEK13_DIR  = SCRIPT_DIR.parents[1] / "Single_Chain" / "Week 13"
DEFAULT_MANIFEST = SCRIPT_DIR / "manifest_targets.yaml"
DEFAULT_INPUTS   = WEEK13_DIR / "inputs"
DEFAULT_OUTPUTS  = WEEK13_DIR / "outputs"
DEFAULT_VALDIR   = WEEK13_DIR / "validation"


def load_manifest(path: Path):
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return data.get("targets", [])


def find_input_file(inputs_dir: Path, target_entry: dict):
    """
    Resolve the input PDB/mmCIF path for a target.
    Looks for an explicit 'file' key first, then searches by target name.
    """
    if "file" in target_entry:
        p = inputs_dir / target_entry["file"]
        if p.exists():
            return p
        # Also try as absolute path
        p2 = Path(target_entry["file"])
        if p2.exists():
            return p2
        return None

    name = target_entry["name"]
    for suffix in (".pdb1", ".pdb1.gz", ".pdb", ".pdb.gz",
                   ".cif", ".cif.gz", ".mmcif", ".mmcif.gz"):
        p = inputs_dir / (name + suffix)
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Week 13 batch target preparation"
    )
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help=f"YAML manifest (default: {DEFAULT_MANIFEST})")
    ap.add_argument("--inputs",   default=str(DEFAULT_INPUTS),
                    help=f"Directory containing input PDB files (default: {DEFAULT_INPUTS})")
    ap.add_argument("--outputs",  default=str(DEFAULT_OUTPUTS),
                    help=f"Root output directory (default: {DEFAULT_OUTPUTS})")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print what would be done without running")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    inputs_dir    = Path(args.inputs)
    outputs_root  = Path(args.outputs)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(2)

    targets = load_manifest(manifest_path)
    if not targets:
        print("[ERROR] No targets found in manifest", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Manifest   : {manifest_path}")
    print(f"[INFO] Inputs dir : {inputs_dir}")
    print(f"[INFO] Outputs    : {outputs_root}")
    print(f"[INFO] Targets    : {len(targets)}")
    print()

    # Import here so errors surface cleanly
    from prepare_target import prepare_target

    results   = []
    n_pass    = 0
    n_review  = 0
    n_error   = 0
    n_skipped = 0

    for entry in targets:
        name = entry.get("name")
        if not name:
            print("[WARN] Manifest entry missing 'name' — skipping")
            n_skipped += 1
            continue

        skip = entry.get("skip", False)
        if skip:
            print(f"[SKIP] {name}  (skip=true in manifest)")
            n_skipped += 1
            results.append({"target": name, "status": "SKIPPED"})
            continue

        pdb_path = find_input_file(inputs_dir, entry)
        if pdb_path is None:
            print(f"[ERROR] {name}  — input file not found in {inputs_dir}")
            n_error += 1
            results.append({"target": name, "status": "ERROR",
                             "message": "input file not found"})
            continue

        chain_pref = entry.get("chain", None)
        resrange   = None
        if "resrange" in entry:
            r = entry["resrange"]
            if isinstance(r, str) and ":" in r:
                a, b = r.split(":")
                resrange = (int(a), int(b))
            elif isinstance(r, list) and len(r) == 2:
                resrange = (int(r[0]), int(r[1]))

        out_dir = outputs_root / name

        if args.dry_run:
            print(f"[DRY-RUN] {name}  pdb={pdb_path.name}  "
                  f"chain={chain_pref or 'auto'}  resrange={resrange or 'all'}")
            results.append({"target": name, "status": "DRY_RUN"})
            continue

        result = prepare_target(
            pdb_path   = pdb_path,
            out_dir    = out_dir,
            target     = name,
            chain_pref = chain_pref,
            resrange   = resrange,
        )
        results.append(result)

        status = result.get("status", "ERROR")
        if status == "PASS":
            n_pass += 1
        elif status == "NEEDS_REVIEW":
            n_review += 1
        else:
            n_error += 1

    # --- Batch summary ---
    print()
    print("=" * 60)
    print(f"Batch complete  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  PASS         : {n_pass}")
    print(f"  NEEDS_REVIEW : {n_review}")
    print(f"  ERROR        : {n_error}")
    print(f"  SKIPPED      : {n_skipped}")

    if n_review:
        print("\nTargets flagged for manual review:")
        for r in results:
            if r.get("status") == "NEEDS_REVIEW":
                print(f"  {r['target']}")
                for f in r.get("flags", []):
                    print(f"    ^ {f}")

    summary = {
        "generated": datetime.now().isoformat(),
        "manifest":  str(manifest_path),
        "n_pass":    n_pass,
        "n_review":  n_review,
        "n_error":   n_error,
        "n_skipped": n_skipped,
        "targets":   results,
    }
    outputs_root.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary written to {summary_path}")

    if n_error:
        sys.exit(2)
    if n_review:
        sys.exit(1)


if __name__ == "__main__":
    main()
