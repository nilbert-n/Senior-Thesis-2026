#!/usr/bin/env python3
"""
Week 13 target preparation pipeline.

Parses a PDB or mmCIF file (optionally gzipped), extracts C3' backbone atoms
for a chosen chain, validates RNA residue completeness and continuity, then
writes five output files into a per-target output directory:

  <target>_rna_full.pdb       cleaned RNA-only full-atom PDB
  <target>_c3only.pdb         C3'-only PDB (one bead per residue)
  <target>.dat                LAMMPS coarse-grained data file
  <target>_sequence.txt       extracted RNA sequence (ACGU)
  <target>_metadata.json      chain info, residue range, sequence, flags
  <target>_validation.txt     residue continuity and C3' completeness report

Usage:
  python prepare_target.py --pdb inputs/1YMO.pdb1 --out outputs/1YMO
  python prepare_target.py --pdb inputs/3NJ6.pdb1.gz --chain B --resrange 5:80
  python prepare_target.py --pdb inputs/4GXY.cif.gz --out outputs/4GXY --chain A

Exits with code 2 and writes a validation report if any ambiguity is detected
(missing C3', discontinuous residues, multiple RNA chains with similar size).
"""

import argparse
import gzip
import json
import re
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RES2TYPE = {
    "A": 1, "ADE": 1, "RA": 1, "DA": 1,
    "C": 2, "CYT": 2, "RC": 2, "DC": 2,
    "G": 3, "GUA": 3, "RG": 3, "DG": 3, "GTP": 3,  # GTP = 5'-triphosphate G, same bead type
    "U": 4, "URA": 4, "RU": 4,
    "T": 4, "DT": 4,           # treat thymine as U for CG model
}

RNA_BASE = {
    "A": "A", "ADE": "A", "RA": "A", "DA": "A",
    "C": "C", "CYT": "C", "RC": "C", "DC": "C",
    "G": "G", "GUA": "G", "RG": "G", "DG": "G", "GTP": "G",
    "U": "U", "URA": "U", "RU": "U",
    "T": "U", "DT": "U",
}

DEFAULT_MASSES = {1: 329.20, 2: 305.20, 3: 345.20, 4: 306.20}
DEFAULT_BOX    = (-500.6, 500.6, -500.6, 500.6, -500.6, 500.6)

# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _open(path: Path):
    s = str(path)
    if s.endswith(".gz"):
        return gzip.open(s, "rt", encoding="utf-8", errors="ignore")
    return open(s, "rt", encoding="utf-8", errors="ignore")


def _is_cif(path: Path) -> bool:
    s = str(path).lower()
    return any(s.endswith(ext) for ext in (".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))


# ---------------------------------------------------------------------------
# PDB parser
# ---------------------------------------------------------------------------

def parse_pdb(path: Path, model=1):
    """
    Returns:
      residues  : OrderedDict[(chain, resseq_int, icode)] -> (resname, chain)
      c3_coords : dict[key] -> (x, y, z)
      full_atoms: list of raw ATOM/HETATM lines for RNA residues only

    If MODEL/ENDMDL records are present, only the requested model is read.
    If no MODEL records are present, all ATOM/HETATM lines are read.
    """
    residues   = OrderedDict()
    c3_coords  = {}
    full_atoms = []

    saw_model = False
    keep_this_model = True

    with _open(path) as fh:
        for line in fh:
            rec = line[:6]

            if rec.startswith("MODEL "):
                saw_model = True
                try:
                    model_num = int(line[10:14].strip())
                except ValueError:
                    model_num = None
                keep_this_model = (model_num == model)
                continue

            if rec.startswith("ENDMDL"):
                keep_this_model = False
                continue

            if rec not in ("ATOM  ", "HETATM"):
                continue

            # If this file has MODEL records, only keep the chosen one
            if saw_model and not keep_this_model:
                continue

            resn = line[17:20].strip().upper()
            if resn not in RES2TYPE:
                continue

            chain = line[21].strip() or "_"
            icode = line[26].strip()

            try:
                resseq = int(line[22:26])
            except ValueError:
                continue

            # Keep only primary altloc
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue

            key = (chain, resseq, icode)
            if key not in residues:
                residues[key] = (resn, chain)

            full_atoms.append(line.rstrip("\n"))

            atom_raw  = line[12:16].strip()
            atom_norm = atom_raw.replace("*", "'").replace("\u2019", "'")
            if atom_norm == "C3'":
                try:
                    c3_coords[key] = (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                except ValueError:
                    pass

    return residues, c3_coords, full_atoms


# ---------------------------------------------------------------------------
# mmCIF parser
# ---------------------------------------------------------------------------

def parse_mmcif(path: Path):
    residues   = OrderedDict()
    c3_coords  = {}
    full_atoms = []

    with _open(path) as fh:
        text = fh.read()

    m = re.search(
        r"loop_\s+(_atom_site\.\S.*?)(?:\n\s*\n|\Z)",
        text, flags=re.S
    )
    if not m:
        return residues, c3_coords, full_atoms

    block = m.group(1).splitlines()
    cols, data_lines = [], []
    for line in block:
        line = line.strip()
        if line.startswith("_atom_site."):
            cols.append(line.split(".")[1])
        elif cols and line and not line.startswith("#"):
            data_lines.append(line)

    if not cols:
        return residues, c3_coords, full_atoms

    idx = {c: i for i, c in enumerate(cols)}
    required = {"label_atom_id", "label_asym_id", "Cartn_x", "Cartn_y", "Cartn_z",
                "label_seq_id", "label_comp_id"}
    if not required.issubset(idx):
        return residues, c3_coords, full_atoms

    for line in data_lines:
        parts = re.findall(r"(?:'[^']*'|\"[^\"]*\"|\S+)", line)
        if len(parts) < len(cols):
            continue

        def _get(key):
            return parts[idx[key]].strip("'\"")

        resn = _get("label_comp_id").upper()
        if resn not in RES2TYPE:
            continue

        chain = _get("label_asym_id") or "_"
        try:
            resseq = int(_get("label_seq_id"))
        except ValueError:
            continue
        icode = ""
        key = (chain, resseq, icode)

        if key not in residues:
            residues[key] = (resn, chain)

        atom_norm = _get("label_atom_id").replace("*", "'")
        if atom_norm == "C3'":
            try:
                c3_coords[key] = (
                    float(_get("Cartn_x")),
                    float(_get("Cartn_y")),
                    float(_get("Cartn_z")),
                )
            except ValueError:
                pass

        # Reconstruct a minimal ATOM line for the full-atom PDB
        try:
            x, y, z = float(_get("Cartn_x")), float(_get("Cartn_y")), float(_get("Cartn_z"))
            serial   = int(parts[idx["id"]]) if "id" in idx else 0
            aname    = _get("label_atom_id")
            full_atoms.append(
                f"ATOM  {serial:5d}  {aname:<3s} {resn:>3s} {chain:1s}"
                f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
        except (ValueError, KeyError):
            pass

    return residues, c3_coords, full_atoms


# ---------------------------------------------------------------------------
# Chain selection
# ---------------------------------------------------------------------------

def choose_chain(residues, preferred=None):
    counts = defaultdict(int)
    for (chain, _, _) in residues:
        counts[chain] += 1

    if preferred and counts.get(preferred, 0) > 0:
        return preferred, False   # (chain, is_ambiguous)

    if not counts:
        return None, False

    sorted_chains = sorted(counts.items(), key=lambda kv: -kv[1])
    best_chain, best_count = sorted_chains[0]

    # Ambiguous: two chains within 10% of each other
    ambiguous = (
        len(sorted_chains) > 1
        and sorted_chains[1][1] >= 0.9 * best_count
    )
    return best_chain, ambiguous


# ---------------------------------------------------------------------------
# Residue ordering and trimming
# ---------------------------------------------------------------------------

def sorted_keys(residues, chain):
    keys = [k for k in residues if k[0] == chain]
    keys.sort(key=lambda k: (k[1], k[2]))
    return keys


def apply_resrange(keys, resrange):
    """Filter keys to [start, end] inclusive (1-based residue numbers)."""
    if resrange is None:
        return keys
    start, end = resrange
    return [k for k in keys if start <= k[1] <= end]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_continuity(keys):
    """
    Returns list of gap tuples (resseq_before, resseq_after) where the
    residue numbering jumps by more than 1.
    """
    gaps = []
    for i in range(1, len(keys)):
        prev, curr = keys[i - 1][1], keys[i][1]
        if curr - prev > 1:
            gaps.append((prev, curr))
    return gaps


def check_c3_completeness(keys, c3_coords):
    missing = [k for k in keys if k not in c3_coords]
    return missing


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_full_atom_pdb(full_atoms, chain, keys_set, out_path: Path):
    """Write RNA-only full-atom PDB, filtering to the selected chain and residues."""
    valid_keys = keys_set
    with open(out_path, "w") as fh:
        for line in full_atoms:
            if len(line) < 27:
                continue
            rec   = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            ch = line[21].strip() or "_"
            if ch != chain:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            if (ch, resseq, icode) not in valid_keys:
                continue
            fh.write(line + "\n")
        fh.write("END\n")


def write_c3_pdb(keys, c3_coords, chain, out_path: Path):
    with open(out_path, "w") as fh:
        for serial, k in enumerate(keys, start=1):
            resn       = c3_coords.get(k)  # coords tuple
            resname, _ = "RNA", "_"        # fallback
            x, y, z    = c3_coords[k]
            fh.write(
                f"ATOM  {serial:5d}  C3' RNA {chain:1s}{k[1]:4d}{k[2] or ' ':1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        fh.write("END\n")


def write_lammps_dat(seq, coords, out_path: Path,
                     masses=DEFAULT_MASSES, box=DEFAULT_BOX):
    N     = len(seq)
    types = [RES2TYPE[b] for b in seq]
    xlo, xhi, ylo, yhi, zlo, zhi = box

    with open(out_path, "w") as fh:
        fh.write(f"LAMMPS data file — single RNA C3' beads: {seq}\n\n")
        fh.write(f"{N} atoms\n{max(N-1,0)} bonds\n{max(N-2,0)} angles\n\n")
        fh.write("4 atom types\n1 bond types\n1 angle types\n\n")
        fh.write(f"{xlo: .6f}    {xhi: .6f}  xlo xhi\n")
        fh.write(f"{ylo: .6f}    {yhi: .6f}  ylo yhi\n")
        fh.write(f"{zlo: .6f}    {zhi: .6f}  zlo zhi\n\n")
        fh.write("Masses\n\n")
        for t in [1, 2, 3, 4]:
            fh.write(f"{t} {masses[t]:.2f}\n")
        fh.write("\nAtoms\n\n")
        for i, (t, (x, y, z)) in enumerate(zip(types, coords), start=1):
            fh.write(f"{i} 1 {t} 0.000000 {x:.6f} {y:.6f} {z:.6f}\n")
        if N >= 2:
            fh.write("\nBonds\n\n")
            for i in range(1, N):
                fh.write(f"{i} 1 {i} {i+1}\n")
        if N >= 3:
            fh.write("\nAngles\n\n")
            for i in range(1, N-1):
                fh.write(f"{i} 1 {i} {i+1} {i+2}\n")


def write_validation_report(out_path: Path, target, chain, keys, missing_c3,
                             gaps, chain_ambiguous, trimmed_from, trimmed_to,
                             flags):
    with open(out_path, "w") as fh:
        fh.write(f"Validation report — {target}\n")
        fh.write(f"Generated: {datetime.now().isoformat()}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Chain selected : {chain}")
        fh.write(" *** AMBIGUOUS — manual review required ***\n" if chain_ambiguous else "\n")
        fh.write(f"Residues used  : {trimmed_from}–{trimmed_to}  ({len(keys)} total)\n\n")

        if gaps:
            fh.write(f"GAPS ({len(gaps)}) — residue numbering is discontinuous:\n")
            for a, b in gaps:
                fh.write(f"  {a} -> {b}  (jump of {b-a})\n")
        else:
            fh.write("Continuity     : OK (no gaps)\n")

        fh.write("\n")
        if missing_c3:
            fh.write(f"MISSING C3' ({len(missing_c3)}) — manual review required:\n")
            for (ch, resseq, icode) in missing_c3:
                fh.write(f"  chain {ch}  resseq {resseq}  icode '{icode}'\n")
        else:
            fh.write("C3' completeness: OK (all residues have C3' atoms)\n")

        if flags:
            fh.write("\nFlags:\n")
            for f in flags:
                fh.write(f"  - {f}\n")

        status = "NEEDS_REVIEW" if (missing_c3 or chain_ambiguous or gaps) else "PASS"
        fh.write(f"\nOverall status : {status}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_target(pdb_path: Path, out_dir: Path, target: str,
                   chain_pref=None, resrange=None):
    """
    Full pipeline for one target. Returns a dict summarising outcome.
    Raises SystemExit(2) only when the problem is unrecoverable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse ---
    if _is_cif(pdb_path):
        residues, c3_coords, full_atoms = parse_mmcif(pdb_path)
    else:
        residues, c3_coords, full_atoms = parse_pdb(pdb_path)

    if not residues:
        msg = f"No RNA/DNA residues found in {pdb_path}"
        _write_error_report(out_dir / f"{target}_validation.txt", target, msg)
        print(f"[ERROR] {msg}", file=sys.stderr)
        return {"target": target, "status": "ERROR", "message": msg}

    # --- Chain selection ---
    chain, chain_ambiguous = choose_chain(residues, preferred=chain_pref)
    if chain is None:
        msg = "Could not determine chain"
        _write_error_report(out_dir / f"{target}_validation.txt", target, msg)
        print(f"[ERROR] {msg}", file=sys.stderr)
        return {"target": target, "status": "ERROR", "message": msg}

    # --- Collect and optionally trim residue keys ---
    keys = sorted_keys(residues, chain)
    if resrange:
        keys = apply_resrange(keys, resrange)
    if not keys:
        msg = f"No residues remain after applying range {resrange}"
        _write_error_report(out_dir / f"{target}_validation.txt", target, msg)
        return {"target": target, "status": "ERROR", "message": msg}

    keys_set = set(keys)

    # --- Validation ---
    gaps       = check_continuity(keys)
    missing_c3 = check_c3_completeness(keys, c3_coords)

    flags = []
    if chain_ambiguous:
        flags.append("Multiple RNA chains of similar size — chain selection may be wrong")
    if gaps:
        flags.append(f"{len(gaps)} gap(s) in residue numbering")
    if missing_c3:
        flags.append(f"{len(missing_c3)} residue(s) missing C3' atom")

    needs_review = bool(flags)

    trimmed_from = keys[0][1]
    trimmed_to   = keys[-1][1]

    # --- Sequence ---
    seq_list = [RNA_BASE.get(residues[k][0].upper(), "N") for k in keys]
    if "N" in seq_list:
        flags.append("Unknown residue name(s) mapped to 'N' — check RES2TYPE")
    sequence = "".join(seq_list)

    # --- Coords for residues that have C3' ---
    coords_available = [k for k in keys if k in c3_coords]
    usable_keys   = coords_available if missing_c3 else keys
    usable_seq    = "".join(RNA_BASE.get(residues[k][0].upper(), "N") for k in usable_keys)
    usable_coords = [c3_coords[k] for k in usable_keys]

    # --- Write outputs ---
    write_validation_report(
        out_dir / f"{target}_validation.txt",
        target, chain, keys, missing_c3, gaps,
        chain_ambiguous, trimmed_from, trimmed_to, flags,
    )

    write_full_atom_pdb(
        full_atoms, chain, keys_set,
        out_dir / f"{target}_rna_full.pdb",
    )

    # Only write C3'/LAMMPS/sequence if we have at least partial coords
    if usable_keys:
        write_c3_pdb(usable_keys, c3_coords, chain, out_dir / f"{target}_c3only.pdb")
        write_lammps_dat(usable_seq, usable_coords, out_dir / f"{target}.dat")
        (out_dir / f"{target}_sequence.txt").write_text(usable_seq + "\n")
    else:
        flags.append("No C3' coordinates available — .dat and c3only.pdb not written")
        needs_review = True

    metadata = {
        "target":          target,
        "source_file":     str(pdb_path),
        "chain":           chain,
        "chain_ambiguous": chain_ambiguous,
        "residue_range":   [trimmed_from, trimmed_to],
        "n_residues":      len(keys),
        "n_usable":        len(usable_keys),
        "sequence":        usable_seq,
        "missing_c3":      len(missing_c3),
        "gaps":            [(a, b) for a, b in gaps],
        "flags":           flags,
        "status":          "NEEDS_REVIEW" if needs_review else "PASS",
        "generated":       datetime.now().isoformat(),
    }
    (out_dir / f"{target}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    status_tag = "[REVIEW]" if needs_review else "[OK]"
    print(f"{status_tag} {target}  chain={chain}  {len(usable_keys)} nt  ->  {out_dir}")
    if flags:
        for f in flags:
            print(f"         ^ {f}")

    return metadata


def _write_error_report(path: Path, target, message):
    with open(path, "w") as fh:
        fh.write(f"Validation report — {target}\n")
        fh.write(f"Generated: {datetime.now().isoformat()}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"ERROR: {message}\n\nOverall status : ERROR\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Week 13 — prepare a single RNA target for LAMMPS simulation"
    )
    ap.add_argument("--pdb",      required=True, help="Input PDB/mmCIF path (gz ok)")
    ap.add_argument("--out",      required=True, help="Output directory for this target")
    ap.add_argument("--target",   default=None,  help="Target name (default: stem of --pdb)")
    ap.add_argument("--chain",    default=None,  help="Chain ID to extract (auto-pick if omitted)")
    ap.add_argument("--resrange", default=None,  help="Residue range to keep, e.g. 5:80")
    args = ap.parse_args()

    pdb_path = Path(args.pdb)
    out_dir  = Path(args.out)
    target   = args.target or pdb_path.stem.split(".")[0]

    resrange = None
    if args.resrange:
        parts = args.resrange.split(":")
        if len(parts) != 2:
            ap.error("--resrange must be in the form START:END, e.g. 5:80")
        resrange = (int(parts[0]), int(parts[1]))

    result = prepare_target(pdb_path, out_dir, target,
                             chain_pref=args.chain, resrange=resrange)

    if result.get("status") == "ERROR":
        sys.exit(2)
    if result.get("status") == "NEEDS_REVIEW":
        sys.exit(1)


if __name__ == "__main__":
    main()
