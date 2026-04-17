#!/usr/bin/env python3
"""
Per-residue SASA for a folded-reference hairpin segment from a PDB.

Typical use for this project:
    python sasa_hairpin_ref.py \
        --pdb 1YMO.pdb1 \
        --chain A \
        --start-res 15 \
        --end-res 46 \
        --loop 13-20 \
        --outdir sasa_fig46

This extracts chain A residues 15-46 from 1YMO, renumbers them 1-32,
computes per-residue SASA using FreeSASA, and produces a thesis-ready
"SASA vs residue" plot.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import freesasa
except ImportError as exc:
    raise SystemExit(
        "FreeSASA Python bindings are required. Try: pip install freesasa\n"
        "If that fails on the cluster, load/install FreeSASA in your environment first."
    ) from exc


def parse_range_list(spec: str) -> list[int]:
    """Parse strings like '13-20' or '1-5,8,10-12' into sorted unique ints."""
    vals: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                start, end = end, start
            vals.update(range(start, end + 1))
        else:
            vals.add(int(part))
    out = sorted(vals)
    if not out:
        raise ValueError("Empty residue range specification")
    return out


def extract_segment_sasa(pdb_file: str, chain: str, start_res: int, end_res: int):
    structure = freesasa.Structure(pdb_file)
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()

    if chain not in residue_areas:
        raise ValueError(f"Chain {chain!r} not found in {pdb_file}")

    rel_res = []
    abs_res = []
    sasa_total = []
    sasa_polar = []
    sasa_apolar = []
    sasa_main = []
    sasa_side = []

    chain_map = residue_areas[chain]
    for res_str in sorted(chain_map.keys(), key=int):
        abs_idx = int(res_str)
        if start_res <= abs_idx <= end_res:
            area = chain_map[res_str]
            rel_idx = abs_idx - start_res + 1
            abs_res.append(abs_idx)
            rel_res.append(rel_idx)
            sasa_total.append(area.total)
            sasa_polar.append(area.polar)
            sasa_apolar.append(area.apolar)
            sasa_main.append(area.mainChain)
            sasa_side.append(area.sideChain)

    if not rel_res:
        raise ValueError(
            f"No residues found for chain {chain} in range {start_res}-{end_res}"
        )

    rel_arr = np.array(rel_res, dtype=int)
    abs_arr = np.array(abs_res, dtype=int)
    total_arr = np.array(sasa_total, dtype=float)
    polar_arr = np.array(sasa_polar, dtype=float)
    apolar_arr = np.array(sasa_apolar, dtype=float)
    main_arr = np.array(sasa_main, dtype=float)
    side_arr = np.array(sasa_side, dtype=float)

    expected = end_res - start_res + 1
    if len(rel_arr) != expected:
        raise ValueError(
            f"Expected {expected} residues from {start_res}-{end_res}, got {len(rel_arr)}."
        )

    return {
        "rel_res": rel_arr,
        "abs_res": abs_arr,
        "total": total_arr,
        "polar": polar_arr,
        "apolar": apolar_arr,
        "main": main_arr,
        "side": side_arr,
        "total_sasa_structure": result.totalArea(),
    }


def write_csv(path: str, data: dict, loop_set: set[int]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "relative_residue",
            "absolute_residue",
            "region",
            "sasa_total_A2",
            "sasa_polar_A2",
            "sasa_apolar_A2",
            "sasa_main_chain_A2",
            "sasa_side_chain_A2",
        ])
        for rr, ar, tot, pol, apo, main, side in zip(
            data["rel_res"],
            data["abs_res"],
            data["total"],
            data["polar"],
            data["apolar"],
            data["main"],
            data["side"],
        ):
            region = "loop" if rr in loop_set else "non-loop"
            writer.writerow([
                rr, ar, region,
                f"{tot:.6f}", f"{pol:.6f}", f"{apo:.6f}", f"{main:.6f}", f"{side:.6f}"
            ])


def write_summary(
    path: str,
    data: dict,
    loop_set: set[int],
    chain: str,
    start_res: int,
    end_res: int,
    title: str,
) -> None:
    rel = data["rel_res"]
    total = data["total"]
    loop_mask = np.array([r in loop_set for r in rel], dtype=bool)
    non_loop_mask = ~loop_mask
    top3_idx = np.argsort(total)[-3:][::-1]

    with open(path, "w") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Segment: chain {chain}, absolute residues {start_res}-{end_res}\n")
        f.write(f"Renumbered residues: 1-{len(rel)}\n")
        f.write(f"Loop residues: {sorted(loop_set)}\n")
        f.write(f"Mean SASA: {total.mean():.4f} A^2\n")
        f.write(f"Std SASA: {total.std():.4f} A^2\n")
        f.write(f"Min SASA: {total.min():.4f} A^2 at residue {rel[np.argmin(total)]}\n")
        f.write(f"Max SASA: {total.max():.4f} A^2 at residue {rel[np.argmax(total)]}\n")
        if loop_mask.any():
            f.write(f"Loop mean SASA: {total[loop_mask].mean():.4f} A^2\n")
            f.write(f"Loop max SASA: {total[loop_mask].max():.4f} A^2\n")
        if non_loop_mask.any():
            f.write(f"Non-loop mean SASA: {total[non_loop_mask].mean():.4f} A^2\n")
        f.write("Top 3 exposed residues:\n")
        for idx in top3_idx:
            f.write(
                f"  Residue {rel[idx]} (abs {data['abs_res'][idx]}): {total[idx]:.4f} A^2\n"
            )


def make_plot(path: str, data: dict, loop_set: set[int], title: str) -> None:
    rel = data["rel_res"]
    total = data["total"]

    bar_colors = ["#F4A261" if r in loop_set else "#4C78A8" for r in rel]

    plt.figure(figsize=(8.4, 4.8))

    if loop_set:
        plt.axvspan(
            min(loop_set) - 0.5,
            max(loop_set) + 0.5,
            color="#F4A261",
            alpha=0.14,
            label="Loop region",
        )

    plt.bar(
        rel,
        total,
        width=0.82,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.88,
        color=bar_colors,
    )
    plt.plot(rel, total, marker="o", linewidth=1.3, markersize=3.2, color="black")

    top3_idx = np.argsort(total)[-3:][::-1]
    for idx in top3_idx:
        plt.text(
            rel[idx],
            total[idx] + max(total) * 0.025,
            str(rel[idx]),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xlabel("Hairpin residue")
    plt.ylabel(r"SASA ($\AA^2$)")
    plt.title(title)
    plt.xlim(0.4, rel[-1] + 0.6)
    plt.ylim(0, max(total) * 1.15)
    plt.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if loop_set:
        plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute SASA vs residue for a hairpin segment from a reference PDB"
    )
    p.add_argument("--pdb", required=True, help="Input PDB file, e.g. 1YMO.pdb1")
    p.add_argument("--chain", default="A", help="Chain to analyze (default: A)")
    p.add_argument(
        "--start-res",
        type=int,
        default=15,
        help="Absolute start residue in PDB numbering",
    )
    p.add_argument(
        "--end-res",
        type=int,
        default=46,
        help="Absolute end residue in PDB numbering",
    )
    p.add_argument(
        "--loop",
        default="13-20",
        help="Relative loop residues after renumbering, e.g. 13-20 or 10-23",
    )
    p.add_argument("--outdir", default="sasa_fig46", help="Output directory")
    p.add_argument(
        "--title",
        default="Fig. 4.6  SASA vs residue for 1YMO-derived hairpin reference",
        help="Plot title",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    loop_res = parse_range_list(args.loop)
    loop_set = set(loop_res)

    data = extract_segment_sasa(args.pdb, args.chain, args.start_res, args.end_res)

    n_res = len(data["rel_res"])
    bad = [r for r in loop_res if r < 1 or r > n_res]
    if bad:
        raise ValueError(f"Loop residues out of range 1-{n_res}: {bad}")

    png_path = os.path.join(args.outdir, "fig_4_6_sasa_vs_residue.png")
    csv_path = os.path.join(args.outdir, "fig_4_6_sasa_vs_residue.csv")
    txt_path = os.path.join(args.outdir, "fig_4_6_sasa_summary.txt")

    make_plot(png_path, data, loop_set, args.title)
    write_csv(csv_path, data, loop_set)
    write_summary(
        txt_path,
        data,
        loop_set,
        args.chain,
        args.start_res,
        args.end_res,
        args.title,
    )

    rel = data["rel_res"]
    total = data["total"]
    loop_mask = np.array([r in loop_set for r in rel], dtype=bool)

    print("=" * 70)
    print("SASA VS RESIDUE ANALYSIS")
    print("=" * 70)
    print(f"PDB:             {args.pdb}")
    print(f"Chain:           {args.chain}")
    print(f"Segment:         {args.start_res}-{args.end_res} (renumbered 1-{len(rel)})")
    print(f"Loop residues:   {sorted(loop_set)}")
    print(f"Mean SASA:       {total.mean():.3f} A^2")
    print(f"Max SASA:        {total.max():.3f} A^2 at residue {rel[np.argmax(total)]}")
    if loop_mask.any():
        print(f"Loop mean SASA:  {total[loop_mask].mean():.3f} A^2")
    print("\nSaved:")
    print(f"  {png_path}")
    print(f"  {csv_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()