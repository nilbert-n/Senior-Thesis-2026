#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np

def center_dump(in_path, out_path):
    # Ensure we can pass Path objects
    in_path = str(in_path)
    out_path = str(out_path)

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        while True:
            line = fin.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                # pass through any preamble (unlikely)
                fout.write(line)
                continue

            # TIMESTEP
            fout.write(line)
            ts = fin.readline()
            if not ts:
                break
            fout.write(ts)

            # NUMBER OF ATOMS
            hdr = fin.readline(); fout.write(hdr)
            n = int(fin.readline()); fout.write(f"{n}\n")

            # BOX BOUNDS header + 3 lines (preserve verbatim)
            box_hdr = fin.readline(); fout.write(box_hdr)
            for _ in range(3):
                fout.write(fin.readline())

            # ATOMS header
            atoms_hdr = fin.readline()
            if not atoms_hdr.startswith("ITEM: ATOMS"):
                raise RuntimeError("Expected 'ITEM: ATOMS' line.")
            fout.write(atoms_hdr)
            cols = atoms_hdr.strip().split()[2:]

            # Choose which coords to center
            if set(("xu", "yu", "zu")).issubset(cols):
                cx, cy, cz = (cols.index("xu"), cols.index("yu"), cols.index("zu"))
            elif set(("x", "y", "z")).issubset(cols):
                cx, cy, cz = (cols.index("x"), cols.index("y"), cols.index("z"))
            else:
                # no coords? just copy block
                for _ in range(n):
                    fout.write(fin.readline())
                continue

            # Read all atoms, compute centroid
            rows = [fin.readline().rstrip("\n").split() for _ in range(n)]
            coords = np.array(
                [[float(r[cx]), float(r[cy]), float(r[cz])] for r in rows],
                dtype=float,
            )
            cen = coords.mean(axis=0)

            # Write centered rows (preserve all other columns)
            for r in rows:
                r[cx] = f"{float(r[cx]) - cen[0]:.6f}"
                r[cy] = f"{float(r[cy]) - cen[1]:.6f}"
                r[cz] = f"{float(r[cz]) - cen[2]:.6f}"
                fout.write(" ".join(r) + "\n")


# ---- Batch driver: apply to all lammpstrj files ----

# Base directory: parent of outputs_50

in_dir = Path("outputs_200")          # relative to ran_seq
out_dir = Path("outputs_200") / "outputs_centreded"  # will be created inside ran_seq

out_dir.mkdir(exist_ok=True)

for in_path in sorted(in_dir.glob("config_*.dat.lammpstrj")):
    out_path = out_dir / in_path.name
    print(f"Centering {in_path.name} -> {out_path}")
    center_dump(in_path, out_path)

print("Done centering all configs.")
