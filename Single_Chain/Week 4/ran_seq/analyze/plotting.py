import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Run from: /scratch/gpfs/JERELLE/nilbert/Single_Chain/analyze
BASE_DIR = "."  # current folder

# --- helper: find an RMSD column robustly ---
def infer_rmsd_col(df: pd.DataFrame) -> str:
    candidates = [
        "RMSD", "rmsd",
        "RMSD_reg", "rmsd_reg",
        "RMSD_enh", "rmsd_enh",
        "dRMSD", "dRmsd", "drmsd"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: pick a column containing 'rmsd'
    for c in df.columns:
        if "rmsd" in c.lower():
            return c
    raise ValueError(f"Couldn't find an RMSD column. Columns: {list(df.columns)}")

def mean_rmsd_for_files(paths):
    """Return list of per-file mean RMSD values (each file treated as one replicate)."""
    vals = []
    for p in paths:
        df = pd.read_csv(p)
        col = infer_rmsd_col(df)
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        vals.append(float(s.mean()))
    return vals

# --- gather files by length ---
patterns = {
    50:  os.path.join(BASE_DIR, "metrics_config_*_len50_*.dat.csv"),
    100: os.path.join(BASE_DIR, "metrics_config_*_len100_*.dat.csv"),
    200: os.path.join(BASE_DIR, "metrics_config_*_len200_*.dat.csv"),
}

file_lists = {L: sorted(glob.glob(pat)) for L, pat in patterns.items()}

for L, files in file_lists.items():
    if not files:
        raise FileNotFoundError(f"No files found for length {L} with pattern: {patterns[L]}")

# --- compute mean + SE across files for each length ---
lengths = []
means = []
ses = []
ns = []

for L in [50, 100, 200]:
    per_file_means = mean_rmsd_for_files(file_lists[L])
    if len(per_file_means) == 0:
        raise ValueError(f"All RMSD values were empty/NaN for length {L}.")

    mu = float(np.mean(per_file_means))
    se = float(np.std(per_file_means, ddof=1) / np.sqrt(len(per_file_means))) if len(per_file_means) > 1 else 0.0

    lengths.append(L)
    means.append(mu)
    ses.append(se)
    ns.append(len(per_file_means))

print("Summary (mean RMSD across files):")
for L, mu, se, n in zip(lengths, means, ses, ns):
    print(f"  len={L:3d} nt: mean={mu:.4f}, SE={se:.4f}, n_files={n}")

# --- horizontal bar plot ---
plt.figure()
y_labels = [f"{L} nt (n={n})" for L, n in zip(lengths, ns)]
plt.barh(y_labels, means, xerr=ses, capsize=4)
plt.xlabel("Mean RMSD")
plt.ylabel("Length")
plt.title("Mean RMSD vs RNA length")
plt.tight_layout()

outpath = os.path.join(BASE_DIR, "mean_RMSD_vs_length.png")
plt.savefig(outpath, dpi=200)
plt.show()

print(f"Saved: {outpath}")
