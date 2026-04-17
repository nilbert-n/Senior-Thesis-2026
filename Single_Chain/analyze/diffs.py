import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) Find the 16 CSV files (8 metrics + 8 PE) in the current folder
# -----------------------------
metrics_paths = sorted(glob.glob("diff_metrics_config_*.dat.csv"))
pe_paths      = sorted(glob.glob("diff_pe_config_*.dat.csv"))

if not metrics_paths:
    raise FileNotFoundError("No files matched: diff_metrics_config_*.dat.csv (are you in analyze/?)")
if not pe_paths:
    raise FileNotFoundError("No files matched: diff_pe_config_*.dat.csv (are you in analyze/?)")

# -----------------------------
# 1) Helpers
# -----------------------------
FNAME_RE = re.compile(r"diff_(metrics|pe)_config_(\d+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)\.dat\.csv$")

def parse_key_label(path: str):
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        raise ValueError(f"Unexpected filename format: {base}")
    kind = m.group(1)
    cfg  = int(m.group(2))
    pdb  = m.group(3)
    chain = m.group(4)
    key = (cfg, pdb, chain)
    label = f"{cfg}:{pdb}_{chain}"
    return kind, key, label

def mean_se(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    mu = float(s.mean()) if n else np.nan
    se = float(s.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mu, se

def infer_pe_column(df: pd.DataFrame) -> str:
    # Try common names first
    for c in ["dPE", "dPe", "deltaPE", "d_pe", "PE_diff", "pe_diff", "PE", "pe", "potential_energy"]:
        if c in df.columns:
            return c
    # Fallback: pick the only numeric column (excluding obvious index/time columns)
    exclude = {"timestep", "step", "time", "frame", "idx", "index"}
    numeric_cols = [
        c for c in df.columns
        if c.lower() not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(f"Couldn't infer PE column. Columns: {list(df.columns)}")

def barplot(labels, means, ses, title, ylabel, outpath):
    plt.figure()
    plt.bar(labels, means, yerr=ses, capsize=4)
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Config")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()

# -----------------------------
# 2) Summarize metrics (dRg, dRMSD)
# -----------------------------
metrics = {}
for p in metrics_paths:
    _, key, label = parse_key_label(p)
    df = pd.read_csv(p)

    if "dRg" not in df.columns or "dRMSD" not in df.columns:
        raise ValueError(f"{os.path.basename(p)} must contain columns dRg and dRMSD. Found: {list(df.columns)}")

    mu_rg, se_rg = mean_se(df["dRg"])
    mu_rmsd, se_rmsd = mean_se(df["dRMSD"])

    metrics[key] = {
        "label": label,
        "mean_dRg": mu_rg, "se_dRg": se_rg,
        "mean_dRMSD": mu_rmsd, "se_dRMSD": se_rmsd,
    }

# -----------------------------
# 3) Summarize PE
# -----------------------------
pes = {}
for p in pe_paths:
    _, key, label = parse_key_label(p)
    df = pd.read_csv(p)

    pe_col = infer_pe_column(df)
    mu_pe, se_pe = mean_se(df[pe_col])

    pes[key] = {
        "label": label,
        "mean_dPE": mu_pe, "se_dPE": se_pe,
        "pe_col_used": pe_col,
    }

# -----------------------------
# 4) Merge on keys (expects 8 matches)
# -----------------------------
keys = sorted(set(metrics.keys()) & set(pes.keys()))
if not keys:
    raise ValueError("No matching (config, pdb, chain) keys between metrics and PE files.")

# Warn if not exactly 8 matches
if len(keys) != 8:
    print(f"WARNING: matched {len(keys)} configs (expected 8).")

labels = [metrics[k]["label"] for k in keys]

# -----------------------------
# 5) Make 3 plots + save PNGs in analyze/
# -----------------------------
barplot(
    labels,
    [metrics[k]["mean_dRg"] for k in keys],
    [metrics[k]["se_dRg"] for k in keys],
    "Mean ΔRg (enh - reg) by config",
    "ΔRg",
    "bar_mean_dRg.png",
)

barplot(
    labels,
    [metrics[k]["mean_dRMSD"] for k in keys],
    [metrics[k]["se_dRMSD"] for k in keys],
    "Mean ΔRMSD (enh - reg) by config",
    "ΔRMSD",
    "bar_mean_dRMSD.png",
)

barplot(
    labels,
    [pes[k]["mean_dPE"] for k in keys],
    [pes[k]["se_dPE"] for k in keys],
    "Mean ΔPE (enh - reg) by config",
    "ΔPE",
    "bar_mean_dPE.png",
)

print("Saved: bar_mean_dRg.png, bar_mean_dRMSD.png, bar_mean_dPE.png")
# Optional debug:
# for k in keys: print(metrics[k]["label"], "PE col:", pes[k]["pe_col_used"])
