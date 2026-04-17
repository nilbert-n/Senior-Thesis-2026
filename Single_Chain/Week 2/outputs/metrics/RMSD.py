# Mean RMSD (Å) from metrics CSV
python - <<'PY' "metrics_config_1_2DER.pdb1_C.dat_298.csv"
import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1])
vals = df["RMSD"].dropna().to_numpy()
print(f"{np.nanmean(vals):.3f}" if vals.size else "NaN")
PY

# Last RMSD (Å)
python - <<'PY' "metrics_config_1_2DER.pdb1_C.dat_298.csv"
import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1]); s = df["RMSD"].dropna()
print(f"{s.iloc[-1]:.3f}" if len(s) else "NaN")
PY
