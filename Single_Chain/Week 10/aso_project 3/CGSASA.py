import numpy as np
import matplotlib.pyplot as plt
import freesasa

PDB_FILE = "1YMO.pdb1"
CHAIN = "A"
START_RES = 15
END_RES = 46

# choose one for highlighting
LOOP_CORE = set(range(13, 21))      # 13–20
LOOP_BROAD = set(range(10, 24))     # 10–23

USE_LOOP = LOOP_CORE   # swap to LOOP_BROAD if desired

# ---- calculate SASA ----
structure = freesasa.Structure(PDB_FILE)
result = freesasa.calc(structure)
residue_areas = result.residueAreas()

resnums_abs = []
resnums_rel = []
sasa_vals = []

for resnum_str in sorted(residue_areas[CHAIN].keys(), key=int):
    resnum = int(resnum_str)
    if START_RES <= resnum <= END_RES:
        rel = resnum - START_RES + 1
        resnums_abs.append(resnum)
        resnums_rel.append(rel)
        sasa_vals.append(residue_areas[CHAIN][resnum_str].total)

resnums_rel = np.array(resnums_rel)
sasa_vals = np.array(sasa_vals)

if len(resnums_rel) != 32:
    raise ValueError(f"Expected 32 residues, got {len(resnums_rel)}")

# ---- define regions ----
region_colors = []
for r in resnums_rel:
    if r in USE_LOOP:
        region_colors.append("#F4A261")   # loop
    else:
        region_colors.append("#457B9D")   # stem/other

# ---- plot ----
plt.figure(figsize=(8, 4.8))

# background highlight for loop
loop_start = min(USE_LOOP) - 0.5
loop_end = max(USE_LOOP) + 0.5
plt.axvspan(loop_start, loop_end, color="#F4A261", alpha=0.15, label="Loop region")

# bars + line
plt.bar(resnums_rel, sasa_vals, color=region_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
plt.plot(resnums_rel, sasa_vals, color="black", linewidth=1.4, marker="o", markersize=3)

# annotate top 3 exposed residues
top3 = np.argsort(sasa_vals)[-3:][::-1]
for idx in top3:
    plt.text(resnums_rel[idx], sasa_vals[idx] + 2, str(resnums_rel[idx]),
             ha="center", va="bottom", fontsize=8)

plt.xlabel("Hairpin residue")
plt.ylabel("SASA (Å²)")
plt.title("Fig. 4.6  SASA vs residue for 1YMO-derived hairpin reference")
plt.xlim(0.5, 32.5)
plt.grid(axis="y", alpha=0.25, linestyle="--")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig_4_6_sasa_vs_residue.png", dpi=300)
plt.show()

# summary
print(f"Mean SASA: {sasa_vals.mean():.2f} Å²")
print(f"Max SASA:  {sasa_vals.max():.2f} Å² at residue {resnums_rel[np.argmax(sasa_vals)]}")

loop_mask = np.array([r in USE_LOOP for r in resnums_rel])
print(f"Loop mean SASA: {sasa_vals[loop_mask].mean():.2f} Å²")
print(f"Non-loop mean SASA: {sasa_vals[~loop_mask].mean():.2f} Å²")