#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Run 6-panel analysis on all ASO-Hairpin simulations
# ═══════════════════════════════════════════════════════════════
# 
# SETUP: Place this script in your aso_project/ directory
#        alongside configs/ and outputs/ folders.
#
# Usage:
#   bash run_analysis.sh            # analyze all simulations
#   bash run_analysis.sh --single   # analyze just one (edit below)
# ═══════════════════════════════════════════════════════════════

set -e

module purge
module module load anaconda3/2025.6 || true

mkdir -p analysis

if [ "$1" == "--single" ]; then
    # Single simulation (edit these paths)
    python3 run_all_analysis.py \
        --dat configs/100ASO_unmod.dat \
        --traj outputs/100ASO_unmod.lammpstrj \
        --thermo outputs/thermo_100ASO_unmod_production.dat \
        --outdir analysis
else
    # All simulations (auto-discovery)
    python3 run_all_analysis.py --base . --outdir analysis
fi

echo ""
echo "Figures saved in analysis/"
ls -la analysis/fig_*.png 2>/dev/null
echo ""
echo "Summary: analysis/campaign_summary.csv"
