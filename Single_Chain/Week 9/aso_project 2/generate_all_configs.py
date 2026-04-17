#!/usr/bin/env python3
"""
=============================================================================
ASO-Hairpin Simulation Config Generator
=============================================================================
Generates LAMMPS data files for:
  1. Concentration sweep  (25, 50, 100, 200 ASOs)
  2. Different ASO designs (unmodified, truncated, AAtoCC, extended, mismatch)
  3. All configs use the same parent PDB: 1YMO hairpin

Atom type mapping (coarse-grained, 1 bead per nucleotide):
  Type 1 = A  (329.20 Da)
  Type 2 = C  (305.20 Da)
  Type 3 = G  (345.20 Da)
  Type 4 = U  (306.20 Da)

Reference sequences from PDB 1YMO pseudoknot → ASO-hairpin:
  ASO  (mol 1, atoms 1-10):  G G G C U G U U U U   (5'→3')
  Hairpin (mol 2, atoms 11-42): 32 residues
=============================================================================
"""

import random
import math
import os
import copy

# ── Atom type map ──
TYPE_MAP = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
MASS_MAP = {1: 329.20, 2: 305.20, 3: 345.20, 4: 306.20}

# ── Template: read base config ──
def read_config(filepath):
    """Parse a LAMMPS data file and return atoms, bonds, angles, masses."""
    atoms, bonds, angles, masses = [], [], [], []
    section = None
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                section_just_set = False
                continue
            # Detect section headers (single-word lines)
            if stripped == "Atoms":
                section = "atoms"; continue
            if stripped == "Bonds":
                section = "bonds"; continue
            if stripped == "Angles":
                section = "angles"; continue
            if stripped == "Masses":
                section = "masses"; continue
            # Skip header lines
            if any(kw in stripped for kw in ["xlo", "ylo", "zlo", "atom types",
                    "bond types", "angle types", "LAMMPS"]):
                continue
            if stripped.endswith("atoms") or stripped.endswith("bonds") or stripped.endswith("angles"):
                continue

            parts = stripped.split()
            if section == "masses" and len(parts) >= 2:
                masses.append(line)
            elif section == "atoms" and len(parts) >= 7:
                atoms.append(parts)
            elif section == "bonds" and len(parts) >= 4:
                bonds.append(parts)
            elif section == "angles" and len(parts) >= 5:
                angles.append(parts)
    return atoms, bonds, angles, masses


def compute_center(atoms):
    """Compute centroid of a list of atom records."""
    n = len(atoms)
    cx = sum(float(a[4]) for a in atoms) / n
    cy = sum(float(a[5]) for a in atoms) / n
    cz = sum(float(a[6]) for a in atoms) / n
    return cx, cy, cz


def box_size_for_concentration(n_aso, n_beads_per_aso=10, target_conc_mM=1.0):
    """
    Compute cubic box half-length for a target molar concentration.
    For coarse-grained sims, we use a practical heuristic:
    scale box so particles have enough room but collisions are meaningful.
    
    Heuristic: box_half = max(80, scale_factor * n_aso^(1/3))
    """
    # Practical scaling: ensure ~15 Å separation minimum
    # V = N / (Na * c) in liters; convert to Å
    # For 1 mM: V(L) = N / (6.022e23 * 1e-3) = N / 6.022e20
    # V(Å³) = V(L) * 1e30 / 1e3 = V(L) * 1e27
    # L = V^(1/3)
    Na = 6.022e23
    c_mol_per_L = target_conc_mM * 1e-3
    V_L = n_aso / (Na * c_mol_per_L)
    V_A3 = V_L * 1e27
    L = V_A3 ** (1.0/3.0)
    half = L / 2.0
    # Minimum box: 80 Å half-length (160 Å box)
    return max(80.0, half)


def create_aso_sequence(variant='unmodified'):
    """
    Return list of atom types for different ASO designs.
    Based on 1YMO pseudoknot ASO arm.
    """
    if variant == 'unmodified':
        # Original 10-mer: G G G C U G U U U U
        return [3, 3, 3, 2, 4, 3, 4, 4, 4, 4]
    
    elif variant == 'truncated':
        # 7-mer (paper): remove bases 8-10 → G G G C U G U
        return [3, 3, 3, 2, 4, 3, 4]
    
    elif variant == 'extended_12mer':
        # Extended 12-mer: add A A to 3' end → G G G C U G U U U U A A
        return [3, 3, 3, 2, 4, 3, 4, 4, 4, 4, 1, 1]
    
    elif variant == 'extended_14mer':
        # Extended 14-mer: G G G C U G U U U U A A G U
        return [3, 3, 3, 2, 4, 3, 4, 4, 4, 4, 1, 1, 3, 4]
    
    elif variant == 'mismatch_G6A':
        # Single mismatch at position 6: G→A
        # G G G C U A U U U U  (disrupts G6 triple bp)
        return [3, 3, 3, 2, 4, 1, 4, 4, 4, 4]
    
    elif variant == 'mismatch_U5C':
        # Single mismatch at position 5: U→C
        # G G G C C G U U U U  (disrupts U5 triple bp)
        return [3, 3, 3, 2, 2, 3, 4, 4, 4, 4]
    
    elif variant == 'all_purine':
        # Purine-rich ASO: G G G A G G A A A A
        return [3, 3, 3, 1, 3, 3, 1, 1, 1, 1]
    
    elif variant == 'scrambled':
        # Same composition as unmodified, scrambled order
        # U G U G U C G U G U
        return [4, 3, 4, 3, 4, 2, 3, 4, 3, 4]
    
    else:
        raise ValueError(f"Unknown ASO variant: {variant}")


def create_hairpin_sequence(variant='unmodified'):
    """
    Return list of 32 atom types for the hairpin (atoms 11-42).
    """
    # Original hairpin from 1YMO:
    # G C U G A C U U U C A G C C C C A A A C A A A A A A G U C A G C
    base = [3, 2, 4, 3, 1, 2, 4, 4, 4, 2,  # 11-20
            1, 3, 2, 2, 2, 2, 1, 1, 1, 2,  # 21-30
            1, 1, 1, 1, 1, 1, 3, 4, 2, 1,  # 31-40
            3, 2]                            # 41-42
    
    if variant == 'unmodified':
        return base
    
    elif variant == 'AAtoCC':
        # Mutate A31→C, A32→C (indices 20,21 in 0-based from hairpin start)
        mutated = base.copy()
        mutated[20] = 2  # A31 → C31
        mutated[21] = 2  # A32 → C32
        return mutated
    
    elif variant == 'loop_GG_to_AA':
        # Mutate G22→A, (index 11 in 0-based from hairpin start)
        mutated = base.copy()
        mutated[11] = 1  # G22 → A22
        return mutated
    
    else:
        return base


def generate_aso_coords_from_template(template_atoms, new_types):
    """
    Generate coordinates for ASO with potentially different length.
    Uses template positions, extends/truncates as needed.
    """
    result = []
    n_template = len(template_atoms)
    n_new = len(new_types)
    
    for i in range(n_new):
        if i < n_template:
            # Use template position
            x = float(template_atoms[i][4])
            y = float(template_atoms[i][5])
            z = float(template_atoms[i][6])
        else:
            # Extend: extrapolate from last two positions
            dx = float(template_atoms[-1][4]) - float(template_atoms[-2][4])
            dy = float(template_atoms[-1][5]) - float(template_atoms[-2][5])
            dz = float(template_atoms[-1][6]) - float(template_atoms[-2][6])
            extra = i - n_template + 1
            x = float(template_atoms[-1][4]) + dx * extra
            y = float(template_atoms[-1][5]) + dy * extra
            z = float(template_atoms[-1][6]) + dz * extra
        
        result.append((new_types[i], x, y, z))
    
    return result


def scatter_positions(n_positions, box_half, exclusion_radius=40.0, min_sep=15.0):
    """Place n_positions random points in box avoiding center and each other."""
    positions = []
    max_attempts = n_positions * 1000
    attempts = 0
    while len(positions) < n_positions and attempts < max_attempts:
        attempts += 1
        px = random.uniform(-box_half + 10, box_half - 10)
        py = random.uniform(-box_half + 10, box_half - 10)
        pz = random.uniform(-box_half + 10, box_half - 10)
        
        if math.sqrt(px**2 + py**2 + pz**2) < exclusion_radius:
            continue
        
        conflict = False
        for (ex, ey, ez) in positions:
            if math.sqrt((px-ex)**2 + (py-ey)**2 + (pz-ez)**2) < min_sep:
                conflict = True
                break
        
        if not conflict:
            positions.append((px, py, pz))
    
    if len(positions) < n_positions:
        print(f"  WARNING: Only placed {len(positions)}/{n_positions} ASOs (box may be too small)")
    
    return positions


def write_lammps_data(filename, title, aso_types, hairpin_types,
                      template_atoms, template_bonds, template_angles,
                      n_free_aso, box_half):
    """
    Write a complete LAMMPS data file with:
    - 1 docked ASO-hairpin complex (from template)
    - n_free_aso scattered free ASOs
    """
    
    # Extract template ASO and hairpin atoms
    aso_template = [a for a in template_atoms if int(a[1]) == 1]
    hairpin_template = [a for a in template_atoms if int(a[1]) == 2]
    
    # Generate ASO coordinates (may differ in length from template)
    aso_coords = generate_aso_coords_from_template(aso_template, aso_types)
    n_aso_beads = len(aso_types)
    n_hairpin_beads = len(hairpin_types)
    
    # Center of docked ASO for translation reference
    cx = sum(c[1] for c in aso_coords) / n_aso_beads
    cy = sum(c[2] for c in aso_coords) / n_aso_beads
    cz = sum(c[3] for c in aso_coords) / n_aso_beads
    
    # Scatter free ASOs
    positions = scatter_positions(n_free_aso, box_half)
    actual_free = len(positions)
    
    # Count totals
    total_atoms = n_aso_beads + n_hairpin_beads + actual_free * n_aso_beads
    total_bonds = (n_aso_beads - 1) + (n_hairpin_beads - 1) + actual_free * (n_aso_beads - 1)
    total_angles = max(0, n_aso_beads - 2) + max(0, n_hairpin_beads - 2) + actual_free * max(0, n_aso_beads - 2)
    
    out_atoms = []
    out_bonds = []
    out_angles = []
    
    # ── Molecule 1: Docked ASO ──
    atom_id = 0
    for i, (atype, x, y, z) in enumerate(aso_coords):
        atom_id += 1
        out_atoms.append(f"{atom_id} 1 {atype} 0.000000 {x:.6f} {y:.6f} {z:.6f}\n")
    
    # ── Molecule 2: Hairpin ──
    for i in range(n_hairpin_beads):
        atom_id += 1
        x = float(hairpin_template[i][4])
        y = float(hairpin_template[i][5])
        z = float(hairpin_template[i][6])
        out_atoms.append(f"{atom_id} 2 {hairpin_types[i]} 0.000000 {x:.6f} {y:.6f} {z:.6f}\n")
    
    # ── Bonds for docked complex ──
    bond_id = 0
    # ASO bonds
    for i in range(1, n_aso_beads):
        bond_id += 1
        out_bonds.append(f"{bond_id} 1 {i} {i+1}\n")
    # Hairpin bonds
    hp_start = n_aso_beads + 1
    for i in range(n_hairpin_beads - 1):
        bond_id += 1
        out_bonds.append(f"{bond_id} 1 {hp_start + i} {hp_start + i + 1}\n")
    
    # ── Angles for docked complex ──
    angle_id = 0
    # ASO angles
    for i in range(1, n_aso_beads - 1):
        angle_id += 1
        out_angles.append(f"{angle_id} 1 {i} {i+1} {i+2}\n")
    # Hairpin angles
    for i in range(n_hairpin_beads - 2):
        angle_id += 1
        out_angles.append(f"{angle_id} 1 {hp_start + i} {hp_start + i + 1} {hp_start + i + 2}\n")
    
    # ── Free ASOs (molecules 3, 4, ...) ──
    curr_mol = 2
    for pos in positions:
        curr_mol += 1
        first_atom_of_this_aso = atom_id + 1
        
        for i, (atype, tx, ty, tz) in enumerate(aso_coords):
            atom_id += 1
            nx = tx - cx + pos[0]
            ny = ty - cy + pos[1]
            nz = tz - cz + pos[2]
            out_atoms.append(f"{atom_id} {curr_mol} {atype} 0.000000 {nx:.6f} {ny:.6f} {nz:.6f}\n")
        
        for i in range(n_aso_beads - 1):
            bond_id += 1
            out_bonds.append(f"{bond_id} 1 {first_atom_of_this_aso + i} {first_atom_of_this_aso + i + 1}\n")
        
        for i in range(max(0, n_aso_beads - 2)):
            angle_id += 1
            out_angles.append(f"{angle_id} 1 {first_atom_of_this_aso + i} {first_atom_of_this_aso + i + 1} {first_atom_of_this_aso + i + 2}\n")
    
    # ── Write file ──
    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file - {title}\n\n")
        f.write(f"{atom_id} atoms\n")
        f.write(f"{bond_id} bonds\n")
        f.write(f"{angle_id} angles\n\n")
        f.write("4 atom types\n1 bond types\n1 angle types\n\n")
        f.write(f"{-box_half:.6f} {box_half:.6f} xlo xhi\n")
        f.write(f"{-box_half:.6f} {box_half:.6f} ylo yhi\n")
        f.write(f"{-box_half:.6f} {box_half:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t, m in MASS_MAP.items():
            f.write(f"{t} {m:.2f}\n")
        f.write("\nAtoms\n\n")
        f.writelines(out_atoms)
        f.write("\nBonds\n\n")
        f.writelines(out_bonds)
        f.write("\nAngles\n\n")
        f.writelines(out_angles)
    
    print(f"  Created {filename}: {atom_id} atoms, {bond_id} bonds, {angle_id} angles")
    return filename


def generate_lammps_input(filename, data_file, temperature, run_ns=1000,
                          ensemble='NVT', tag_suffix=''):
    """
    Generate a LAMMPS input file for the given conditions.
    
    Improvements over original NVE_4us.in:
    - NVT with Langevin thermostat (better for sampling)
    - Proper equilibration protocol
    - More frequent trajectory output for analysis
    """
    timestep = 10  # fs
    steps_per_ns = int(1e6 / timestep * 1000)  # 1e5 steps per ns with dt=10fs
    # Actually: 1 ns = 1e6 fs. steps = 1e6/10 = 1e5 per ns
    n_run = int(run_ns * 1e5)  # total steps
    n_eq = int(50 * 1e5)  # 100 ns equilibration
    
    tag = os.path.splitext(os.path.basename(data_file))[0]
    if tag_suffix:
        tag = f"{tag}_{tag_suffix}"
    
    dump_stride = 50000
    thermo_stride = 10000
    
    content = f"""# ===============================
# {ensemble} Simulation at {temperature}K
# Data: {data_file}
# Total: {run_ns} ns production + 50 ns equilibration
# ===============================

shell mkdir -p outputs logs

variable df      string {data_file}
variable T_target equal {temperature:.1f}
variable tag      string {tag}

units real
boundary p p p
atom_style full

read_data ${{df}}
log outputs/log.${{tag}}.lammps

# ===============================
# FORCE FIELD (same as original)
# ===============================
bond_style harmonic
bond_coeff 1 7.5 5.9

angle_style harmonic
angle_coeff 1 5.0 150.0

pair_style hybrid/overlay lj/cut 10.0 base/rna 18.0
pair_modify shift yes
pair_coeff * * lj/cut 2.0 8.9089871814

pair_coeff 1 1 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 1 2 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 1 3 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 1 4 base/rna 3.33333 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 2 2 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 2 3 base/rna 5.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 2 4 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 3 3 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 3 4 base/rna 3.33333 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345
pair_coeff 4 4 base/rna 0.00000 3.0 13.8 1.5 0.5 1.8326 0.9425 1.8326 1.1345

special_bonds lj 0.0 0.0 1.0
neighbor 15.0 bin
neigh_modify every 10 delay 0
comm_style tiled
timestep {timestep}

# ===============================
# 1. MINIMIZATION
# ===============================
minimize 1.0e-4 1.0e-6 1000 10000
reset_timestep 0

# ===============================
# 2. OUTPUTS
# ===============================
dump d1 all custom {dump_stride} outputs/${{tag}}.lammpstrj id mol type q xu yu zu
dump_modify d1 sort id

thermo_style custom step pe ke etotal temp press density
thermo {thermo_stride}
thermo_modify flush yes

variable vpe equal pe
variable vke equal ke
variable vT  equal temp
variable vE  equal etotal
variable vRg equal gyration(all)

fix favg all ave/time {thermo_stride} 10 {thermo_stride*10} v_vpe v_vke v_vT v_vE v_vRg &
    file outputs/thermo_${{tag}}_averaged.dat

# ===============================
# 3. EQUILIBRATION ({ensemble})
# ===============================
velocity all create ${{T_target}} {random.randint(10000,99999)} rot yes dist gaussian
"""
    
    if ensemble == 'NVT':
        content += f"""
# Langevin thermostat for equilibration
fix fxlang all langevin ${{T_target}} ${{T_target}} 1000.0 {random.randint(10000,99999)}
fix fxnve  all nve
fix recenter all momentum 1000 linear 1 1 1

# Equilibration run
run {n_eq}

# ===============================
# 4. PRODUCTION
# ===============================
# Reset averages for production
unfix favg
fix favg all ave/time {thermo_stride} 10 {thermo_stride*10} v_vpe v_vke v_vT v_vE v_vRg &
    file outputs/thermo_${{tag}}_production.dat

restart 10000000 outputs/backup_${{tag}}.*.restart
run {n_run}
"""
    elif ensemble == 'NVE':
        content += f"""
# NVE integration (microcanonical)
fix fxnve all nve
fix recenter all momentum 1000 linear 1 1 1

restart 10000000 outputs/backup_${{tag}}.*.restart
run {n_run}
"""
    
    content += f"""
write_data outputs/${{tag}}_final.data
"""
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"  Created {filename}")
    return filename


def generate_slurm_script(filename, job_name, lammps_input, ntasks=4,
                          walltime="04:00:00", mem_per_cpu="2G"):
    """
    Generate an optimized SLURM script.
    
    Key optimization: For ~1000-atom coarse-grained systems,
    4 MPI ranks is near-optimal. 32 ranks causes massive
    communication overhead (explains your 34% efficiency).
    """
    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={walltime}
#SBATCH --mail-type=all
#SBATCH --mail-user=nn9775@princeton.edu
#SBATCH --constraint=rh9
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

mkdir -p outputs logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Tasks: $SLURM_NTASKS"

srun $HOME/.local/bin/lmp_amd -in {lammps_input}

echo "Job finished: $(date)"
"""
    with open(filename, 'w') as f:
        f.write(content)
    os.chmod(filename, 0o755)
    print(f"  Created {filename}")


def generate_batch_launcher(filename, slurm_scripts):
    """Create a master script that submits all jobs."""
    content = "#!/bin/bash\n"
    content += "# Master launcher - submits all simulation jobs\n"
    content += "# Run: bash launch_all.sh\n\n"
    content += "echo '=== ASO-Hairpin Simulation Campaign ==='\n"
    content += f"echo 'Submitting {len(slurm_scripts)} jobs...'\n\n"
    
    for script in slurm_scripts:
        content += f"JOB=$(sbatch {script} | awk '{{print $4}}')\n"
        content += f"echo \"Submitted {script} -> Job $JOB\"\n\n"
    
    content += "echo 'All jobs submitted!'\n"
    content += "echo 'Monitor with: squeue -u $USER'\n"
    
    with open(filename, 'w') as f:
        f.write(content)
    os.chmod(filename, 0o755)
    print(f"\nCreated master launcher: {filename}")


# =============================================================================
# MAIN GENERATION
# =============================================================================
def main():
    random.seed(42)  # Reproducibility
    
    config_file = '/mnt/user-data/uploads/config_1YMO_exact__1_.dat'
    template_atoms, template_bonds, template_angles, template_masses = read_config(config_file)
    
    output_dir = '/home/claude/aso_project'
    os.makedirs(f"{output_dir}/configs", exist_ok=True)
    os.makedirs(f"{output_dir}/inputs", exist_ok=True)
    os.makedirs(f"{output_dir}/slurm", exist_ok=True)
    
    all_slurm_scripts = []
    
    # =====================================================================
    # PART 1: CONCENTRATION SWEEP (at 300K, unmodified ASO)
    # =====================================================================
    print("\n" + "="*60)
    print("PART 1: CONCENTRATION SWEEP")
    print("="*60)
    
    conc_sweep = {
        25:  80.0,   # 160 Å box (dilute)
        50:  80.0,   # 160 Å box
        100: 80.0,   # 160 Å box (your original)
        200: 110.0,  # 220 Å box (need bigger box for 200)
    }
    
    aso_unmod = create_aso_sequence('unmodified')
    hp_unmod = create_hairpin_sequence('unmodified')
    
    for n_aso, box_half in conc_sweep.items():
        print(f"\n--- {n_aso} ASOs, box ±{box_half} Å ---")
        
        dat_file = f"configs/{n_aso}ASO_unmod.dat"
        dat_path = os.path.join(output_dir, dat_file)
        write_lammps_data(
            dat_path,
            f"1 Docked ASO + {n_aso-1} Free ASOs + 1 Hairpin (unmodified)",
            aso_unmod, hp_unmod,
            template_atoms, template_bonds, template_angles,
            n_free_aso=n_aso - 1,
            box_half=box_half
        )
        
        in_file = f"inputs/NVT_{n_aso}ASO_300K.in"
        in_path = os.path.join(output_dir, in_file)
        generate_lammps_input(in_path, dat_file, 300.0, run_ns=1000, ensemble='NVT')
        
        sh_file = f"slurm/run_{n_aso}ASO_300K.sh"
        sh_path = os.path.join(output_dir, sh_file)
        # Scale tasks slightly with system size
        ntasks = 4 if n_aso <= 100 else 8
        generate_slurm_script(sh_path, f"ASO{n_aso}_300K", in_file, ntasks=ntasks)
        all_slurm_scripts.append(sh_file)
    
    # =====================================================================
    # PART 2: TEMPERATURE SWEEP (100 ASOs, unmodified)
    # =====================================================================
    print("\n" + "="*60)
    print("PART 2: TEMPERATURE SWEEP (250K - 350K)")
    print("="*60)
    
    temperatures = [250, 275, 300, 325, 350]
    
    for T in temperatures:
        print(f"\n--- T = {T}K ---")
        
        # Use the 100ASO config (same structure, different thermostat)
        dat_file = "configs/100ASO_unmod.dat"
        
        in_file = f"inputs/NVT_100ASO_{T}K.in"
        in_path = os.path.join(output_dir, in_file)
        generate_lammps_input(in_path, dat_file, float(T), run_ns=1000, ensemble='NVT')
        
        sh_file = f"slurm/run_100ASO_{T}K.sh"
        sh_path = os.path.join(output_dir, sh_file)
        generate_slurm_script(sh_path, f"ASO100_{T}K", in_file, ntasks=4)
        all_slurm_scripts.append(sh_file)
    
    # =====================================================================
    # PART 3: DIFFERENT ASO DESIGNS (100 ASOs, 300K)
    # =====================================================================
    print("\n" + "="*60)
    print("PART 3: DIFFERENT ASO DESIGNS")
    print("="*60)
    
    aso_variants = [
        ('truncated',      'unmodified',  '7mer truncated ASO'),
        ('extended_12mer', 'unmodified',  '12mer extended ASO'),
        ('extended_14mer', 'unmodified',  '14mer extended ASO'),
        ('mismatch_G6A',   'unmodified',  'G6A mismatch ASO'),
        ('mismatch_U5C',   'unmodified',  'U5C mismatch ASO'),
        ('all_purine',     'unmodified',  'Purine-rich ASO'),
        ('scrambled',      'unmodified',  'Scrambled sequence ASO'),
        ('unmodified',     'AAtoCC',      'Unmod ASO + AAtoCC hairpin'),
        ('unmodified',     'loop_GG_to_AA', 'Unmod ASO + G22A hairpin'),
    ]
    
    for aso_var, hp_var, description in aso_variants:
        print(f"\n--- {description} ---")
        
        aso_seq = create_aso_sequence(aso_var)
        hp_seq = create_hairpin_sequence(hp_var)
        
        safe_name = f"{aso_var}_{hp_var}".replace(' ', '_')
        dat_file = f"configs/100ASO_{safe_name}.dat"
        dat_path = os.path.join(output_dir, dat_file)
        
        write_lammps_data(
            dat_path,
            f"{description} - 1 Docked + 99 Free + 1 Hairpin",
            aso_seq, hp_seq,
            template_atoms, template_bonds, template_angles,
            n_free_aso=99,
            box_half=80.0
        )
        
        in_file = f"inputs/NVT_{safe_name}_300K.in"
        in_path = os.path.join(output_dir, in_file)
        generate_lammps_input(in_path, dat_file, 300.0, run_ns=1000, ensemble='NVT')
        
        sh_file = f"slurm/run_{safe_name}_300K.sh"
        sh_path = os.path.join(output_dir, sh_file)
        generate_slurm_script(sh_path, f"{safe_name}", in_file, ntasks=4)
        all_slurm_scripts.append(sh_file)
    
    # =====================================================================
    # MASTER LAUNCHER
    # =====================================================================
    generate_batch_launcher(os.path.join(output_dir, "launch_all.sh"), all_slurm_scripts)
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "="*60)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nTotal simulations: {len(all_slurm_scripts)}")
    print(f"\nConcentration sweep: {list(conc_sweep.keys())} ASOs at 300K")
    print(f"Temperature sweep:   {temperatures}K with 100 ASOs")
    print(f"ASO variants:        {len(aso_variants)} designs at 300K")
    print(f"\nAll files in: {output_dir}/")
    print(f"  configs/  - LAMMPS data files")
    print(f"  inputs/   - LAMMPS input scripts")
    print(f"  slurm/    - SLURM submission scripts")
    print(f"  launch_all.sh - submit everything at once")


if __name__ == '__main__':
    main()
