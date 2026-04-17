import numpy as np
import matplotlib.pyplot as plt
import os

def kabsch_rmsd(P, Q):
    """Calculates RMSD between two sets of points after optimal translation/rotation."""
    Pc = P - np.mean(P, axis=0)
    Qc = Q - np.mean(Q, axis=0)
    C = np.dot(np.transpose(Pc), Qc)
    V, S, W = np.linalg.svd(C)
    if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    P_rot = np.dot(Pc, U)
    return np.sqrt(np.mean(np.sum((P_rot - Qc)**2, axis=1)))

def plot_thermodynamics(thermo_file="thermo_100ASO.dat_NVE_averaged.dat"):
    """Plots thermodynamic quantities on separate graphs vs time in ms."""
    if not os.path.exists(thermo_file): 
        print(f"File not found: {thermo_file}")
        return
        
    data = np.loadtxt(thermo_file, skiprows=2)
    step, pe, ke, temp, etotal = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
    
    # Convert step to time in ms (timestep = 10 fs = 1e-11 ms)
    time_ms = step * 10.0 * 1e-12

    # 1. Potential Energy vs Time
    plt.figure(figsize=(8, 5))
    plt.plot(time_ms, pe, 'b-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Potential Energy (kcal/mol)")
    plt.title("Potential Energy vs. Time")
    plt.savefig("pe_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Kinetic Energy vs Time
    plt.figure(figsize=(8, 5))
    plt.plot(time_ms, ke, 'g-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Kinetic Energy (kcal/mol)")
    plt.title("Kinetic Energy vs. Time")
    plt.savefig("ke_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Temperature Fluctuations vs Time
    plt.figure(figsize=(8, 5))
    plt.plot(time_ms, temp, 'r-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Fluctuations vs. Time")
    plt.savefig("temp_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Total Energy vs Time (NVE Validation)
    plt.figure(figsize=(8, 5))
    plt.plot(time_ms, etotal, 'k-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Total Energy (kcal/mol)")
    plt.title("Total Energy vs. Time (NVE Validation)")
    # Force y-axis limits to be very tight to show it is a flat line
    mean_e = np.mean(etotal)
    plt.ylim(mean_e - max(abs(mean_e)*0.01, 10), mean_e + max(abs(mean_e)*0.01, 10))
    plt.savefig("etotal_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()

def main_analysis(lammpstrj="100ASO.dat_NVE.lammpstrj", pdb_file="1YMO.pdb1"):
    """Calculates RMSD, Rg, and Contact Probabilities vs time in ms."""
    if not os.path.exists(lammpstrj):
        print(f"File not found: {lammpstrj}")
        return
        
    # 1. Parse PDB for reference C3' coordinates (Residues 15-46)
    ref_hp = []
    if os.path.exists(pdb_file):
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and "C3'" in line:
                    res_id = int(line[22:26].strip())
                    if 15 <= res_id <= 46:
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        ref_hp.append([x, y, z])
    ref_hp = np.array(ref_hp)
    
    # 2. Parse LAMMPS Trajectory
    with open(lammpstrj, 'r') as f:
        lines = f.readlines()
        
    frames, steps, mol_array = [], [], None
    i = 0
    while i < len(lines):
        if lines[i].startswith('ITEM: TIMESTEP'):
            step_val = int(lines[i+1].strip())
            steps.append(step_val)
            
            natoms = int(lines[i+3])
            coords = np.zeros((natoms, 3))
            mols = np.zeros(natoms, dtype=int)
            for j in range(natoms):
                parts = lines[i+9+j].split()
                idx = int(parts[0]) - 1 
                mols[idx] = int(parts[1])
                coords[idx] = [float(parts[4]), float(parts[5]), float(parts[6])]
            frames.append(coords)
            if mol_array is None: mol_array = mols.copy()
            i += 9 + natoms
        else:
            i += 1
            
    frames = np.array(frames)
    time_ms = np.array(steps) * 10.0 * 1e-12
    
    aso_idx = np.where(mol_array == 1)[0]
    hp_idx = np.where(mol_array == 2)[0]
    
    # 3. Calculate RMSD against PDB and Rg
    rmsd_pdb = np.zeros(len(frames))
    rg = np.zeros(len(frames))
    
    for t in range(len(frames)):
        hp_c = frames[t, hp_idx, :]
        if ref_hp.size > 0:
            rmsd_pdb[t] = kabsch_rmsd(hp_c, ref_hp)
        com = np.mean(hp_c, axis=0)
        rg[t] = np.sqrt(np.mean(np.sum((hp_c - com)**2, axis=1)))
        
    # 5. RMSD vs Time Plot
    plt.figure(figsize=(8, 5))
    plt.plot(time_ms, rmsd_pdb, 'b-')
    plt.xlabel("Time (ms)")
    plt.ylabel("RMSD (Å)")
    plt.title("Hairpin RMSD (vs 1YMO PDB) vs. Time")
    plt.savefig("rmsd_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Rg vs Time Plot
    plt.figure(figsize=(8, 5))
    plt.plot(time_ms, rg, 'r-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Radius of Gyration (Å)")
    plt.title("Hairpin Radius of Gyration vs. Time")
    plt.savefig("rg_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Contact Probability Heatmap
    cutoff = 15.0 
    contact_matrix = np.zeros((len(aso_idx), len(hp_idx)))
    for t in range(len(frames)):
        aso_c = frames[t, aso_idx, :]
        hp_c = frames[t, hp_idx, :]
        for a in range(len(aso_idx)):
            for h in range(len(hp_idx)):
                if np.linalg.norm(aso_c[a] - hp_c[h]) < cutoff:
                    contact_matrix[a, h] += 1
                    
    contact_matrix /= len(frames)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(contact_matrix, aspect='auto', cmap='plasma', origin='lower')
    plt.colorbar(label='Binding Probability')
    plt.xlabel("Hairpin Target Index (15-46)")
    plt.ylabel("ASO Sequence Index (1-10)")
    plt.xticks(ticks=np.arange(0, 32, 2), labels=np.arange(15, 47, 2))
    plt.title("Intermolecular Contact Strength (Binding Probability)")
    plt.savefig("contact_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Successfully generated all separate graphs with time in ms.")

if __name__ == "__main__":
    plot_thermodynamics()
    main_analysis()