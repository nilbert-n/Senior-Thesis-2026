import os

# 1. Update 02_dynamics_analysis.py
file_02 = '02_dynamics_analysis.py'
if os.path.exists(file_02):
    with open(file_02, 'r') as f:
        code = f.read()
    
    # Convert timestep math to nanoseconds (10 fs step = 1e-5 ns conversion)
    code = code.replace('.astype(float) / 1e6', '.astype(float) * 1e-5')
    code = code.replace('d["thermo_ts"] / 1e6', 'd["thermo_ts"] * 1e-5')
    # Update X-axis labels
    code = code.replace('Timestep (×10⁶)', 'Time (ns)')
    
    with open(file_02, 'w') as f:
        f.write(code)
    print(f"Successfully updated time axes to ns in {file_02}")

# 2. Update 03_contact_analysis.py
file_03 = '03_contact_analysis.py'
if os.path.exists(file_03):
    with open(file_03, 'r') as f:
        code = f.read()
    
    # Convert timestep math and labels
    code = code.replace('.astype(float) / 1e6', '.astype(float) * 1e-5')
    code = code.replace('Timestep (×10⁶)', 'Time (ns)')
    
    # Update the rigorous thermodynamic cutoffs
    code = code.replace('CONTACT_CUTOFF = 10.0', 'CONTACT_CUTOFF = 16.56')
    code = code.replace('CLOSE_CUTOFF   =  8.0', 'CLOSE_CUTOFF   = 13.80')
    
    with open(file_03, 'w') as f:
        f.write(code)
    print(f"Successfully updated cutoffs and time axes in {file_03}")

# 3. Update 04_binding_analysis.py
file_04 = '04_binding_analysis.py'
if os.path.exists(file_04):
    with open(file_04, 'r') as f:
        code = f.read()
    
    # Convert timestep math and labels
    code = code.replace('.astype(float) / 1e6', '.astype(float) * 1e-5')
    code = code.replace('Timestep (×10⁶)', 'Time (ns)')
    
    # Update the rigorous thermodynamic cutoffs
    code = code.replace('BIND_CUTOFF  = 12.0', 'BIND_CUTOFF  = 16.56')
    code = code.replace('CLOSE_CUTOFF =  8.0', 'CLOSE_CUTOFF = 13.80')
    
    with open(file_04, 'w') as f:
        f.write(code)
    print(f"Successfully updated cutoffs and time axes in {file_04}")

print("\nAll scripts have been rewritten! You can now run them to generate the updated nanosecond plots.")