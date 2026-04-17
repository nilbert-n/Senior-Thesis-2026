from pathlib import Path
import re


def read_first_timestep(path: Path) -> int:
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"No timestep found in {path}")
            if line.startswith("ITEM: TIMESTEP"):
                return int(f.readline().strip())


def iter_dump_frames(path: Path):
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            ts_line = f.readline()
            if not ts_line:
                break
            ts = int(ts_line.strip())

            block = [line, ts_line]

            num_atoms_header = f.readline()
            if not num_atoms_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Unexpected format in {path}: expected NUMBER OF ATOMS")
            block.append(num_atoms_header)

            natoms_line = f.readline()
            natoms = int(natoms_line.strip())
            block.append(natoms_line)

            box_header = f.readline()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected format in {path}: expected BOX BOUNDS")
            block.append(box_header)

            for _ in range(3):
                block.append(f.readline())

            atoms_header = f.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected format in {path}: expected ATOMS")
            block.append(atoms_header)

            for _ in range(natoms):
                block.append(f.readline())

            yield ts, block


def get_base_file_for_restart(restart_file: Path) -> Path:
    name = restart_file.name

    base_name = re.sub(r"_restart[^/\\]*\.lammpstrj$", ".lammpstrj", name)
    base_path = restart_file.with_name(base_name)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Could not find matching base trajectory for {restart_file.name}. "
            f"Expected: {base_path.name}"
        )

    return base_path


restart_files = sorted(Path(".").glob("*_restart*.lammpstrj"))

if not restart_files:
    raise RuntimeError("No *_restart*.lammpstrj files found.")

restart_files = sorted(restart_files, key=lambda f: read_first_timestep(f))

print("Restart files found:")
for r in restart_files:
    print(f"    {r.name} -> {read_first_timestep(r)}")

for rest in restart_files:
    orig = get_base_file_for_restart(rest)
    restart_start = read_first_timestep(rest)

    out = rest.with_name(rest.name.replace("_restart.lammpstrj", "_spliced.lammpstrj"))

    print(f"\nProcessing restart file: {rest.name}")
    print(f"Matching base file:      {orig.name}")
    print(f"Restart starts at:       {restart_start}")
    print(f"Output file:             {out.name}")

    with out.open("w") as fout:
        # keep original only before restart point
        for ts, block in iter_dump_frames(orig):
            if ts < restart_start:
                fout.writelines(block)

        # append restart from restart point onward
        for ts, block in iter_dump_frames(rest):
            if ts >= restart_start:
                fout.writelines(block)

    print(f"Done: wrote {out.name}")