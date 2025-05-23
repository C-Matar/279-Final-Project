import h5py
import numpy as np
import csv
from collections import Counter

def get_formula(species_bytes):
    elements = [s.decode("utf-8") for s in species_bytes]
    counts = Counter(elements)

    atomic_order = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    sorted_elems = sorted(counts.items(), key=lambda x: atomic_order.get(x[0], 999))

    formula = "".join(f"{elem}{count if count > 1 else ''}" for elem, count in sorted_elems)
    return formula

# Atomic number lookup for common elements in ANI-1
Z_TABLE = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
}

filename = "ANI-1_release/ani_gdb_s05.h5"  # replace with active ANI file
output_file = "ani_s05.csv"

with h5py.File(filename, "r") as f, open(output_file, "w", newline="") as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(["molecule_id", "z1", "z2", "distance", "energy", "formula"])

    group = f["gdb11_s05"]
    mol_count = 0

    for mol_key in group.keys():
        try:
            species = group[f"{mol_key}/species"][:]  # example format: [b'C', b'H', ...]
            formula = get_formula(species)
            Z = [Z_TABLE[s.decode("utf-8")] for s in species]
            R = group[f"{mol_key}/coordinates"][:] # shape: (n_confs, n_atoms, 3)
            E = group[f"{mol_key}/energies"][:] # shape: (n_confs,)
        except Exception as e:
            print(f"Skipping {mol_key} due to error: {e}")
            continue

        for conf_idx in range(len(R)):
            coords = R[conf_idx]
            energy = E[conf_idx]
            mol_id = f"mol_{mol_count:05d}"
            mol_count += 1

            for i in range(len(Z)):
                for j in range(i + 1, len(Z)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    writer.writerow([mol_id, Z[i], Z[j], dist, energy, formula])

print(f"Finished writing {mol_count} molecules to {output_file}")
