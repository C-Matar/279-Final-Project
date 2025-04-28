import random
import math
import argparse

ATOM_TYPES = ["H", "C", "N", "O"]
DEFAULT_NUM_MOLECULES = 2000
DEFAULT_MIN_ATOMS = 5
DEFAULT_MAX_ATOMS = 7
DEFAULT_BOX_SIZE = 5
OUTPUT_FILENAME = "molecules_example.txt"

def calculate_distance(atom1_coords, atom2_coords):
    dx = atom1_coords[0] - atom2_coords[0]
    dy = atom1_coords[1] - atom2_coords[1]
    dz = atom1_coords[2] - atom2_coords[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def calculate_dummy_energy(atoms):
    """
    Calculates a dummy total energy based on pair distances and types.
    """
    total_energy = 0.0
    lj_epsilon = 2  # Lennard-Jones like dummy parameters
    lj_sigma = 2.5

    # Generate dummy pair interactions
    num_atoms = len(atoms)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom1_type, atom1_coords = atoms[i]
            atom2_type, atom2_coords = atoms[j]

            dist = calculate_distance(atom1_coords, atom2_coords)
            if dist < 0.1: dist = 0.1 # Avoid division by zero

            # Lennard-Jones term, this is dummy
            sigma_over_r = lj_sigma / dist
            sigma_over_r6 = sigma_over_r**6
            pair_lj = 10.0 * lj_epsilon * (sigma_over_r6**2 - sigma_over_r6) # This lj potential does not consider atom types

            total_energy += pair_lj

    return total_energy

def generate_data(num_molecules, min_atoms, max_atoms, box_size, filename):
    """
    Generates molecule data and saves it to a file.
    """
    print(f"Generating {num_molecules} molecules...")
    with open(filename, 'w') as f:
        for i in range(num_molecules):
            num_atoms_in_mol = random.randint(min_atoms, max_atoms)
            atoms_in_mol = []

            for _ in range(num_atoms_in_mol):
                atom_type = random.choice(ATOM_TYPES)
                coords = [random.uniform(-box_size, box_size) for _ in range(3)]
                atoms_in_mol.append((atom_type, coords))

            total_energy = calculate_dummy_energy(atoms_in_mol)

            if total_energy > 0: # NOTE: Skip if energy is positive
                continue

            f.write(f"{num_atoms_in_mol} {total_energy:.6f}\n")
            for atom_type, coords in atoms_in_mol:
                f.write(f"{atom_type} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")

            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{num_molecules} molecules...")

    print(f"Data generation complete. Saved to '{filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_molecules", type=int, default=DEFAULT_NUM_MOLECULES, help="Number of molecules to generate")
    parser.add_argument("--min_atoms", type=int, default=DEFAULT_MIN_ATOMS, help="Minimum number of atoms per molecule")
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS, help="Maximum number of atoms per molecule")
    parser.add_argument("--output", type=str, default=OUTPUT_FILENAME, help="Output filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)

    generate_data(args.num_molecules, args.min_atoms, args.max_atoms, DEFAULT_BOX_SIZE, args.output)