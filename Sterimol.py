import numpy as np
from sklearn.decomposition import PCA

# Van der Waals radii (Å) from Bondi (1964)
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, 
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98
}

def read_xyz(filename):
    """Reads an XYZ file and returns atomic symbols and coordinates."""
    with open(filename, "r") as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0].strip())  # First line: Number of atoms
    atoms = []
    coordinates = []
    
    for line in lines[2:num_atoms+2]:  # Skip first two lines (header)
        parts = line.split()
        atoms.append(parts[0])  # Atom symbol
        coordinates.append([float(x) for x in parts[1:]])  # X, Y, Z coordinates
    
    return atoms, np.array(coordinates)

def compute_principal_axis_pca(coords):
    """
    Use PCA to calculate the principal axis (L) by aligning the molecule along its principal component.
    """
    pca = PCA(n_components=1)
    pca.fit(coords)
    principal_axis = pca.components_[0]  # Direction of the largest variance
    return principal_axis

def compute_sterimol_with_nh2_refined(atoms, coords):
    """
    Compute Sterimol Parameters (L, B1, B5) using refined treatment for NH2 group in aniline.
    """
    # Step 1: Use PCA to find the principal axis (L)
    principal_axis = compute_principal_axis_pca(coords)
    L_vector = principal_axis
    L = np.linalg.norm(L_vector)  # Length along the principal axis

    # Step 2: Project all atomic positions onto the principal axis
    projections = np.dot(coords - coords.mean(axis=0), principal_axis)[:, np.newaxis] * principal_axis
    perpendicular_coords = coords - projections

    # Step 3: Compute distances from the principal axis (widths)
    distances = np.linalg.norm(perpendicular_coords, axis=1)

    # Adjust distances using Van der Waals radii
    radii = np.array([VDW_RADII.get(atom, 1.7) for atom in atoms])
    distances += radii  # Adding atomic radii to distances

    # Step 4: Compute B1 (10th percentile) and B5 (90th percentile) for widths
    B1, B5 = np.percentile(distances, [10, 90])

    return L, B1, B5

def calculate_sterimol(xyz_file):
    """Reads an XYZ file and calculates Sterimol descriptors."""
    atoms, coords = read_xyz(xyz_file)
    L, B1, B5 = compute_sterimol_with_nh2_refined(atoms, coords)

    print(f"Sterimol Descriptors for {xyz_file}:")
    print(f"L = {L:.3f}, B1 = {B1:.3f}, B5 = {B5:.3f}")

    return L, B1, B5

# Example usage:
xyz_filename = "data/xyz/Aniline.xyz"  # Replace with your file
calculate_sterimol(xyz_filename)
