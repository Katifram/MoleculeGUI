import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

from skimage import measure

from scipy.interpolate import RegularGridInterpolator

from rdkit import Chem
from rdkit.Chem import AllChem

def create_xyz_string(name, smiles, charge):  # Add 'self' as the first parameter
        """Generates a 3D structure and returns it in XYZ format as a string without atom count or molecule name."""

        mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule object
        mol = Chem.AddHs(mol)  # Add hydrogen atoms

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

        # Prepare XYZ string format without the number of atoms or molecule name
        conf = mol.GetConformer()  # Get the conformation of the molecule
        xyz_string = ""
        for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                # Format each line with increased spacing and align values for a cleaner look
                xyz_string += f"{atom.GetSymbol():<2}    {pos.x: .10f}    {pos.y: .10f}    {pos.z: .10f}\n"

        return xyz_string.strip()  # Remove any trailing newline


def parse_cube(filename):
    #from: https://github.com/psi4/psi4numpy/blob/6ed03e715689ec82bf96fbb23c1855fbe7835b90/Tutorials/14_Visualization/vizualize.ipynb
    """ Parses a cube file, returning a dict of the information contained.
        The cubefile itself is stored in a numpy array. """
    with open(filename) as fp:
        results = {}

        # skip over the title
        fp.readline()
        fp.readline()

        origin = fp.readline().split()
        natoms = int(origin[0])
        results['minx'] = minx = float(origin[1])
        results['miny'] = miny = float(origin[2])
        results['minz'] = minz = float(origin[3])

        infox = fp.readline().split()
        numx = int(infox[0])
        incx = float(infox[1])
        results['incx'] = incx
        results['numx'] = numx
        results['maxx'] = minx + incx * numx

        infoy = fp.readline().split()
        numy = int(infoy[0])
        incy = float(infoy[2])
        results['incy'] = incy
        results['numy'] = numy
        results['maxy'] = miny + incy * numy

        infoz = fp.readline().split()
        numz = int(infoz[0])
        incz = float(infoz[3])
        results['incz'] = incz
        results['numz'] = numz
        results['maxz'] = minz + incz * numz

        atnums = []
        coords = []
        for atom in range(natoms):
            coordinfo = fp.readline().split()
            atnums.append(int(coordinfo[0]))
            coords.append(list(map(float, coordinfo[2:])))
        results['atom_numbers'] = np.array(atnums)
        results['atom_coords'] = np.array(coords)

        data = np.array([ float(entry) for line in fp for entry in line.split() ])
        if len(data) != numx*numy*numz:
            raise Exception("Amount of parsed data is inconsistent with header in Cube file!")
        results['data'] = data.reshape((numx,numy,numz))

        return results
    

def draw_isosurface(parsed_cube, iso_value):

    vert, faces, norm, values= measure.marching_cubes(parsed_cube['data'], iso_value, spacing=(parsed_cube['incx'],parsed_cube['incy'],parsed_cube['incz']))
        
    # Set up a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D mesh from vertices and faces
    mesh = Poly3DCollection(vert[faces], alpha=0.3, edgecolor='k', linewidth=0.1)
    mesh.set_facecolor([0.5, 0.5, 1, 0.9])  # Set color and transparency

    ax.add_collection3d(mesh)

    # Plot atoms with correct positions relative to the origin
    atom_coords = parsed_cube['atom_coords'] - np.array([parsed_cube['minx'], parsed_cube['miny'], parsed_cube['minz']])
    atom_numbers = parsed_cube['atom_numbers']
    colors = cm.get_cmap("viridis", len(set(atom_numbers)))  # Color map for atom types

    

    # Plot atoms and labels
    for i, (atom_num, coord) in enumerate(zip(atom_numbers, atom_coords)):
        ax.scatter(*coord, color=colors(i), s=100, label=f'Atom {atom_num}')
        ax.text(*coord, f'{atom_num}', color='black', fontsize=12, ha='center')  # Label with atomic number

    # Set plot limits based on the shape of the cube data
    ax.set_xlim(0, parsed_cube['data'].shape[0] * parsed_cube['incx'])
    ax.set_ylim(0, parsed_cube['data'].shape[1] * parsed_cube['incy'])
    ax.set_zlim(0, parsed_cube['data'].shape[2] * parsed_cube['incz'])

    

    # Set axis labels
    ax.set_xlabel("X / bohr")
    ax.set_ylabel("Y / bohr")
    ax.set_zlabel("Z / bohr")

    # Show the plot
    plt.show()


def ray_cube_distance(cube_min, cube_max, point, direction):
    """
    Calculate the distance from a point to the edge of a cube in a given direction.
    
    Args:
    - cube_min: ndarray of shape (3,), minimum coordinates of the cube.
    - cube_max: ndarray of shape (3,), maximum coordinates of the cube.
    - point: ndarray of shape (3,), the starting point.
    - direction: ndarray of shape (3,), the direction vector.
    
    Returns:
    - Distance to the cube edge (float).
    """
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Calculate t for each slab (x, y, z planes)
    t_min = (cube_min - point) / direction
    t_max = (cube_max - point) / direction
    
    # Sort t_min and t_max to ensure correct ordering
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    
    # Find the largest t1 and smallest t2
    t_near = np.max(t1)
    t_far = np.min(t2)
    
    # If t_near > t_far or t_far < 0, the ray does not intersect the cube
    if t_near > t_far or t_far < 0:
        return None  # No intersection
    
    # Return the nearest positive t
    return t_near if t_near > 0 else t_far


def values_along_direction(parsed_cube, start, direction, num_points):
    """
    Extracts interpolated values along a line from `start` in `direction` in cube data and computes distances.

    Parameters:
    - cube: dict, parsed cube data with keys 'data', 'minx', 'miny', 'minz', 'incx', 'incy', 'incz'
    - start: list or array of [x, y, z] coordinates for the starting point in Angstroms.
    - direction: list or array of [dx, dy, dz], the direction vector.
    - num_points: Number of points to sample along the line.
    - step_size: Distance between points along the direction vector in Angstroms.

    Returns:
    - line_values: numpy array of interpolated values along the direction line.
    - line_points: numpy array of sampled points along the direction line.
    - distances: numpy array of distances from the starting point.
    """

    # Normalize the direction vector
    direction = np.array(direction) / np.linalg.norm(direction)

    # find distance between points from parsed_cube by
    min_array = [parsed_cube['minx'], parsed_cube['miny'], parsed_cube['minz']]
    max_array = [parsed_cube['maxx'], parsed_cube['maxy'], parsed_cube['maxz']]
    step_size = ray_cube_distance(min_array, max_array, start, direction) / num_points

    # Generate points along the line at each step_size interval
    line_points = np.array([start + i * step_size * direction for i in range(num_points)])

    # Calculate distances from the start point
    distances = np.arange(num_points) * step_size  # Distance = index * step_size

    # Define the grid axes based on cube grid
    x = np.linspace(start=parsed_cube['minx'], stop=parsed_cube['maxx'], num=parsed_cube['numx'])
    y = np.linspace(start=parsed_cube['miny'], stop=parsed_cube['maxy'], num=parsed_cube['numy'])
    z = np.linspace(start=parsed_cube['miny'], stop=parsed_cube['maxy'], num=parsed_cube['numy'])

    # Create an interpolator function for the 3D grid
    interpolator = RegularGridInterpolator((x, y, z), parsed_cube['data'])

    # Interpolate data values along the line points
    line_values = interpolator(line_points)

    return line_values, line_points, distances


# Function to calculate charge from SMILES
def calculate_charge(smiles):
    try:
        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # Return None for invalid SMILES

        # Calculate formal charge
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        return charge
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

import os

def generate_gaussian_input(xyz_file, output_dir, functional="b3lyp", basis_set="def2TZVP", jobtype="calcall", solvent="water"):
    """
    Generate a Gaussian input file based on an XYZ file.
    
    Parameters:
    xyz_file (str): Path to the XYZ file.
    output_dir (str): Directory where the Gaussian input files will be saved.
    functional (str): DFT functional to use (default is M06-2X).
    basis_set (str): Basis set to use (default is Def2TZVP).
    solvent (str): Solvent for implicit solvent model (default is water).
    """
    # Extract the molecule name from the file name
    molecule_name = os.path.splitext(os.path.basename(xyz_file))[0]
    
    # Read the coordinates from the XYZ file
    with open(xyz_file, 'r') as file:
        lines = file.readlines()
    
    coordinates = lines
    
    # Construct the Gaussian input file content
    gaussian_input = f"""%nproc=7
#P {functional} {basis_set} Opt={jobtype} scf=(QC,Noincfock,NoVarAcc)

{molecule_name}: {functional}/{basis_set} with scf=(QC,Noincfock,NoVarAcc)

0 1
"""
    gaussian_input += "".join(coordinates)
    gaussian_input += f"""

"""  # Add an additional newline at the end of the file
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the Gaussian input file
    output_file = os.path.join(output_dir, f"{molecule_name}.in")
    with open(output_file, 'w') as file:
        file.write(gaussian_input)
    
    print(f"Gaussian input file created: {output_file}")


def generate_qchem_input(xyz_file, output_dir, functional="M06-2X", basis_set="def2-TZVP", jobtype="OPT", solvent="water"):
    """
    Generate a Gaussian input file based on an XYZ file.
    
    Parameters:
    xyz_file (str): Path to the XYZ file.
    output_dir (str): Directory where the Gaussian input files will be saved.
    functional (str): DFT functional to use (default is M06-2X).
    basis_set (str): Basis set to use (default is Def2TZVP).
    solvent (str): Solvent for implicit solvent model (default is water).
    """
    # Extract the molecule name from the file name
    molecule_name = os.path.splitext(os.path.basename(xyz_file))[0]
    
    # Read the coordinates from the XYZ file
    with open(xyz_file, 'r') as file:
        lines = file.readlines()
    
    coordinates = lines
    
    # Construct the Qchem input file content
    qchem_input = f"""$comment
{molecule_name}: {functional}/{basis_set} 
$end

$rem
JOBTYPE         	{jobtype}
METHOD          	{functional}
BASIS           	{basis_set}
S2THRESH		15
THRESH			14
SCF_ALGORITHM 		DIIS_GDM
THRESH_DIIS_SWITCH 	5 
MAX_DIIS_CYCLES 	100
MAX_SCF_CYCLES 	200
$end

$molecule
0 1
"""
    qchem_input += "".join(coordinates)
    qchem_input += f"""$end"""
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the Qchem input file
    output_file = os.path.join(output_dir, f"{molecule_name}.qcin")
    with open(output_file, 'w') as file:
        file.write(qchem_input)
    
    print(f"QCHEM input file created: {output_file}")