import pandas as pd
from rdkit import Chem

def load_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, sep=';', engine='python', quotechar='"')
    
    # Clean the 'Name' column
    df['Name'] = df['Name'].str.split(r' \(', n=1).str[0]
    
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

    # Function to assign sp2 or sp hybridization for nitrogen
    def assign_hybridization(smiles):
        try:
            # Convert SMILES to RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None  # Return None for invalid SMILES

            # Initialize variables for charge and hybridization
            sp2_nitrogens = 0
            sp3_nitrogens = 0
            sp_nitrogens = 0
            negative_charge_nitrogens = []

            # Check the hybridization and charge of nitrogen atoms
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N':  # If atom is nitrogen
                    charge = atom.GetFormalCharge()
                    hybridization = atom.GetHybridization()

                    if charge == -1:
                        negative_charge_nitrogens.append(hybridization)  # Prioritize this nitrogen
                    elif hybridization == Chem.rdchem.HybridizationType.SP:
                        sp_nitrogens += 1
                    elif hybridization == Chem.rdchem.HybridizationType.SP2:
                        sp2_nitrogens += 1
                    elif hybridization == Chem.rdchem.HybridizationType.SP3:
                        sp3_nitrogens += 1

            # Prioritize nitrogen with negative charge first
            if negative_charge_nitrogens:
                # If there are any nitrogen atoms with a negative charge, take the hybridization of those
                if Chem.rdchem.HybridizationType.SP2 in negative_charge_nitrogens:
                    return 2  # Return 2 for sp2 hybridized nitrogen with a negative charge
                elif Chem.rdchem.HybridizationType.SP3 in negative_charge_nitrogens:
                    return 3  # Return 3 for sp3 hybridized nitrogen with a negative charge
                elif Chem.rdchem.HybridizationType.SP in negative_charge_nitrogens:
                    return 1  # Return 1 for sp hybridized nitrogen with a negative charge

            # If no negative charge nitrogen, check the general hybridization
            if sp2_nitrogens > 0:
                return 2  # sp2 present
            elif sp3_nitrogens > 0:
                return 3  # no sp2 but sp3 present
            elif sp_nitrogens > 0:
                return 1  # all nitrogens are sp
            else:
                return None  # no nitrogen
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            return None

    # Update the 'charge' and 'sp' columns
    df['charge'] = df.apply(
        lambda row: calculate_charge(row['Smiles']) if pd.notna(row['Smiles']) else row['charge'],
        axis=1
    )

    df['sp'] = df.apply(
        lambda row: assign_hybridization(row['Smiles']) if pd.notna(row['Smiles']) else row['sp'],
        axis=1
    )

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, sep=';', index=False, quotechar='"')
    
    # Select relevant columns for display or further processing
    columns_of_interest = ['Name', 'Smiles', 'N Params', 'sN Params', 'charge', 'sp']
    extracted_data = df[columns_of_interest]
    
    return extracted_data

# Example usage
file_path = 'data/N_Nucleophiles/Nucleophile.csv'
extracted_data = load_data(file_path)
print(extracted_data)
