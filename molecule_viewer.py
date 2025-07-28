import tkinter as tk
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageTk
import pandas as pd
from utils import generate_gaussian_input, generate_qchem_input, create_xyz_string
import os

class MoleculeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Molecule Viewer")

        self.current_index = 0
        self.extracted_data = pd.DataFrame()  # Initialize empty dataframe
        self.filtered_data = pd.DataFrame()  # To store filtered data based on solvent
        self.solvent_options = []  # To store unique solvents
        self.selected_solvent = tk.StringVar()  # To store the selected solvent
        self.sp_options = []  # To store unique sp values
        self.selected_sp = tk.StringVar()  # To store the selected sp value
        self.N_number = 0
        self.modified_smiles = []  # List of smiles for modifications
        self.modified_names = []  # List of names for modified smiles
        self.modified_charge = []  # List of charges for modified smiles

        # Create UI components
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=20)

        self.name_label = tk.Label(root, font=('Arial', 14))
        self.name_label.pack(pady=10)

        self.n_params_label = tk.Label(root, font=('Arial', 12))
        self.n_params_label.pack(pady=5)

        self.sn_params_label = tk.Label(root, font=('Arial', 12))
        self.sn_params_label.pack(pady=5),

        self.sp_label = tk.Label(root, font=('Arial', 12))
        self.sp_label.pack(pady=5)

        # Solvent dropdown        
        self.solvent_menu = tk.OptionMenu(root, self.selected_solvent, [])
        self.solvent_menu.pack(pady=5)

        # sp dropdown
        self.sp_menu = tk.OptionMenu(root, self.selected_sp, [])
        self.sp_menu.pack(pady=5)
        
        # Initially disable the dropdown menus
        self.solvent_menu.config(state="disabled")
        self.sp_menu.config(state="disabled")

        # Molecule count label
        self.molecule_count_label = tk.Label(root, font=('Arial', 12))
        self.molecule_count_label.pack(pady=5)

        # Checkbox for CH3+ addition
        self.add_ch3_var = tk.BooleanVar()
        self.add_ch3_checkbox = tk.Checkbutton(root, text="Add CH3+ to N", variable=self.add_ch3_var, command=self.modify_molecule)
        self.add_ch3_checkbox.pack(pady=5)
        
        # Checkbox for Protonation
        self.add_h_var = tk.BooleanVar()
        self.add_h_checkbox = tk.Checkbutton(root, text="Add H+ to N", variable=self.add_h_var, command=self.modify_molecule)
        self.add_h_checkbox.pack(pady=5)

        # button for selecting nitrogen for protonation/methylation
        self.swap_nitrogen_button = tk.Button(root, text="Swap Nitrogen", command=self.swap_nitrogen)
        self.swap_nitrogen_button.pack(pady=5)

        # Buttons for switching files
        n_button = tk.Button(root, text="N-Nucleophiles",
                             command=lambda: self.switch_file('data/N_Nucleophiles/Nucleophile.csv'))
        n_button.pack(side='top', padx=20, pady=5)

        c_button = tk.Button(root, text="C-Nucleophiles",
                             command=lambda: self.switch_file('data/C_Nucleophiles/Nucleophile.csv'))
        c_button.pack(side='top', padx=20, pady=5)

        o_button = tk.Button(root, text="O-Nucleophiles",
                             command=lambda: self.switch_file('data/O_Nucleophiles/Nucleophile.csv'))
        o_button.pack(side='top', padx=20, pady=5)

        # New button for "My_Nucleophiles"
        my_n_button = tk.Button(root, text="My Molecules",
                                command=lambda: self.switch_file('data/My_Molecules/Nucleophile.csv'))
        my_n_button.pack(side='top', padx=20, pady=5)

        # Previous and next buttons
        prev_button = tk.Button(root, text="Vorheriges Molekül", command=self.previous_molecule)
        prev_button.pack(side='left', padx=20)

        next_button = tk.Button(root, text="Nächstes Molekül", command=self.next_molecule)
        next_button.pack(side='right', padx=20)

        # "Calculate Molecule" button
        calc_molecule_button = tk.Button(root, text="Calculate Molecule", command=self.create_input_for_molecule)
        calc_molecule_button.pack(side='bottom', padx=20, pady=5)

        # "Calculate Category" button (new button)
        calc_category_button = tk.Button(root, text="Calculate Category", command=self.create_input_for_category)
        calc_category_button.pack(side='bottom', padx=20, pady=5)

        self.show_molecule(self.current_index)

    def load_data(self, file_path):
        df = pd.read_csv(file_path, sep=';', engine='python', quotechar='"')
        df['Name'] = df['Name'].str.split(r' \(', n=1).str[0]
        df = df.dropna(subset=['Smiles'])
        columns_of_interest = ['Name', 'Smiles', 'N Params', 'sN Params', 'charge', 'Solvent', 'sp']
        self.extracted_data = df[columns_of_interest]

        # Extract unique solvents and sp values
        self.solvent_options = sorted(self.extracted_data['Solvent'].dropna().unique())
        self.solvent_options.insert(0, "all")  # Add "all" for solvents

        self.sp_options = sorted(self.extracted_data['sp'].dropna().unique())
        self.sp_options.insert(0, "all")  # Add "all" for sp values

        # Update dropdown menus
        if self.solvent_options:
            self.selected_solvent.set(self.solvent_options[0])  # Default to "all"
            self.update_solvent_dropdown()

        if self.sp_options:
            self.selected_sp.set(self.sp_options[0])  # Default to "all"
            self.update_sp_dropdown()

        # Apply initial filtering
        self.apply_filters()

        # Enable dropdown menus
        self.solvent_menu.config(state="normal")
        self.sp_menu.config(state="normal")

        self.show_molecule(self.current_index)

    def update_solvent_dropdown(self):
        # Update the options in the solvent dropdown menu
        menu = self.solvent_menu['menu']
        menu.delete(0, 'end')
        for solvent in self.solvent_options:
            menu.add_command(label=solvent, command=tk._setit(self.selected_solvent, solvent))

        # Bind the selection event for solvent
        self.selected_solvent.trace('w', self.on_filter_change)

    def update_sp_dropdown(self):
        # Update the options in the sp dropdown menu
        menu = self.sp_menu['menu']
        menu.delete(0, 'end')
        for sp in self.sp_options:
            menu.add_command(label=sp, command=tk._setit(self.selected_sp, sp))

        # Bind the selection event for sp
        self.selected_sp.trace('w', self.on_filter_change)

    def on_filter_change(self, *args):
        # Apply filters whenever solvent or sp changes
        self.apply_filters()
        self.show_molecule(self.current_index)

    def apply_filters(self):
        # Start with the full dataset
        filtered = self.extracted_data

        # Convert 'sp' column to string for comparison if needed
        filtered['sp'] = filtered['sp'].astype(str)

        # Filter by solvent
        selected_solvent = self.selected_solvent.get()
        if selected_solvent != "all":
            filtered = filtered[filtered['Solvent'] == selected_solvent]

        # Filter by sp
        selected_sp = self.selected_sp.get()
        if selected_sp != "all":
            filtered = filtered[filtered['sp'] == selected_sp]

        self.filtered_data = filtered if not filtered.empty else pd.DataFrame()
        self.modified_smiles = self.filtered_data['Smiles'].tolist()
        self.modified_names = self.filtered_data['Name'].tolist()
        self.modified_charge = self.filtered_data['charge'].tolist()

        # Reset the index if no data matches
        if self.filtered_data.empty:
            self.current_index = 0


    def show_molecule(self, index):
        if self.filtered_data.empty:
            self.name_label.config(text="No data loaded")
            self.n_params_label.config(text="")
            self.sn_params_label.config(text="")
            self.img_label.config(image='')  # Clear the image label
            self.sp_label.config(text="sp: Not available")  # Clear sp value
            self.molecule_count_label.config(text="No molecules available")  # Update count
            return

        index = index % len(self.filtered_data)

        smiles = self.modified_smiles[index]
        name = self.modified_names[index]
        n_params = self.filtered_data.iloc[index]['N Params']
        sn_params = self.filtered_data.iloc[index]['sN Params']
        charge = self.modified_charge[index]
        solvent = self.filtered_data.iloc[index]['Solvent']
        sp = self.filtered_data.iloc[index]['sp']  # Get the sp value

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol)
            img.save('current_molecule.png')
            self.display_image()

        self.name_label.config(text=name)
        self.n_params_label.config(text=f"N Params: {n_params}")
        self.sn_params_label.config(text=f"sN Params: {sn_params}")
        self.sp_label.config(text=f"sp: {sp}")  # Display the sp value
        self.molecule_count_label.config(text=f"{index + 1}/{len(self.filtered_data)}")  # Update count

    def swap_nitrogen(self):
        self.N_number +=1
        self.modify_molecule()
            
    def modify_molecule(self):
        if self.filtered_data.empty:
            return
        
        index = self.current_index % len(self.filtered_data)
        original_smiles = self.filtered_data.iloc[index]['Smiles']
        mol = Chem.MolFromSmiles(original_smiles)
        
        if not mol:
            return
        
        editable_mol = Chem.RWMol(mol)
        nitrogen_idx = None
        
        nitrogen_indices = [
            atom.GetIdx() for atom in mol.GetAtoms()
            if atom.GetSymbol() == "N"
        ]

        nitrogen_idx = nitrogen_indices[self.N_number % len(nitrogen_indices)]
        
        if nitrogen_idx is not None:
            if self.add_ch3_var.get():
                carbon_idx = editable_mol.AddAtom(Chem.Atom(6))
                editable_mol.AddBond(nitrogen_idx, carbon_idx, Chem.BondType.SINGLE)
                editable_mol.GetAtomWithIdx(nitrogen_idx).SetFormalCharge(1)
                self.modified_names[index] = f"{self.filtered_data.iloc[index]['Name']}_MCA"
            elif self.add_h_var.get():
                editable_mol.GetAtomWithIdx(nitrogen_idx).SetFormalCharge(1)
                self.modified_names[index] = f"{self.filtered_data.iloc[index]['Name']}_PA"
            else:  # remove suffixes if no modification is chosen ->original smile is used
                self.modified_names[index] = self.filtered_data.iloc[index]['Name']
        
        modified_mol = editable_mol.GetMol()
        charge = self.modified_charge[index] = Chem.GetFormalCharge(modified_mol)
        new_smiles = Chem.MolToSmiles(modified_mol)
        
        self.modified_smiles[index] = new_smiles

        self.display_updated_molecule(new_smiles)
        
    def display_image(self):
        img = Image.open('current_molecule.png')
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk  # Keep a reference

    def next_molecule(self):
        self.current_index = (self.current_index + 1) % len(self.filtered_data)
        self.show_molecule(self.current_index)

    def previous_molecule(self):
        self.current_index = (self.current_index - 1) % len(self.filtered_data)
        self.show_molecule(self.current_index)

    def switch_file(self, file_path):
        self.current_index = 0
        self.load_data(file_path)

        # Enable the solvent dropdown after a nucleophile file is selected
        self.solvent_menu.config(state="normal")
    

    def display_updated_molecule(self, smiles):
        """Displays the molecule image after modification."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol)
            img.save('current_molecule.png')
            self.display_image()

    def create_input_for_molecule(self, input_dir="data/xyz", output_dir="data/input_files/QCHEM"):
        """Generate input files using the current molecule, including modifications if applicable."""
        
        if self.filtered_data.empty:
            return

        index = self.current_index % len(self.filtered_data)
        # original_smiles = self.filtered_data.iloc[index]['Smiles']
        modified_smiles = self.modified_smiles[index]
        name = self.modified_names[index]
        charge = self.modified_charge[index]

        # Check if CH3+ should be added and modify SMILES accordingly
#        if self.add_ch3_var.get():
#            mol = Chem.MolFromSmiles(original_smiles)
#            if mol:
#                editable_mol = Chem.RWMol(mol)
#                nitrogen_idx = None
#
#                for atom in editable_mol.GetAtoms():
#                    if atom.GetSymbol() == "N":
#                        nitrogen_idx = atom.GetIdx()
#                        atom.SetFormalCharge(1)  # Ensure nitrogen gets a +1 charge
#                        break  
#
#                if nitrogen_idx is not None:
#                    # Add CH3+ to nitrogen
#                    carbon_idx = editable_mol.AddAtom(Chem.Atom(6))  # Add Carbon (C)
#
#                    # Attach CH3+ to nitrogen
#                    editable_mol.AddBond(nitrogen_idx, carbon_idx, Chem.BondType.SINGLE)
#
#                   modified_mol = editable_mol.GetMol()
#                    updated_smiles = Chem.MolToSmiles(modified_mol)
#                else:
#                    updated_smiles = original_smiles  # No nitrogen found, use original
#            else:
#                updated_smiles = original_smiles  # If SMILES parsing fails
#            name += "_MCA"  # Append "_MCA" to the filename
#        else:
#            updated_smiles = original_smiles  # If checkbox is unchecked, use original

        # Construct the expected XYZ file path with "_MCA" if CH3+ is added
        xyz_file = os.path.join(input_dir, f"{name}.xyz")

        # xyz_data = create_xyz_string(name, updated_smiles, charge)
        xyz_data = create_xyz_string(name, modified_smiles, charge)
        # Write the XYZ data to a file
        with open(xyz_file, 'w') as file:
            file.write(f"{xyz_data}\n")
        print(f"Generated and saved XYZ file: {xyz_file}")

        # Generate the Gaussian input file using the updated XYZ file
        # generate_qchem_input(xyz_file, output_dir)



    def create_input_for_category(self):
        # Iterate through the filtered data and create input files for each molecule
        for index in range(len(self.filtered_data)):
            self.current_index = index
            self.create_input_for_molecule()



# Example usage:
# root = tk.Tk()
# viewer = MoleculeViewer(root)
# root.mainloop()
