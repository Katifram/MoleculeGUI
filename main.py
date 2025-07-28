import tkinter as tk
from molecule_viewer import MoleculeViewer

if __name__ == "__main__":
    root = tk.Tk()
    viewer = MoleculeViewer(root)
    root.mainloop()
