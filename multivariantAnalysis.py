import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class CSVPlotter:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        if 'solvent' not in self.data.columns:
            raise ValueError("CSV must have a column named 'solvent'.")
        self.data = self.data[self.data['solvent'].str.contains('AN', na=False)]
        if 'N' not in self.data.columns:
            raise ValueError("CSV must have a column named 'N'.")
        self.columns = [col for col in self.data.columns if col != 'N' and col != 'solvent']
        if len(self.columns) == 0:
            raise ValueError("CSV must have at least one column to plot besides 'N' and 'solvent'.")

        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        self.plot_data()

        self.axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        
        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next_column)
        
        self.bprev = Button(self.axprev, 'Previous')
        self.bprev.on_clicked(self.previous_column)

    def plot_data(self):
        self.ax.clear()
        x = self.data['N']
        y = self.data[self.columns[self.current_index]]
        labels = self.data['para group']  # Extract "para group" values

        self.ax.scatter(x, y, label=f"N vs {self.columns[self.current_index]}", alpha=0.7)

        # Annotate each point with its "para group" value
        for i in range(len(x)):
            self.ax.text(x.iloc[i], y.iloc[i], str(labels.iloc[i]), fontsize=10, ha='right', va='bottom', fontweight='bold')

        self.ax.set_xlabel('N', fontsize=14)
        self.ax.set_ylabel(self.columns[self.current_index], fontsize=14)
        self.ax.legend(fontsize=12)
        self.ax.grid(True)
        self.fig.canvas.draw()



    def next_column(self, event):
        self.current_index = (self.current_index + 1) % len(self.columns)
        self.plot_data()

    def previous_column(self, event):
        self.current_index = (self.current_index - 1) % len(self.columns)
        self.plot_data()

# Usage:
# Replace 'your_file.csv' with the path to your CSV file.
# Ensure the column 'N' and 'solvent' exist in your CSV file.

csv_file = 'data/multivariant.csv'  # Replace with your CSV file
plotter = CSVPlotter(csv_file)
plt.show()
