# src/gui/gui.py
import os
import sys
# Adjust the path so we can import from the core package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyReach.core.ContSet import contSet

import tkinter as tk
from tkinter import messagebox

# Try importing matplotlib and matplotlib-venn for visualization.
try:
    from matplotlib_venn import venn2
    import matplotlib.pyplot as plt
except ImportError:
    venn2 = None  # We’ll check later whether venn2 is available.


def parse_input(input_str):
    """
    Parse a comma-separated string into a list of nonempty, stripped items.
    """
    return [x.strip() for x in input_str.split(",") if x.strip()]


def show_venn_diagram(setA, setB, operation):
    """
    Display a Venn diagram for the two sets and highlight the region corresponding to the given operation.
    This function works only if matplotlib-venn is installed.
    """
    if venn2 is None:
        messagebox.showinfo("Visualization",
                            "matplotlib-venn is not installed, cannot display Venn diagram.")
        return

    # Convert the elements to Python sets.
    A = set(setA.elements.tolist())
    B = set(setB.elements.tolist())

    plt.figure()
    v = venn2([A, B], set_labels=('Set A', 'Set B'))

    # Highlight different regions based on the operation.
    if operation == "union":
        for region in ['10', '01', '11']:
            patch = v.get_patch_by_id(region)
            if patch:
                patch.set_alpha(0.5)
    elif operation == "intersection":
        patch = v.get_patch_by_id('11')
        if patch:
            patch.set_alpha(0.5)
    elif operation == "A-B":
        patch = v.get_patch_by_id('10')
        if patch:
            patch.set_alpha(0.5)
    elif operation == "B-A":
        patch = v.get_patch_by_id('01')
        if patch:
            patch.set_alpha(0.5)
    elif operation == "symmetric_difference":
        for region in ['10', '01']:
            patch = v.get_patch_by_id(region)
            if patch:
                patch.set_alpha(0.5)

    plt.title(f"Venn Diagram: {operation.replace('_', ' ').title()}")
    plt.show()


def main():
    # Create the main application window.
    root = tk.Tk()
    root.title("Set Operations Visualizer")

    # Create and place labels and entry fields for Set A and Set B.
    labelA = tk.Label(root, text="Set A (comma separated):")
    labelA.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    entryA = tk.Entry(root, width=40)
    entryA.grid(row=0, column=1, padx=5, pady=5)

    labelB = tk.Label(root, text="Set B (comma separated):")
    labelB.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    entryB = tk.Entry(root, width=40)
    entryB.grid(row=1, column=1, padx=5, pady=5)

    # Create a text widget to display the results.
    output_text = tk.Text(root, height=6, width=60)
    output_text.grid(row=3, column=0, columnspan=4, padx=5, pady=10)

    def perform_operation(operation):
        """
        Get the input from the two entry fields, perform the selected operation,
        display the result, and (if applicable) visualize the operation using a Venn diagram.
        """
        try:
            listA = parse_input(entryA.get())
            listB = parse_input(entryB.get())
            setA = contSet(listA)
            setB = contSet(listB)

            # Clear previous output.
            output_text.delete("1.0", tk.END)

            if operation == "union":
                result = setA.union(setB)
            elif operation == "intersection":
                result = setA.intersection(setB)
            elif operation == "A-B":
                result = setA.difference(setB)
            elif operation == "B-A":
                result = setB.difference(setA)
            elif operation == "symmetric_difference":
                result = setA.symmetric_difference(setB)
            elif operation == "cartesian_product":
                result = setA.cartesian_product(setB)
            elif operation == "is_subset":
                result = setA.is_subset(setB)
            elif operation == "is_superset":
                result = setA.is_superset(setB)
            else:
                result = "Unknown Operation"

            output_text.insert(tk.END, str(result))

            # For operations that make sense to visualize via a Venn diagram,
            # pop up a matplotlib window.
            if operation in ["union", "intersection", "A-B", "B-A", "symmetric_difference"]:
                show_venn_diagram(setA, setB, operation)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Create buttons for various set operations.
    btn_union = tk.Button(root, text="Union", width=18,
                          command=lambda: perform_operation("union"))
    btn_union.grid(row=2, column=0, padx=5, pady=5)

    btn_intersection = tk.Button(root, text="Intersection", width=18,
                                 command=lambda: perform_operation("intersection"))
    btn_intersection.grid(row=2, column=1, padx=5, pady=5)

    btn_diff_AB = tk.Button(root, text="A - B", width=18,
                            command=lambda: perform_operation("A-B"))
    btn_diff_AB.grid(row=2, column=2, padx=5, pady=5)

    btn_diff_BA = tk.Button(root, text="B - A", width=18,
                            command=lambda: perform_operation("B-A"))
    btn_diff_BA.grid(row=2, column=3, padx=5, pady=5)

    btn_sym_diff = tk.Button(root, text="Symmetric Difference", width=18,
                             command=lambda: perform_operation("symmetric_difference"))
    btn_sym_diff.grid(row=4, column=0, padx=5, pady=5)

    btn_cartesian = tk.Button(root, text="Cartesian Product", width=18,
                              command=lambda: perform_operation("cartesian_product"))
    btn_cartesian.grid(row=4, column=1, padx=5, pady=5)

    btn_subset = tk.Button(root, text="A ⊆ B ?", width=18,
                           command=lambda: perform_operation("is_subset"))
    btn_subset.grid(row=4, column=2, padx=5, pady=5)

    btn_superset = tk.Button(root, text="A ⊇ B ?", width=18,
                             command=lambda: perform_operation("is_superset"))
    btn_superset.grid(row=4, column=3, padx=5, pady=5)

    # Start the Tkinter event loop.
    root.mainloop()


if __name__ == '__main__':
    main()
