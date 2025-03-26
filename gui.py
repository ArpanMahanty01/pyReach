import tkinter as tk
from tkinter import ttk, messagebox
from examples.power_system.core.power_system_parser import PowerSystemParser
from examples.power_system.core.power_system_visualizer import PyPSANetworkPlotter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os

def get_power_system_network():
    parser = PowerSystemParser()
    return parser.parse_ini_file("./power_system.ini")

class PowerSystemAnalyzer:
    @staticmethod
    def compute_y_bus(network):
        K = network.incidence_matrix()
        Y_branch = np.diag(1 / network.lines.x.values)
        return K @ Y_branch @ K.T

class ConfigManager:
    @staticmethod
    def load_config_from_file(filename):
        if os.path.exists(filename):
            with open(filename, "r") as file:
                return file.read()
        return None
    
    @staticmethod
    def save_config_to_file(filename, content):
        with open(filename, "w") as file:
            file.write(content)

class NetworkVisualizer:
    @staticmethod
    def create_plot(network, frame):
        for widget in frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        PyPSANetworkPlotter.create_network_plot(network, fig, ax)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return canvas

class FullscreenApp:
    def __init__(self, root):
        self.root = root
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.config_file = "./power_system.ini"
        self.setup_ui()

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        notebook.add(tab1, text="Playground")
        notebook.add(tab2, text="Power System Analysis")
        notebook.pack(expand=True, fill="both")
        self.create_playground_tab(tab1)
        self.create_power_system_analysis_tab(tab2)

    def create_playground_tab(self, tab):
        pass

    def create_power_system_analysis_tab(self, tab):
        paned_window = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(paned_window)
        right_frame = ttk.LabelFrame(paned_window, text="Power System Network Diagram")
        paned_window.add(left_frame, weight=1)
        paned_window.add(right_frame, weight=2)
        
        left_paned = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)
        
        config_frame = ttk.LabelFrame(left_paned, text="Power System Configuration")
        reachability_frame = ttk.LabelFrame(left_paned, text="Reachability Analysis")
        left_paned.add(config_frame, weight=1)
        left_paned.add(reachability_frame, weight=6)
        
        text_container = ttk.Frame(config_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        update_button = ttk.Button(text_container, text="Update Diagram", command=self.update_diagram)
        update_button.pack(side=tk.TOP, anchor=tk.NE, padx=5, pady=5)
        
        self.config_text = tk.Text(text_container, wrap="none")
        self.config_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        y_bus_frame = ttk.LabelFrame(text_container, text="Y BUS MATRIX")
        y_bus_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.y_bus_text = tk.Text(y_bus_frame, wrap="none", state="disabled")
        self.y_bus_text.pack(fill=tk.BOTH, expand=True)
        
        self.plot_frame = ttk.Frame(right_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.load_config()
        self.update_diagram()

    def load_config(self):
        try:
            config_content = ConfigManager.load_config_from_file(self.config_file)
            if config_content:
                self.config_text.delete(1.0, tk.END)
                self.config_text.insert(tk.END, config_content)
            else:
                messagebox.showwarning("File Not Found", f"{self.config_file} file not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading configuration: {str(e)}")
    
    def save_config(self):
        try:
            config_content = self.config_text.get(1.0, tk.END)
            ConfigManager.save_config_to_file(self.config_file, config_content)
        except Exception as e:
            messagebox.showerror("Error", f"Error saving configuration: {str(e)}")

    def update_y_bus_display(self, y_bus):
        self.y_bus_text.config(state="normal")
        self.y_bus_text.delete(1.0, tk.END)
        self.y_bus_text.insert(tk.END, str(y_bus))
        self.y_bus_text.config(state="disabled")

    def update_diagram(self):
        try:
            self.save_config()
            network = get_power_system_network()
            
            # Recompute Y_bus matrix
            y_bus = PowerSystemAnalyzer.compute_y_bus(network)
            self.update_y_bus_display(y_bus)
            
            # Update network visualization
            NetworkVisualizer.create_plot(network, self.plot_frame)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error updating diagram: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = FullscreenApp(root)
    root.mainloop()