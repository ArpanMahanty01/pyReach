import tkinter as tk
from tkinter import ttk,messagebox
from examples.power_system.core.power_system_parser import PowerSystemParser
from examples.power_system.core.power_system_visualizer import PyPSANetworkPlotter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os

def get_power_system_network():
    parser = PowerSystemParser()
    return parser.parse_ini_file("./power_system.ini")


class FullscreenApp:
    def __init__(self, root):
        self.root = root
        self.root.attributes('-fullscreen', True) 
        self.root.bind("<Escape>", lambda e: self.root.quit()) 
        
        self.setup_ui()


    def setup_ui(self):
        """Create the UI components."""
        notebook = ttk.Notebook(self.root)  

        # Create tabs
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)

        notebook.add(tab1, text="Playground")
        notebook.add(tab2, text="Power System Analysis")

        notebook.pack(expand=True, fill="both") 

        self.create_playground_tab(tab1)
        self.create_power_system_analysis_tab(tab2)

    def create_playground_tab(self, tab):
        """Setup the Playground tab."""
        network = get_power_system_network()
        y_bus = network.adjacency_matrix().todense()
        print(y_bus)

    def create_power_system_analysis_tab(self, tab):
        """Setup the Power System Analysis tab with editable .ini file and network diagram."""
        # Create a PanedWindow to divide the tab into two sections
        paned_window = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left section - Contains both text editor and reachability analysis
        left_frame = ttk.Frame(paned_window)
        
        # Right section - Network diagram
        right_frame = ttk.LabelFrame(paned_window, text="Power System Network Diagram")
        
        # Change the weight ratio between left and right sections (1:3 instead of 1:2)
        paned_window.add(left_frame, weight=1)
        paned_window.add(right_frame, weight=2)
        
        # Create a vertical paned window inside the left frame to split it 50/50
        left_paned = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)
        
        # Top section - Configuration editor
        config_frame = ttk.LabelFrame(left_paned, text="Power System Configuration")
        
        # Bottom section - Reachability analysis
        reachability_frame = ttk.LabelFrame(left_paned, text="Reachability Analysis")
        
        # Add both frames to the paned window with very different weights to create dramatic height difference
        left_paned.add(config_frame, weight=1)    # Much smaller height for configuration
        left_paned.add(reachability_frame, weight=6)  # Much larger height for reachability analysis
        
        # Text section frame with update button at top right
        text_container = ttk.Frame(config_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        # Add Update Diagram button at the top right
        update_button = ttk.Button(text_container, text="Update Diagram", command=self.update_diagram)
        update_button.pack(side=tk.TOP, anchor=tk.NE, padx=5, pady=5)
        
        # Add a Text widget for editing the .ini file without scrollbars
        self.config_text = tk.Text(text_container, wrap="none")
        
        # Layout for text widget
        self.config_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add placeholder content for reachability analysis
        # reachability_label = ttk.Label(reachability_frame, text="Reachability analysis tools will be displayed here")
        # reachability_label.pack(pady=20)
        
        # Create a frame for the matplotlib figure
        self.plot_frame = ttk.Frame(right_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initially load the config file and draw the diagram
        self.load_config()
        self.update_diagram()

    def load_config(self):
        """Load the power system configuration from the .ini file."""
        try:
            if os.path.exists("./power_system.ini"):
                with open("./power_system.ini", "r") as file:
                    config_content = file.read()
                
                # Clear the text widget and insert the file content
                self.config_text.delete(1.0, tk.END)
                self.config_text.insert(tk.END, config_content)
            else:
                messagebox.showwarning("File Not Found", "power_system.ini file not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading configuration: {str(e)}")
    
    def save_config(self):
        """Save the edited configuration to the .ini file."""
        try:
            config_content = self.config_text.get(1.0, tk.END)
            
            with open("./power_system.ini", "w") as file:
                file.write(config_content)
        except Exception as e:
            messagebox.showerror("Error", f"Error saving configuration: {str(e)}")

    def update_diagram(self):
        """Update the network diagram using network.plot function."""
        try:
            # Save the current configuration first
            self.save_config()
            
            # Get the power system network
            network = get_power_system_network()
            
            # Clear existing plot
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 6))

            PyPSANetworkPlotter.create_network_plot(network, fig, ax)

            # Create canvas to display the plot
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error updating diagram: {str(e)}")
            # Print detailed error information
            import traceback
            traceback.print_exc()
    
    def show_error(self, message):
        """Display an error message."""
        messagebox.showerror("Error", message)

    def show_info(self, message):
        """Display an information message."""
        messagebox.showinfo("Information", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = FullscreenApp(root)
    root.mainloop()