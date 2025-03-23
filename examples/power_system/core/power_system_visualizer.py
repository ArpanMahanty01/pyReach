import tkinter as tk
from tkinter import ttk, filedialog
from example.power_system.core.power_system_parser import PowerSystemParser
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np

class PowerSystemVisualizer:
    """GUI for visualizing PyPSA networks."""
    
    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
            self.root.title("PyPSA Network Visualizer")
            self.root.geometry("1200x800")
        else:
            self.root = root
            
        self.parser = PowerSystemParser()
        self.network = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the GUI components."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control frame (top)
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        # File selection
        ttk.Label(control_frame, text="INI File:").pack(side=tk.LEFT, padx=(0, 5))
        self.file_path_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Load Network", command=self.load_network).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Run Power Flow", command=self.run_power_flow).pack(side=tk.LEFT, padx=(0, 5))
        
        # Notebook for visualization and data display
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Graph visualization tab
        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="Network Graph")
        
        # Create the matplotlib figure for the graph
        self.graph_figure = Figure(figsize=(10, 6), dpi=100)
        self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, master=self.graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Data display tabs
        self.create_data_tabs()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
    def create_data_tabs(self):
        """Create tabs for displaying network component data."""
        component_types = [
            ("Buses", "buses"),
            ("Generators", "generators"),
            ("Loads", "loads"),
            ("Lines", "lines"),
            ("Transformers", "transformers"),
            ("Storage Units", "storage_units")
        ]
        
        self.data_frames = {}
        self.treeviews = {}
        
        for label, attr in component_types:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=label)
            self.data_frames[attr] = frame
            
            # Create treeview with scrollbars
            tree_frame = ttk.Frame(frame)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            tree_scroll_y = ttk.Scrollbar(tree_frame)
            tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            
            tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
            tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
            tree.pack(fill=tk.BOTH, expand=True)
            
            tree_scroll_y.config(command=tree.yview)
            tree_scroll_x.config(command=tree.xview)
            
            self.treeviews[attr] = tree
            
    def browse_file(self):
        """Open file dialog to select INI file."""
        file_path = filedialog.askopenfilename(
            title="Select INI File",
            filetypes=[("INI Files", "*.ini"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            
    def load_network(self):
        """Load the network from the selected INI file."""
        ini_file = self.file_path_var.get()
        if not ini_file:
            self.status_var.set("Error: No INI file selected")
            return
            
        try:
            self.status_var.set(f"Loading network from {ini_file}...")
            self.root.update_idletasks()
            
            self.network = self.parser.parse_ini_file(ini_file)
            self.status_var.set(f"Network loaded successfully: {self.network.name}")
            
            # Update visualization and data displays
            self.update_network_graph()
            self.update_data_displays()
            
        except Exception as e:
            self.status_var.set(f"Error loading network: {str(e)}")
            raise
            
    def run_power_flow(self):
        """Run power flow calculation on the network."""
        if self.network is None:
            self.status_var.set("Error: No network loaded")
            return
            
        try:
            self.status_var.set("Running power flow calculation...")
            self.root.update_idletasks()
            
            self.network.pf()
            self.status_var.set("Power flow calculation completed")
            
            # Update data displays with the new results
            self.update_data_displays()
            
        except Exception as e:
            self.status_var.set(f"Error in power flow calculation: {str(e)}")
            
    def update_network_graph(self):
        """Update the network graph visualization."""
        if self.network is None:
            return
            
        # Clear previous plot
        self.graph_figure.clear()
        ax = self.graph_figure.add_subplot(111)
        
        # Create a networkx graph
        G = nx.Graph()
        
        # Add buses as nodes
        for bus_name, bus in self.network.buses.iterrows():
            G.add_node(bus_name, pos=(float(bus.x), float(bus.y)))
            
        # Add lines as edges
        for line_name, line in self.network.lines.iterrows():
            G.add_edge(line.bus0, line.bus1, weight=line.length if hasattr(line, 'length') else 1.0)
            
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # If positions are not provided, use spring layout
        if not pos:
            pos = nx.spring_layout(G)
            
        # Draw nodes (buses)
        bus_types = {}
        for bus_name, bus in self.network.buses.iterrows():
            bus_type = 'normal'
            
            # Check if bus has generators
            if hasattr(self.network, 'generators') and not self.network.generators.empty:
                if bus_name in self.network.generators.bus.values:
                    bus_type = 'generator'
                    
            # Check if bus has loads
            if hasattr(self.network, 'loads') and not self.network.loads.empty:
                if bus_name in self.network.loads.bus.values:
                    if bus_type == 'generator':
                        bus_type = 'both'
                    else:
                        bus_type = 'load'
                        
            bus_types[bus_name] = bus_type
            
        # Draw with different colors based on bus type
        gen_nodes = [node for node, type in bus_types.items() if type == 'generator']
        load_nodes = [node for node, type in bus_types.items() if type == 'load']
        both_nodes = [node for node, type in bus_types.items() if type == 'both']
        normal_nodes = [node for node, type in bus_types.items() if type == 'normal']
        
        nx.draw_networkx_nodes(G, pos, nodelist=gen_nodes, node_color='green', node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=load_nodes, node_color='red', node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=both_nodes, node_color='purple', node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color='blue', node_size=500, alpha=0.8, ax=ax)
        
        # Draw edges (lines)
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.7, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Set title and remove axis
        ax.set_title(f"Network Graph: {self.network.name}")
        ax.set_axis_off()
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Generator Bus'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Load Bus'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Gen+Load Bus'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Connection Bus')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Update the canvas
        self.graph_canvas.draw()
        
    def update_data_displays(self):
        """Update the data display tabs with network component information."""
        if self.network is None:
            return
            
        self._update_component_table('buses', self.network.buses)
        self._update_component_table('generators', self.network.generators)
        self._update_component_table('loads', self.network.loads)
        self._update_component_table('lines', self.network.lines)
        self._update_component_table('transformers', self.network.transformers)
        self._update_component_table('storage_units', self.network.storage_units)
        
    def _update_component_table(self, component_type, component_df):
        """Update a specific component table with data."""
        tree = self.treeviews[component_type]
        
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
            
        # If the dataframe is empty, just return
        if component_df.empty:
            tree["columns"] = ("message",)
            tree.column("#0", width=0, stretch=tk.NO)
            tree.column("message", width=400, anchor=tk.CENTER)
            tree.heading("message", text="Message")
            tree.insert("", tk.END, values=(f"No {component_type} found in the network",))
            return
            
        # Set up columns based on the DataFrame
        columns = list(component_df.columns)
        tree["columns"] = columns
        
        # Configure columns
        tree.column("#0", width=0, stretch=tk.NO)  # Hide the first column
        for col in columns:
            tree.column(col, anchor=tk.W, width=100)
            tree.heading(col, text=col)
            
        # Add data rows
        for idx, row in component_df.iterrows():
            values = [row[col] for col in columns]
            formatted_values = []
            for val in values:
                if isinstance(val, (float, np.float64)):
                    formatted_values.append(f"{val:.4g}")
                else:
                    formatted_values.append(str(val))
            tree.insert("", tk.END, text=idx, values=formatted_values)
            
    def run(self):
        """Run the application main loop."""
        self.root.mainloop()