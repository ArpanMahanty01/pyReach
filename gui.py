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
        """
        Create the playground tab with Example 7 from the pyReach paper.
        This example demonstrates a distributed computation of projections of a 
        high-dimensional extrusion generated polytope.
        """
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import tkinter as tk
        from tkinter import ttk
        
        # Main frame to hold everything
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create title and description
        # title_label = ttk.Label(main_frame, 
        #                     text="Example 7: Distributed Computation of Extrusion Generated Polytope",
        #                     font=("Arial", 14, "bold"))
        # title_label.pack(pady=5)
        
        # desc_text = ("This visualization demonstrates Example 7 from the pyReach paper, which shows\n"
        #             "how a high-dimensional polytope can be computed in a distributed way using\n"
        #             "Algorithm 1. The graph has 5 nodes, each with an associated axis set and polytope.")
        # desc_label = ttk.Label(main_frame, text=desc_text)
        # desc_label.pack(pady=5)

        # Create a frame to hold the visualization components
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Graph visualization
        left_frame = ttk.LabelFrame(content_frame, text="Graph Topology")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Polytope projections visualization
        right_frame = ttk.LabelFrame(content_frame, text="Polytope Projections")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the graph visualization
        graph_frame = ttk.Frame(left_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Define the axis sets for Example 7
        axis_sets = {
            1: [1, 2],
            2: [3, 4],
            3: [5, 6],
            4: [1, 3, 5],
            5: [2, 7]
        }
        
        # Define the initial polytopes
        initial_polytopes = {
            1: np.array([[1, 2], [3, 2], [2, 4]]),
            2: np.array([[2, 4], [3, 3], [2, 0]]),
            3: np.array([[5, 5], [4, 0], [2, 0]]),
            4: np.array([[0, 1, 4], [3, 3, 0], [5, 0, 3], [5, 2, 5]]),
            5: np.array([[2, 1], [4, 1], [5, 3]])
        }
        
        # Define the final projections (solutions)
        final_projections = {
            1: np.array([[2, 2], [3, 2], [2, 4]]),
            2: np.array([[2, 4], [2, 0], [2.39, 1.17], [2.39, 3.61]]),
            3: np.array([[3.29, 0], [3.29, 2.14], [2, 0]]),
            4: np.array([[3, 2, 2], [3, 2.39, 2], [3, 2, 3.29], 
                    [2, 2, 2.43], [2, 2.13, 2], [2, 2, 2]]),
            5: np.array([[2, 1], [3, 1], [3, 1.67]])
        }
        
        # Create the graph figure
        graph_fig = plt.Figure(figsize=(5, 4), dpi=100)
        graph_ax = graph_fig.add_subplot(111)
        graph_canvas = FigureCanvasTkAgg(graph_fig, graph_frame)
        graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Node positions
        node_pos = {
            1: [0.2, 0.7],
            2: [0.5, 0.8],
            3: [0.8, 0.7],
            4: [0.3, 0.3],
            5: [0.7, 0.3]
        }
        
        # Draw the graph
        graph_ax.set_xlim(0, 1)
        graph_ax.set_ylim(0, 1)
        graph_ax.set_title("Graph Topology for Example 7")
        graph_ax.axis('off')
        
        # Draw edges
        edges = [(1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (4, 5)]
        for i, j in edges:
            graph_ax.plot([node_pos[i][0], node_pos[j][0]], 
                        [node_pos[i][1], node_pos[j][1]], 'k-', lw=1.5)
        
        # Draw nodes
        for node, pos in node_pos.items():
            graph_ax.scatter(pos[0], pos[1], s=500, c='skyblue', edgecolors='black', zorder=10)
            graph_ax.text(pos[0], pos[1], str(node), ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add axis set labels
        for node, pos in node_pos.items():
            label = f"B{node} = {axis_sets[node]}"
            graph_ax.text(pos[0], pos[1]-0.07, label, ha='center', va='center', fontsize=8)
        
        graph_fig.tight_layout()
        graph_canvas.draw()
        
        # Create the polytope visualization
        polytope_frame = ttk.Frame(right_frame)
        polytope_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls for iterations
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        iteration_label = ttk.Label(control_frame, text="Iteration:")
        iteration_label.pack(side=tk.LEFT, padx=5)
        
        # Define the intermediate iterations
        iterations = {
            0: initial_polytopes,  # Initial state
            1: {  # First iteration results
                1: np.array([[2, 2], [3, 2], [2, 4]]),
                2: np.array([[2, 4], [2, 0], [2.39, 1.17], [2.39, 3.61]]),
                3: np.array([[6, -3], [1, 0], [-1, 0]]),
                4: np.array([[-2, 0, 1], [5, 3, -2], [1, 7, -4]]),
                5: np.array([[7, -4], [0, 1], [3, -2]])
            },
            2: final_projections  # Final converged state
        }
        
        current_iteration = tk.IntVar(value=0)
        
        def update_polytope_display():
            nonlocal polytope_fig, polytope_canvas
            iteration_data = iterations[current_iteration.get()]
            
            # Clear previous plots
            for node, ax in polytope_axes.items():
                if node != 'summary':  # Skip the summary plot
                    ax.clear()
            
            # Update the plots with current iteration data
            for node, ax in polytope_axes.items():
                if node == 'summary':  # Skip summary in this loop
                    continue
                    
                points = iteration_data[node]
                
                # For 2D polytopes
                if points.shape[1] == 2:
                    # Try to compute the convex hull for better visualization
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        ax.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.5, color='skyblue', edgecolor='black')
                    except:
                        # Fallback if ConvexHull fails
                        ax.scatter(points[:, 0], points[:, 1], color='blue')
                        ax.plot(np.append(points[:, 0], points[0, 0]), 
                                np.append(points[:, 1], points[0, 1]), 'k-')
                    
                    for i, point in enumerate(points):
                        ax.text(point[0], point[1], f"({point[0]}, {point[1]})", fontsize=8)
                
                # For 3D polytopes (node 4)
                elif points.shape[1] == 3:
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue')
                    # Connect points to visualize the polytope shape
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            ax.plot([points[i, 0], points[j, 0]], 
                                    [points[i, 1], points[j, 1]], 
                                    [points[i, 2], points[j, 2]], 'k-', alpha=0.3)
                                    
                    # Add point labels
                    for point in points:
                        ax.text(point[0], point[1], point[2], 
                            f"({point[0]}, {point[1]}, {point[2]})", fontsize=7)
                
                # Set labels
                if node == 4:  # 3D plot
                    ax.set_xlabel('z1')
                    ax.set_ylabel('z3')
                    ax.set_zlabel('z5')
                else:  # 2D plot
                    ax.set_xlabel(f'z{axis_sets[node][0]}')
                    ax.set_ylabel(f'z{axis_sets[node][1]}')
                
                ax.set_title(f"Node {node}: B{node} = {axis_sets[node]}")
                ax.grid(True)
            
            # Update the summary plot separately
            update_summary_plot()
            
            polytope_fig.tight_layout()
            polytope_canvas.draw()
        
        # Create iteration control buttons
        prev_btn = ttk.Button(control_frame, text="Previous", 
                            command=lambda: [current_iteration.set(max(0, current_iteration.get()-1)), 
                                            update_polytope_display()])
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = ttk.Button(control_frame, text="Next", 
                            command=lambda: [current_iteration.set(min(2, current_iteration.get()+1)), 
                                            update_polytope_display()])
        next_btn.pack(side=tk.LEFT, padx=5)
        
        iteration_display = ttk.Label(control_frame, textvariable=current_iteration)
        iteration_display.pack(side=tk.LEFT, padx=5)
        
        status_var = tk.StringVar(value="Initial polytopes")
        status_label = ttk.Label(control_frame, textvariable=status_var)
        status_label.pack(side=tk.LEFT, padx=20)
        
        def update_status():
            iter_val = current_iteration.get()
            if iter_val == 0:
                status_var.set("Initial polytopes")
            elif iter_val == 1:
                status_var.set("After first iteration")
            else:
                status_var.set("Final converged projections")
        
        current_iteration.trace_add("write", lambda *args: update_status())
        
        # Explanation for each step
        explanation_frame = ttk.LabelFrame(control_frame, text="Current Step Explanation")
        explanation_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5, expand=True)
        
        explanation_text = tk.Text(explanation_frame, height=4, wrap=tk.WORD)
        explanation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        def update_explanation():
            iter_val = current_iteration.get()
            explanation_text.config(state=tk.NORMAL)
            explanation_text.delete(1.0, tk.END)
            
            if iter_val == 0:
                text = ("Initial step: Each node starts with its own local polytope S_i(0). "
                    "These initial sets may not be consistent with each other across the network. "
                    "The algorithm will iteratively refine these sets to ensure global consistency.")
            elif iter_val == 1:
                text = ("First iteration: Each node shares its polytope with its neighbors. "
                    "Then each node computes a new polytope by projecting the intersection of "
                    "the extruded sets from its neighbors. This step begins to enforce consistency "
                    "between neighboring nodes.")
            else:
                text = ("Final iteration: The algorithm has converged. Each node now has the exact "
                    "projection of the global solution onto its local axis set. "
                    "The distributed computation has achieved the same result as would be obtained "
                    "with a centralized approach, demonstrating Theorem 2 from the paper.")
            
            # explanation_text.insert(tk.END, text)
            # explanation_text.config(state=tk.DISABLED)
        
        current_iteration.trace_add("write", lambda *args: update_explanation())
        
        # Create the polytope figure with subplots
        polytope_fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Create individual axes for each node's polytope
        polytope_axes = {}
        polytope_axes[1] = polytope_fig.add_subplot(2, 3, 1)
        polytope_axes[2] = polytope_fig.add_subplot(2, 3, 2)
        polytope_axes[3] = polytope_fig.add_subplot(2, 3, 3)
        polytope_axes[4] = polytope_fig.add_subplot(2, 3, 4, projection='3d')
        polytope_axes[5] = polytope_fig.add_subplot(2, 3, 5)
        
        # Add a summary subplot
        summary_ax = polytope_fig.add_subplot(2, 3, 6)
        summary_ax.axis('off')
        summary_ax.set_title("Summary")
        polytope_axes['summary'] = summary_ax
        
        # Update function to manage the summary plot
        def update_summary_plot():
            summary_ax.clear()
            summary_ax.axis('off')
            summary_ax.set_title("Summary")
            
            iter_val = current_iteration.get()
            text_content = [
                f"Iteration: {iter_val}",
                "",
                "Network Stats:",
                f"• Nodes: 5",
                f"• Edges: {len(edges)}",
                f"• Axis Sets: 7 dimensions total",
                "",
                "Convergence Progress:",
                f"• {'✓' if iter_val >= 2 else '○'} Convergence achieved",
                f"• {'✓' if iter_val >= 1 else '○'} Local consistency",
                f"• {'✓' if iter_val >= 2 else '○'} Global consistency",
            ]
            
            y_pos = 0.9
            for line in text_content:
                summary_ax.text(0.1, y_pos, line, ha='left', fontsize=9)
                y_pos -= 0.07
        
        # Add summary plot update to the main update function
        original_update_polytope = update_polytope_display
        def enhanced_update_polytope():
            original_update_polytope()
            # Note: update_summary_plot is now called inside original_update_polytope
        
        update_polytope_display = enhanced_update_polytope
        
        polytope_canvas = FigureCanvasTkAgg(polytope_fig, polytope_frame)
        polytope_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Description text for the algorithm
        algorithm_frame = ttk.LabelFrame(main_frame, text="Algorithm Description")
        algorithm_frame.pack(fill=tk.X, padx=10, pady=10)
        
        algorithm_text = ("Algorithm 1 (Distributed Extrusion Generated Set):\n"
                        "1. Each node i initializes with a local polytope S_i(0)\n"
                        "2. At each iteration κ, each node sends its current set to its neighbors\n"
                        "3. Node i computes S_i(κ+1) using the formula:\n"
                        "   S_i(κ+1) = P_{B_M_i}^{B_i}[∩_{j∈M_i} E_{B_M_i}^{B_j}(S_j(κ))]\n"
                        "4. The algorithm converges to the local projection of the centralized solution\n"
                        "5. In this example, convergence is achieved after only 2 iterations")
        
        algorithm_label = ttk.Label(algorithm_frame, text=algorithm_text, justify=tk.LEFT)
        algorithm_label.pack(padx=10, pady=10)
        
        # Add a mathematical explanation section
        math_frame = ttk.LabelFrame(main_frame, text="Mathematical Background")
        math_frame.pack(fill=tk.X, padx=10, pady=10)
        
        math_text = ("Key Concepts from the pyReach Paper:\n\n"
                    "• Extrusion Generated Set: A set that can be reconstructed from its projections\n"
                    "• Projection Operator (P): Projects high-dimensional vectors to specific axis subsets\n"
                    "• Extrusion Operator (E): Extends low-dimensional vectors to higher dimensions\n"
                    "• The distributed approach computes the projections of the centralized solution\n"
                    "• Each node i retains only the minimal information needed for the distributed computation\n\n"
                    "Theorem 2 guarantees that Algorithm 1 will converge to the correct projection")
        
        math_label = ttk.Label(math_frame, text=math_text, justify=tk.LEFT)
        math_label.pack(padx=10, pady=10)
        
        # Add animation controls
        animation_frame = ttk.Frame(control_frame)
        animation_frame.pack(side=tk.RIGHT, padx=10)
        
        animate_button = ttk.Button(animation_frame, text="Animate Algorithm", 
                                command=lambda: start_animation())
        animate_button.pack(side=tk.LEFT, padx=5)
        
        speed_label = ttk.Label(animation_frame, text="Speed:")
        speed_label.pack(side=tk.LEFT, padx=5)
        
        speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(animation_frame, from_=0.5, to=3.0, 
                            variable=speed_var, orient=tk.HORIZONTAL, length=100)
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        def start_animation():
            """Start automated animation of the algorithm iterations"""
            current_iteration.set(0)
            update_polytope_display()
            
            def next_frame():
                current = current_iteration.get()
                if current < 2:
                    current_iteration.set(current + 1)
                    update_polytope_display()
                    # Schedule next frame based on speed
                    delay = int(1000 / speed_var.get())
                    if current + 1 < 2:
                        animation_id = tab.after(delay, next_frame)
                        setattr(self, 'animation_id', animation_id)
            
            # Schedule first transition
            delay = int(1000 / speed_var.get())
            animation_id = tab.after(delay, next_frame)
            setattr(self, 'animation_id', animation_id)
        
        # Initialize the display
        update_polytope_display()

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