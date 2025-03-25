import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class PyPSANetworkPlotter:
    """
    Utility class for creating PyPSA network visualizations.
    This class is decoupled from any GUI framework and can be used with any
    Matplotlib-compatible canvas.
    """
    
    @staticmethod
    def create_network_plot(network, figure=None, ax=None):
        """
        Create a visualization of a PyPSA network.
        
        Parameters:
        -----------
        network : PyPSA Network object
            The network to visualize
        figure : matplotlib Figure, optional
            If provided, this figure will be used for plotting
        ax : matplotlib Axes, optional
            If provided, this axes will be used for plotting
            
        Returns:
        --------
        figure : matplotlib Figure
            The figure containing the network plot
        ax : matplotlib Axes
            The axes containing the network plot
        """
        # Create figure and axes if not provided
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        if ax is None:
            ax = figure.add_subplot(111)
        else:
            ax.clear()
        
        # Create a networkx graph
        G = nx.Graph()
        
        # Add buses as nodes
        for bus_name, bus in network.buses.iterrows():
            # Use x, y coordinates if available, otherwise use 0, 0
            x = float(bus.x) if hasattr(bus, 'x') else 0
            y = float(bus.y) if hasattr(bus, 'y') else 0
            G.add_node(bus_name, pos=(x, y))
            
        # Add lines as edges
        if hasattr(network, 'lines') and not network.lines.empty:
            for line_name, line in network.lines.iterrows():
                weight = line.length if hasattr(line, 'length') else 1.0
                G.add_edge(line.bus0, line.bus1, weight=weight)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # If positions are not provided, use spring layout
        if not pos:
            pos = nx.spring_layout(G)
            
        # Categorize buses as slack, load, or generator
        bus_types = {}
        for bus_name in G.nodes():
            # Default to load bus unless identified otherwise
            bus_type = 'load'
            
            # Check if bus is a slack bus (reference bus)
            if hasattr(network, 'buses') and 'control' in network.buses.columns:
                bus_data = network.buses.loc[bus_name]
                if hasattr(bus_data, 'control') and bus_data.control == 'Slack':
                    bus_type = 'slack'
                    continue  # Slack bus identification takes precedence
            
            # Check if bus has generators
            if hasattr(network, 'generators') and not network.generators.empty:
                if bus_name in network.generators.bus.values:
                    bus_type = 'generator'
                    
            bus_types[bus_name] = bus_type
            
        # Draw with different colors based on bus type
        slack_nodes = [node for node, type in bus_types.items() if type == 'slack']
        gen_nodes = [node for node, type in bus_types.items() if type == 'generator']
        load_nodes = [node for node, type in bus_types.items() if type == 'load']
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=slack_nodes, node_color='gold', 
                               node_size=600, alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=gen_nodes, node_color='green', 
                               node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=load_nodes, node_color='red', 
                               node_size=500, alpha=0.8, ax=ax)
        
        # Draw edges (lines)
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.7, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        
        # Remove axis
        ax.set_axis_off()
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10, 
                   label='Slack Bus'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                   label='Generator Bus'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
                   label='Load Bus')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Adjust layout
        figure.tight_layout()
        
        return figure, ax
    
    @staticmethod
    def get_power_flow_colors(network):
        """
        Get edge colors based on power flow values.
        
        Parameters:
        -----------
        network : PyPSA Network object
            The network with power flow results
            
        Returns:
        --------
        edge_colors : list
            List of colors for each edge in the network
        edge_widths : list
            List of widths for each edge in the network
        """
        edge_colors = []
        edge_widths = []
        
        if hasattr(network, 'lines') and not network.lines.empty and 'p0' in network.lines.columns:
            # Normalize power flows for color mapping
            p_values = np.abs(network.lines.p0.values)
            p_max = p_values.max() if len(p_values) > 0 else 1
            
            for line_name, line in network.lines.iterrows():
                # Calculate loading percentage
                if hasattr(line, 's_nom') and line.s_nom > 0:
                    loading = abs(line.p0) / line.s_nom
                else:
                    loading = abs(line.p0) / p_max if p_max > 0 else 0
                
                # Determine color based on loading
                if loading > 0.9:
                    edge_colors.append('red')
                elif loading > 0.7:
                    edge_colors.append('orange')
                elif loading > 0.5:
                    edge_colors.append('yellow')
                else:
                    edge_colors.append('green')
                
                # Width based on absolute power flow
                width = 1 + 3 * (abs(line.p0) / p_max) if p_max > 0 else 1
                edge_widths.append(width)
        
        return edge_colors, edge_widths

    @staticmethod
    def create_power_flow_plot(network, figure=None, ax=None):
        """
        Create a visualization of power flow results in a PyPSA network.
        
        Parameters:
        -----------
        network : PyPSA Network object
            The network with power flow results
        figure : matplotlib Figure, optional
            If provided, this figure will be used for plotting
        ax : matplotlib Axes, optional
            If provided, this axes will be used for plotting
            
        Returns:
        --------
        figure : matplotlib Figure
            The figure containing the power flow plot
        ax : matplotlib Axes
            The axes containing the power flow plot
        """
        # First create the basic network plot
        figure, ax = PyPSANetworkPlotter.create_network_plot(network, figure, ax)
        
        # If power flow results exist, update the plot with flow information
        if hasattr(network, 'lines') and not network.lines.empty and 'p0' in network.lines.columns:
            # Create a new networkx graph for the power flow visualization
            G = nx.Graph()
            
            # Add buses as nodes
            for bus_name, bus in network.buses.iterrows():
                x = float(bus.x) if hasattr(bus, 'x') else 0
                y = float(bus.y) if hasattr(bus, 'y') else 0
                G.add_node(bus_name, pos=(x, y))
            
            # Add lines as edges with power flow data
            edge_list = []
            for line_name, line in network.lines.iterrows():
                G.add_edge(line.bus0, line.bus1, 
                           weight=line.length if hasattr(line, 'length') else 1.0,
                           power=line.p0)
                edge_list.append((line.bus0, line.bus1))
            
            # Get node positions
            pos = nx.get_node_attributes(G, 'pos')
            if not pos:
                pos = nx.spring_layout(G)
            
            # Get edge colors and widths based on power flow
            edge_colors, edge_widths = PyPSANetworkPlotter.get_power_flow_colors(network)
            
            # Clear the previous edges and draw new ones with power flow information
            ax.collections[1].remove()  # Remove the old edges
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=edge_widths, 
                                  edge_color=edge_colors, alpha=0.8, ax=ax)
            
            # Update the title
            ax.set_title(f"Power Flow Results: {network.name}")
            
            # Add a legend for power flow
            from matplotlib.lines import Line2D
            flow_legend = [
                Line2D([0], [0], color='green', lw=2, label='< 50% Loading'),
                Line2D([0], [0], color='yellow', lw=2, label='50-70% Loading'),
                Line2D([0], [0], color='orange', lw=2, label='70-90% Loading'),
                Line2D([0], [0], color='red', lw=2, label='> 90% Loading')
            ]
            
            # Get the existing legend handles and add the new ones
            handles, labels = ax.get_legend_handles_labels()
            all_handles = handles + flow_legend
            all_labels = labels + [h.get_label() for h in flow_legend]
            
            # Remove the old legend and create a new one
            ax.get_legend().remove()
            ax.legend(handles=all_handles, labels=all_labels, loc='best')
        
        figure.tight_layout()
        return figure, ax