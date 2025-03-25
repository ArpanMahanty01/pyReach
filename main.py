# example_linear_affine_system.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from pyReach.core.AxisSet import AxisSet
from pyReach.core.Polytope import Polytope
from pyReach.core.Network import Network
from pyReach.core.Agent import Agent
from pyReach.operations.create_axis_sets import create_axis_sets
from pyReach.operations.distributed_backward_reachability import distributed_backward_reachability


def create_linear_affine_network():
    """
    Create a network of linear affine systems:
    - 3 agents with state coupling
    - Agent 1 influences Agent 2
    - Agent 2 influences Agent 3
    - Agent 3 influences Agent 1
    """
    # Create the network
    network = Network()
    
    # Create agents with 2D states and 1D inputs
    agent1 = Agent(agent_id=1, state_dim=2, input_dim=1, system_type='linear_affine')
    agent2 = Agent(agent_id=2, state_dim=2, input_dim=1, system_type='linear_affine')
    agent3 = Agent(agent_id=3, state_dim=2, input_dim=1, system_type='linear_affine')
    
    # Add agents to the network
    network.add_agent(1, agent1)
    network.add_agent(2, agent2)
    network.add_agent(3, agent3)
    
    # Define dynamics for Agent 1
    # x₁(t+1) = A₁₁x₁(t) + A₁₃x₃(t) + B₁₁u₁(t) + K₁
    A11 = np.array([[0.8, 0.1], [0, 0.9]])  # Self dynamics
    A13 = np.array([[0, 0.2], [0.1, 0]])    # Influence from Agent 3
    B11 = np.array([[1.0], [0.5]])          # Input matrix
    K1 = np.array([[0.1], [0.1]])           # Affine term
    
    agent1.set_linear_dynamics(1, A11, B11)
    agent1.set_linear_dynamics(3, A13, np.zeros((2, 1)))
    agent1.set_affine_term(K1)
    
    # Define dynamics for Agent 2
    # x₂(t+1) = A₂₂x₂(t) + A₂₁x₁(t) + B₂₂u₂(t) + K₂
    A22 = np.array([[0.85, 0.05], [-0.05, 0.85]])  # Self dynamics
    A21 = np.array([[0.1, 0], [0, 0.1]])           # Influence from Agent 1
    B22 = np.array([[0.8], [0.4]])                 # Input matrix
    K2 = np.array([[-0.1], [0.2]])                 # Affine term
    
    agent2.set_linear_dynamics(2, A22, B22)
    agent2.set_linear_dynamics(1, A21, np.zeros((2, 1)))
    agent2.set_affine_term(K2)
    
    # Define dynamics for Agent 3
    # x₃(t+1) = A₃₃x₃(t) + A₃₂x₂(t) + B₃₃u₃(t) + K₃
    A33 = np.array([[0.9, 0], [0.05, 0.85]])  # Self dynamics
    A32 = np.array([[0, 0.15], [0.1, 0]])     # Influence from Agent 2
    B33 = np.array([[0.7], [0.6]])            # Input matrix
    K3 = np.array([[0], [-0.1]])              # Affine term
    
    agent3.set_linear_dynamics(3, A33, B33)
    agent3.set_linear_dynamics(2, A32, np.zeros((2, 1)))
    agent3.set_affine_term(K3)
    
    # Define state constraints for each agent (Box constraints)
    # Agent 1: -5 ≤ x₁ ≤ 5, -5 ≤ x₂ ≤ 5
    A1_state = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
    b1_state = np.array([5, 5, 5, 5])
    agent1.set_state_constraints(Polytope.from_hyperplanes(A1_state, b1_state))
    
    # Agent 2: -5 ≤ x₁ ≤ 5, -5 ≤ x₂ ≤ 5
    A2_state = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
    b2_state = np.array([5, 5, 5, 5])
    agent2.set_state_constraints(Polytope.from_hyperplanes(A2_state, b2_state))
    
    # Agent 3: -5 ≤ x₁ ≤ 5, -5 ≤ x₂ ≤ 5
    A3_state = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
    b3_state = np.array([5, 5, 5, 5])
    agent3.set_state_constraints(Polytope.from_hyperplanes(A3_state, b3_state))
    
    # Define input constraints for each agent
    # Agent 1: -2 ≤ u₁ ≤ 2
    A1_input = np.array([[1], [-1]])
    b1_input = np.array([2, 2])
    agent1.set_input_constraints(Polytope.from_hyperplanes(A1_input, b1_input))
    
    # Agent 2: -2 ≤ u₂ ≤ 2
    A2_input = np.array([[1], [-1]])
    b2_input = np.array([2, 2])
    agent2.set_input_constraints(Polytope.from_hyperplanes(A2_input, b2_input))
    
    # Agent 3: -2 ≤ u₃ ≤ 2
    A3_input = np.array([[1], [-1]])
    b3_input = np.array([2, 2])
    agent3.set_input_constraints(Polytope.from_hyperplanes(A3_input, b3_input))
    
    # Add connections in the network
    # Dynamics connections
    network.add_dynamics_connection(3, 1)  # Agent 3 influences Agent 1
    network.add_dynamics_connection(1, 2)  # Agent 1 influences Agent 2
    network.add_dynamics_connection(2, 3)  # Agent 2 influences Agent 3
    
    return network


def define_target_sets(network):
    """
    Define the target sets for each agent.
    Each target set is a polytope in the state space of the agent.
    """
    target_sets = {}
    
    # Agent 1: Target set is a box centered at (2, 1)
    A1 = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
    b1 = np.array([3, -1, 2, 0])  # Box: 1 ≤ x₁ ≤ 3, 0 ≤ x₂ ≤ 2
    target_sets[1] = Polytope.from_hyperplanes(A1, b1)
    
    # Agent 2: Target set is a box centered at (-1, 1)
    A2 = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
    b2 = np.array([0, 2, 2, 0])  # Box: -2 ≤ x₁ ≤ 0, 0 ≤ x₂ ≤ 2
    target_sets[2] = Polytope.from_hyperplanes(A2, b2)
    
    # Agent 3: Target set is a box centered at (0, -1)
    A3 = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
    b3 = np.array([1, 1, 0, 2])  # Box: -1 ≤ x₁ ≤ 1, -2 ≤ x₂ ≤ 0
    target_sets[3] = Polytope.from_hyperplanes(A3, b3)
    
    return target_sets


def visualize_results(network, target_sets, backward_reachable_sets):
    """
    Visualize the target sets and backward reachable sets for each agent.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, agent_id in enumerate(network.get_agent_ids()):
        ax = axes[i]
        
        # Get target set vertices
        target_set = target_sets[agent_id]
        target_vertices = target_set.get_generators()[0]
        
        # Get backward reachable set vertices
        br_set = backward_reachable_sets[agent_id]
        br_vertices = br_set.get_generators()[0]
        
        # Plot target set
        if target_vertices.shape[0] > 2:
            hull = ConvexHull(target_vertices[:, :2])
            target_polygon = Polygon(target_vertices[hull.vertices, :2], alpha=0.3, color='blue',
                                    label=f'Target Set (Agent {agent_id})')
            ax.add_patch(target_polygon)
        else:
            ax.plot(target_vertices[:, 0], target_vertices[:, 1], 'bo-', alpha=0.5,
                   label=f'Target Set (Agent {agent_id})')
        
        # Plot backward reachable set
        if br_vertices.shape[0] > 2:
            hull = ConvexHull(br_vertices[:, :2])
            br_polygon = Polygon(br_vertices[hull.vertices, :2], alpha=0.3, color='red',
                                label=f'Backward Reachable Set (Agent {agent_id})')
            ax.add_patch(br_polygon)
        else:
            ax.plot(br_vertices[:, 0], br_vertices[:, 1], 'ro-', alpha=0.5,
                   label=f'Backward Reachable Set (Agent {agent_id})')
        
        # Set axis properties
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel(f'x₁ (Agent {agent_id})')
        ax.set_ylabel(f'x₂ (Agent {agent_id})')
        ax.set_title(f'Agent {agent_id}')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('backward_reachability_results.png')
    plt.show()


def main():
    # Step 1: Create the network of linear affine systems
    print("Creating the network of linear affine systems...")
    network = create_linear_affine_network()
    
    # Step 2: Define the target sets
    print("Defining target sets...")
    target_sets = define_target_sets(network)
    
    # Step 3: Create axis sets for the reachability analysis
    print("Creating axis sets...")
    horizon = 3  # Time horizon H for backward reachability
    axis_sets = create_axis_sets(network, horizon)
    
    # Step 4: Run the distributed backward reachability algorithm
    print(f"Running distributed backward reachability (horizon={horizon})...")
    backward_reachable_sets, admissible_control_sequences = distributed_backward_reachability(
        network, target_sets, axis_sets, horizon)
    
    # Step 5: Visualize the results
    print("Visualizing results...")
    visualize_results(network, target_sets, backward_reachable_sets)
    
    print("Done!")
    print("The backward reachable sets represent the initial states from which")
    print(f"each agent can reach its target set in {horizon} time steps.")
    
    # Optionally, save the computed sets for later use
    # save_results(backward_reachable_sets, admissible_control_sequences)


if __name__ == "__main__":
    main()