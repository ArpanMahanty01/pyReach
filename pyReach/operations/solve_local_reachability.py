# reachability.py

import numpy as np
from typing import Dict, List, Set, Tuple, Any
from scipy.optimize import linprog

from pyReach.core.AxisSet import AxisSet
from pyReach.core.Polytope import Polytope
from pyReach.core.Network import Network

def solve_local_reachability(network: Network, 
                            agent_id: int, 
                            target_set: Polytope, 
                            axis_sets: Dict[str, AxisSet], 
                            horizon: int) -> Polytope:
    """
    Solve the local reachability problem for an agent as defined in eq. (15) of the paper.
    
    This function computes S_kh,i by solving a linear program to find the set of all
    initial states and input sequences that drive the agent to the target set.
    
    Args:
        network: Network of agents
        agent_id: ID of the agent to solve for
        target_set: Target set S_h,i for the agent
        axis_sets: Dictionary of axis sets created for the network
        horizon: Time horizon H
        
    Returns:
        Polytope representing the solution of the local reachability problem S_kh,i
    """
    agent = network.get_agent(agent_id)
    
    # Only implemented for linear affine systems for now
    if agent.system_type != 'linear_affine':
        raise ValueError("Only linear affine systems are supported")
    
    # Get the neighborhood of the agent
    neighbors = network.get_communication_neighbors(agent_id)
    
    # Step 1: Build dynamics constraints F_i^i z_i = f_i
    F_i, f_i = _build_dynamics_constraints(network, agent_id, neighbors, horizon)
    
    # Step 2: Build static constraints G_i^i z_i ≤ g_i
    G_i, g_i = _build_static_constraints(network, agent_id, neighbors, target_set, horizon)
    
    # Step 3: Account for disturbances if present
    if agent.disturbance_dim > 0:
        delta_i = _compute_delta_vector(network, agent_id, G_i)
        g_i = g_i - delta_i
    
    # Solve the LP to find the feasible set
    return _solve_lp_feasible_set(F_i, f_i, G_i, g_i, network, agent_id, axis_sets, horizon)


def _build_dynamics_constraints(network: Network, 
                               agent_id: int, 
                               neighbors: Set[int], 
                               horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the dynamics constraints for the local reachability problem.
    
    This constructs the matrix F_i^i and vector f_i in eq. (22) of the paper.
    
    Args:
        network: Network of agents
        agent_id: ID of the agent
        neighbors: Set of neighbor agent IDs
        horizon: Time horizon H
        
    Returns:
        Tuple containing F_i^i and f_i
    """
    agent = network.get_agent(agent_id)
    
    # Get dimensions
    state_dim = agent.state_dim
    n_i = sum(network.get_agent(j).state_dim for j in neighbors)  # Total state dimension of neighborhood
    m_i = sum(network.get_agent(j).input_dim for j in neighbors)  # Total input dimension of neighborhood
    
    # Initialize matrices
    F_i = np.zeros((state_dim * horizon, (horizon + 1) * (n_i + m_i)))
    f_i = np.zeros(state_dim * horizon)
    
    # For each time step t=0,1,...,H-1
    for t in range(horizon):
        # Get the axis indices for state at t+1
        row_start = t * state_dim
        
        # For each neighbor j
        col_offset_state = 0
        col_offset_input = 0
        
        for j in sorted(neighbors):  # Sort to ensure consistent ordering
            neighbor = network.get_agent(j)
            
            # State at time t
            if j in agent.A_matrices:
                A_ij = agent.A_matrices[j]
                col_indices = t * (n_i + m_i) + col_offset_state
                for r in range(state_dim):
                    for c in range(neighbor.state_dim):
                        F_i[row_start + r, col_indices + c] = -A_ij[r, c]
            
            col_offset_state += neighbor.state_dim
            
            # Input at time t
            if j in agent.B_matrices:
                B_ij = agent.B_matrices[j]
                col_indices = t * (n_i + m_i) + n_i + col_offset_input
                for r in range(state_dim):
                    for c in range(neighbor.input_dim):
                        F_i[row_start + r, col_indices + c] = -B_ij[r, c]
            
            col_offset_input += neighbor.input_dim
        
        # State at time t+1
        col_offset_state = 0
        for j in sorted(neighbors):
            neighbor = network.get_agent(j)
            if j == agent_id:
                col_indices = (t + 1) * (n_i + m_i) + col_offset_state
                for r in range(state_dim):
                    F_i[row_start + r, col_indices + r] = 1
            col_offset_state += neighbor.state_dim
        
        # Affine term K_i
        f_i[row_start:row_start + state_dim] = agent.K.flatten()
    
    return F_i, f_i


def _build_static_constraints(network: Network, 
                             agent_id: int, 
                             neighbors: Set[int], 
                             target_set: Polytope, 
                             horizon: int) -> Tuple[np.ndarray, List[float]]:
    """
    Build the static constraints for the local reachability problem.
    
    This constructs the matrix G_i^i and vector g_i in eq. (23) of the paper.
    
    Args:
        network: Network of agents
        agent_id: ID of the agent
        neighbors: Set of neighbor agent IDs
        target_set: Target set S_h,i for the agent
        horizon: Time horizon H
        
    Returns:
        Tuple containing G_i^i and g_i
    """
    # Get dimensions
    n_i = sum(network.get_agent(j).state_dim for j in neighbors)  # Total state dimension of neighborhood
    m_i = sum(network.get_agent(j).input_dim for j in neighbors)  # Total input dimension of neighborhood
    
    # Collect all constraints
    constraints_list = []
    
    # 1. State constraints for each time step t=0,1,...,H
    for t in range(horizon + 1):
        for j in sorted(neighbors):  # Sort to ensure consistent ordering
            neighbor = network.get_agent(j)
            
            # Skip if no state constraints defined
            if neighbor.state_constraints is None:
                continue
                
            # Get constraint matrices
            A_state, b_state = neighbor.state_constraints.get_constraints()
            
            # Number of constraints
            n_constraints = A_state.shape[0]
            
            # Create a constraint matrix for this time step and neighbor
            G_block = np.zeros((n_constraints, (horizon + 1) * (n_i + m_i)))
            
            # Find the column indices for this neighbor's state at time t
            col_offset = 0
            for k in sorted(neighbors):
                if k == j:
                    break
                col_offset += network.get_agent(k).state_dim
            
            col_start = t * (n_i + m_i) + col_offset
            
            # Fill in the constraint matrix
            for r in range(n_constraints):
                for c in range(neighbor.state_dim):
                    G_block[r, col_start + c] = A_state[r, c]
            
            constraints_list.append((G_block, b_state))
    
    # 2. Input constraints for each time step t=0,1,...,H-1
    for t in range(horizon):
        for j in sorted(neighbors):  # Sort to ensure consistent ordering
            neighbor = network.get_agent(j)
            
            # Skip if no input constraints defined
            if neighbor.input_constraints is None:
                continue
                
            # Get constraint matrices
            A_input, b_input = neighbor.input_constraints.get_constraints()
            
            # Number of constraints
            n_constraints = A_input.shape[0]
            
            # Create a constraint matrix for this time step and neighbor
            G_block = np.zeros((n_constraints, (horizon + 1) * (n_i + m_i)))
            
            # Find the column indices for this neighbor's input at time t
            col_offset = 0
            for k in sorted(neighbors):
                if k == j:
                    break
                col_offset += network.get_agent(k).input_dim
            
            col_start = t * (n_i + m_i) + n_i + col_offset
            
            # Fill in the constraint matrix
            for r in range(n_constraints):
                for c in range(neighbor.input_dim):
                    G_block[r, col_start + c] = A_input[r, c]
            
            constraints_list.append((G_block, b_input))
    
    # 3. Target set constraints at time H
    if target_set is not None:
        A_target, b_target = target_set.get_constraints()
        n_constraints = A_target.shape[0]
        
        G_block = np.zeros((n_constraints, (horizon + 1) * (n_i + m_i)))
        
        # Find the column indices for agent's state at time H
        col_offset = 0
        for k in sorted(neighbors):
            if k == agent_id:
                break
            col_offset += network.get_agent(k).state_dim
        
        col_start = horizon * (n_i + m_i) + col_offset
        
        # Fill in the constraint matrix
        for r in range(n_constraints):
            for c in range(network.get_agent(agent_id).state_dim):
                G_block[r, col_start + c] = A_target[r, c]
        
        constraints_list.append((G_block, b_target))
    
    # 4. Joint constraints (if any)
    for j in sorted(neighbors):
        agent = network.get_agent(j)
        for joint_constraint in agent.joint_constraints:
            # This would depend on the specific implementation of joint constraints
            # For now, assume joint_constraint provides a method to get constraints
            # in the form of A_joint and b_joint
            A_joint, b_joint = joint_constraint.get_constraints()
            # Add to constraints_list similar to above
    
    # Combine all constraints
    if not constraints_list:
        # No constraints
        return np.zeros((0, (horizon + 1) * (n_i + m_i))), np.zeros(0)
    
    total_constraints = sum(A.shape[0] for A, _ in constraints_list)
    G_i = np.zeros((total_constraints, (horizon + 1) * (n_i + m_i)))
    g_i = np.zeros(total_constraints)
    
    row_offset = 0
    for G_block, b_block in constraints_list:
        n_rows = G_block.shape[0]
        G_i[row_offset:row_offset + n_rows, :] = G_block
        g_i[row_offset:row_offset + n_rows] = b_block
        row_offset += n_rows
    
    return G_i, g_i


def _compute_delta_vector(network: Network, agent_id: int, G_i: np.ndarray, horizon: int) -> np.ndarray:
    """
    Compute the delta vector to account for disturbances as in eq. (26) of the paper.
    
    This function implements the robust approach for handling disturbances by computing
    the maximum effect of disturbances on the constraints.
    
    Args:
        network: Network of agents
        agent_id: ID of the agent
        G_i: Constraint matrix
        horizon: Time horizon H
        
    Returns:
        Delta vector for robust constraints
    """
    agent = network.get_agent(agent_id)
    
    if agent.disturbance_dim == 0 or agent.disturbance_set is None:
        # No disturbance or no disturbance set defined
        return np.zeros(G_i.shape[0])
    
    # Get the number of constraints
    n_constraints = G_i.shape[0]
    
    # Initialize delta vector
    delta_i = np.zeros(n_constraints)
    
    # Get disturbance set
    disturbance_set = agent.disturbance_set
    
    # Get neighbors
    neighbors = network.get_communication_neighbors(agent_id)
    
    # For each constraint row in G_i, solve the maximization problem
    for j in range(n_constraints):
        # Get the j-th row of G_i as the vector ν_j
        nu_j = G_i[j, :]
        
        # Compute the maximum effect of disturbances on this constraint
        delta_i[j] = _maximize_disturbance_effect(network, agent_id, nu_j, neighbors, disturbance_set, horizon)
    
    return delta_i


def _maximize_disturbance_effect(network: Network, 
                                agent_id: int, 
                                nu_j: np.ndarray, 
                                neighbors: Set[int], 
                                disturbance_set: Polytope, 
                                horizon: int) -> float:
    """
    Maximize the effect of disturbances on a constraint as in eq. (26) of the paper.
    
    This function solves the maximization problem:
    δ_i^(j) := max_{d∈D} ν_j^T G_i^i L_i d
    
    Args:
        network: Network of agents
        agent_id: ID of the agent
        nu_j: Vector ν_j (j-th row of G_i)
        neighbors: Set of neighbor agent IDs
        disturbance_set: Polytope representing the disturbance set D_i
        horizon: Time horizon H
        
    Returns:
        Maximum value of the objective function
    """
    agent = network.get_agent(agent_id)
    
    # Get dimensions
    n_i = sum(network.get_agent(j).state_dim for j in neighbors)  # Total state dimension of neighborhood
    m_i = sum(network.get_agent(j).input_dim for j in neighbors)  # Total input dimension of neighborhood
    
    # First, compute the matrix L_i as described in Section 6 of the paper
    # This matrix maps disturbances to their effect on states
    L_i = _compute_disturbance_mapping(network, agent_id, neighbors, horizon)
    
    # Compute the objective function coefficient vector: ν_j^T G_i^i L_i
    # This represents how disturbances affect the j-th constraint
    objective = nu_j @ L_i
    
    # If the objective is zero, the constraint is not affected by disturbances
    if np.allclose(objective, 0):
        return 0.0
    
    # Get the vertices of the disturbance set
    # We're assuming disturbance_set is a polytope with a V-representation
    vertices = disturbance_set.get_generators()[0]  # Assuming get_generators returns (vertices, rays)
    
    # For a convex polytope, the maximum of a linear function is attained at a vertex
    # Evaluate the objective at each vertex and take the maximum
    max_value = float('-inf')
    for vertex in vertices:
        value = objective @ vertex
        max_value = max(max_value, value)
    
    # If the disturbance set is unbounded, we would need to check rays as well
    # For simplicity, we're assuming a bounded disturbance set
    
    return max_value


def _compute_disturbance_mapping(network: Network, 
                                agent_id: int, 
                                neighbors: Set[int], 
                                horizon: int) -> np.ndarray:
    """
    Compute the matrix L_i that maps disturbances to their effect on states.
    
    This implements the disturbance mapping as described in eq. (24) of the paper.
    
    Args:
        network: Network of agents
        agent_id: ID of the agent
        neighbors: Set of neighbor agent IDs
        horizon: Time horizon H
        
    Returns:
        Matrix mapping disturbances to states
    """
    agent = network.get_agent(agent_id)
    
    # Get dimensions
    state_dim = agent.state_dim
    disturbance_dim = agent.disturbance_dim
    n_i = sum(network.get_agent(j).state_dim for j in neighbors)  # Total state dimension of neighborhood
    m_i = sum(network.get_agent(j).input_dim for j in neighbors)  # Total input dimension of neighborhood
    
    # Initialize the mapping matrix
    # This maps disturbances over the horizon to their effect on states and inputs
    # Size is [(horizon+1)*(n_i+m_i)] × [horizon*disturbance_dim]
    L_i = np.zeros(((horizon + 1) * (n_i + m_i), horizon * disturbance_dim))
    
    # For each time step t in the horizon
    for t in range(1, horizon + 1):  # Start from t=1 since disturbances affect future states
        # Compute the effect of past disturbances on the current state
        for tau in range(t - 1):  # Past time steps
            # Compute the matrix A^(t-tau-2)
            A_power = _compute_A_power(network, agent_id, neighbors, t - tau - 2)
            
            # Compute the effect of disturbance at time tau on state at time t
            # From eq. (24): χ_j(t) = L_j ∑_{τ=0}^{t-2} A^{t-τ-2} E d(τ)
            
            # Find the indices for the state of agent_id at time t in L_i
            col_offset_state = 0
            for j in sorted(neighbors):
                if j == agent_id:
                    break
                col_offset_state += network.get_agent(j).state_dim
            
            row_start = t * (n_i + m_i) + col_offset_state
            row_end = row_start + state_dim
            
            # Find the indices for the disturbance at time tau in L_i
            col_start = tau * disturbance_dim
            col_end = col_start + disturbance_dim
            
            # Set the mapping: L_i[state indices, disturbance indices] = A^(t-tau-2) * E
            L_i[row_start:row_end, col_start:col_end] = A_power @ agent.E
    
    return L_i


def _compute_A_power(network: Network, 
                    agent_id: int, 
                    neighbors: Set[int], 
                    power: int) -> np.ndarray:
    """
    Compute the power of the state transition matrix A.
    
    Args:
        network: Network of agents
        agent_id: ID of the agent
        neighbors: Set of neighbor agent IDs
        power: Power to raise A to
        
    Returns:
        A^power
    """
    agent = network.get_agent(agent_id)
    
    # Get dimensions
    state_dim = agent.state_dim
    
    # Get the state transition matrix for the agent itself
    A = agent.A_matrices.get(agent_id, np.zeros((state_dim, state_dim)))
    
    # Compute the power
    if power == 0:
        return np.eye(state_dim)
    elif power < 0:
        return np.zeros((state_dim, state_dim))
    else:
        result = A.copy()
        for _ in range(power - 1):
            result = result @ A
        return result


def _solve_lp_feasible_set(F_i: np.ndarray, 
                          f_i: np.ndarray, 
                          G_i: np.ndarray, 
                          g_i: np.ndarray, 
                          network: Network, 
                          agent_id: int, 
                          axis_sets: Dict[str, AxisSet], 
                          horizon: int) -> Polytope:
    """
    Solve the linear program to find the feasible set for the local reachability problem.
    
    Args:
        F_i: Dynamics constraint matrix
        f_i: Dynamics constraint vector
        G_i: Static constraint matrix
        g_i: Static constraint vector
        network: Network of agents
        agent_id: ID of the agent
        axis_sets: Dictionary of axis sets
        horizon: Time horizon
        
    Returns:
        Polytope representing the solution set
    """
    # The LP problem is to find the feasible set defined by:
    # F_i z_i = f_i
    # G_i z_i ≤ g_i
    
    # This is a set-based problem, not a optimization problem
    # We need to compute the H-representation of the feasible set
    
    # For equality constraints F_i z_i = f_i, we convert to inequality constraints:
    # F_i z_i ≤ f_i
    # -F_i z_i ≤ -f_i
    
    # Combine with the existing inequality constraints
    if F_i.shape[0] > 0:
        G_combined = np.vstack([G_i, F_i, -F_i])
        g_combined = np.hstack([g_i, f_i, -f_i])
    else:
        G_combined = G_i
        g_combined = g_i
    
    # Create a polytope from the constraints
    # This is the solution set S_kh,i
    return Polytope.from_hyperplanes(G_combined, g_combined)