# algorithm2.py

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from pyReach.core.AxisSet import AxisSet
from pyReach.core.Polytope import Polytope
from pyReach.core.Network import Network
from pyReach.operations.solve_local_reachability import solve_local_reachability


def distributed_backward_reachability(network: Network, 
                                     target_sets: Dict[int, Polytope], 
                                     axis_sets: Dict[str, AxisSet],
                                     horizon: int,
                                     max_iterations: int = 100,
                                     convergence_tol: float = 1e-6) -> Tuple[Dict[int, Polytope], Dict[int, Polytope]]:
    """
    Implementation of Algorithm 2 from the paper: Distributed computation of backward reachable set.
    
    This algorithm computes the backwards reachable set and admissible control sequences
    for each agent in a distributed manner, without requiring any centralized operations.
    
    Args:
        network: Network of agents
        target_sets: Dictionary mapping agent IDs to their target sets S_h,i
        axis_sets: Dictionary of axis sets
        horizon: Time horizon H
        max_iterations: Maximum number of iterations for the distributed algorithm
        convergence_tol: Tolerance for convergence detection
        
    Returns:
        Tuple of dictionaries mapping agent IDs to their backward reachable sets S_k,i
        and admissible control sequences Φ_kh,i
    """
    # Step 1: Solve the local reachability problem for each agent (Line 1-2 in Algorithm 2)
    local_solutions = {}
    print("Solving local reachability problems...")
    for agent_id in network.get_agent_ids():
        print(f"  Agent {agent_id}...")
        S_kh_i = solve_local_reachability(network, agent_id, target_sets[agent_id], axis_sets, horizon)
        local_solutions[agent_id] = S_kh_i
    
    # Step 2: Initialize for the distributed computation (Line 3-5 in Algorithm 2)
    S_kh_i_0 = local_solutions.copy()
    S_kh_i_current = {agent_id: poly for agent_id, poly in S_kh_i_0.items()}
    
    # Exchange initial solutions with neighbors (Line 4-5 in Algorithm 2)
    neighbor_solutions = {agent_id: {} for agent_id in network.get_agent_ids()}
    for agent_id in network.get_agent_ids():
        neighbors = network.get_communication_neighbors(agent_id)
        for neighbor_id in neighbors:
            if neighbor_id != agent_id:
                neighbor_solutions[agent_id][neighbor_id] = S_kh_i_0[neighbor_id]
    
    # Step 3: Run the distributed algorithm (Line 6 in Algorithm 2)
    print("Running distributed algorithm...")
    iteration = 0
    all_converged = False
    
    while not all_converged and iteration < max_iterations:
        iteration += 1
        print(f"  Iteration {iteration}...")
        
        # Store previous solutions for convergence check
        S_kh_i_previous = {agent_id: poly for agent_id, poly in S_kh_i_current.items()}
        
        # Update each agent's solution based on neighbors' information
        for agent_id in network.get_agent_ids():
            # Get the communication neighbors
            neighbors = network.get_communication_neighbors(agent_id)
            
            # Get the axis sets for the neighborhood
            B_H_M_i = axis_sets[f'B_H_M_{agent_id}']
            B_H_i = axis_sets[f'B_H_{agent_id}']
            
            # Collect the extruded sets from all neighbors including self
            extruded_sets = []
            for neighbor_id in neighbors:
                neighbor_solution = S_kh_i_current[neighbor_id]
                B_H_j = axis_sets[f'B_H_{neighbor_id}']
                
                # Extrude the neighbor's set to the neighborhood axis set (Line 12 in Algorithm 1)
                extruded_set = neighbor_solution.extrude(B_H_j, B_H_M_i)
                extruded_sets.append(extruded_set)
            
            # Intersect all extruded sets (Line 12 in Algorithm 1)
            intersection = Polytope.intersect_many(extruded_sets)
            
            # Project back to agent i's axis set (Line 12 in Algorithm 1)
            S_kh_i_updated = intersection.project(B_H_M_i, B_H_i)
            
            # Update the current solution
            S_kh_i_current[agent_id] = S_kh_i_updated
        
        # Exchange updated solutions with neighbors
        for agent_id in network.get_agent_ids():
            neighbors = network.get_communication_neighbors(agent_id)
            for neighbor_id in neighbors:
                if neighbor_id != agent_id:
                    neighbor_solutions[agent_id][neighbor_id] = S_kh_i_current[neighbor_id]
        
        # Check for convergence
        all_converged = True
        for agent_id in network.get_agent_ids():
            current = S_kh_i_current[agent_id]
            previous = S_kh_i_previous[agent_id]
            
            # Check if the sets are equal or have converged
            if not _check_set_convergence(current, previous, convergence_tol):
                all_converged = False
                break
    
    print(f"Distributed algorithm converged after {iteration} iterations.")
    
    # Step 4: Compute the backward reachable sets (Line 7 in Algorithm 2)
    backward_reachable_sets = {}
    for agent_id in network.get_agent_ids():
        B_H_i = axis_sets[f'B_H_{agent_id}']
        B_x_0_i = axis_sets[f'B_x_0_{agent_id}']
        
        # Project to get S_k,i (Line 7 in Algorithm 2)
        backward_reachable_sets[agent_id] = S_kh_i_current[agent_id].project(B_H_i, B_x_0_i)
    
    # Step 5: Compute the admissible control sequences (Line 8 in Algorithm 2)
    admissible_control_sequences = {}
    for agent_id in network.get_agent_ids():
        B_H_i = axis_sets[f'B_H_{agent_id}']
        B_x_0_i = axis_sets[f'B_x_0_{agent_id}']
        B_H_u_i = axis_sets[f'B_H_u_{agent_id}']
        
        # Create the axis set for initial state and control sequence
        combined_axes = B_x_0_i.union(B_H_u_i)
        
        # Project to get Φ_kh,i (Line 8 in Algorithm 2)
        admissible_control_sequences[agent_id] = S_kh_i_current[agent_id].project(B_H_i, combined_axes)
    
    return backward_reachable_sets, admissible_control_sequences


def _check_set_convergence(current: Polytope, previous: Polytope, tol: float) -> bool:
    """
    Check if two polytopes have converged to each other.
    
    Args:
        current: Current polytope
        previous: Previous polytope
        tol: Convergence tolerance
        
    Returns:
        True if the polytopes have converged, False otherwise
    """
    if current == previous:
        return True
    
    # Check if current is approximately a subset of previous
    # According to Lemma 1.ii from the paper, S_i(κ+1) ⊆ S_i(κ)
    try:
        # Get vertices of current
        vertices = current.get_generators()[0]
        
        # Get constraints of previous
        A_prev, b_prev = previous.get_constraints()
        
        # Check if all vertices of current satisfy all constraints of previous
        for vertex in vertices:
            if not np.all(A_prev @ vertex <= b_prev + tol):
                return False
        
        # Additionally, check if volumes are close enough
        current_vol = current.volume()
        prev_vol = previous.volume()
        
        if abs(current_vol - prev_vol) > tol * max(current_vol, prev_vol):
            return False
            
        return True
    
    except Exception as e:
        print(f"Warning in convergence check: {e}")
        return False