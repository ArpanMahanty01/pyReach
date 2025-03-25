

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from pyReach.core.AxisSet import AxisSet
from pyReach.core.Polytope import Polytope
from pyReach.core.Network import Network


class DistributedExtrusion:
    """
    Implements Algorithm 1 from the paper: "Distributed extrusion generated set".
    
    This class handles the distributed computation of the projections of an extrusion
    generated set through iterative local operations and information exchange.
    """
    
    def __init__(self, network: Network, axis_sets: Dict[str, AxisSet], max_iterations: int = 100, 
                 convergence_tol: float = 1e-6):
        """
        Initialize the distributed algorithm.
        
        Args:
            network: Network of agents
            axis_sets: Dictionary mapping names to axis sets
            max_iterations: Maximum number of iterations to run
            convergence_tol: Tolerance for detecting convergence based on polytope volume
        """
        self.network = network
        self.axis_sets = axis_sets
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        
        # Store the sets S_i(κ) for each agent i at iteration κ
        self.agent_sets = {}
        
    def initialize(self, initial_sets: Dict[int, Polytope]):
        """
        Initialize the algorithm with starting sets S_i,0 for each agent.
        
        Args:
            initial_sets: Dictionary mapping agent IDs to their initial polytopes
        """
        self.agent_sets = {agent_id: {'current': poly, 'previous': None} 
                           for agent_id, poly in initial_sets.items()}
    
    def run(self) -> Dict[int, Polytope]:
        """
        Run the distributed algorithm until convergence or max iterations.
        
        Returns:
            Dictionary mapping agent IDs to their final polytope projections
        """
        iteration = 0
        converged = False
        
        while not converged and iteration < self.max_iterations:
            # Store the current sets for message passing
            current_sets = {agent_id: data['current'] 
                            for agent_id, data in self.agent_sets.items()}
            
            # Update each agent's set based on neighbors' information
            for agent_id in self.network.get_agent_ids():
                # Get the communication neighbors
                neighbors = self.network.get_communication_neighbors(agent_id)
                
                # Get the union of axis sets for the neighborhood
                B_M_i_key = f'B_H_M_{agent_id}'
                B_i_key = f'B_H_{agent_id}'
                
                B_M_i = self.axis_sets[B_M_i_key]
                B_i = self.axis_sets[B_i_key]
                
                # Collect the extruded sets from all neighbors
                extruded_sets = []
                for neighbor_id in neighbors:
                    neighbor_set = current_sets[neighbor_id]
                    neighbor_key = f'B_H_{neighbor_id}'
                    B_j = self.axis_sets[neighbor_key]
                    
                    # Extrude the neighbor's set to the neighborhood axis set
                    extruded_set = neighbor_set.extrude(B_j, B_M_i)
                    extruded_sets.append(extruded_set)
                
                # Intersect all extruded sets
                intersection = Polytope.intersect_many(extruded_sets)
                
                # Project back to agent i's axis set
                new_set = intersection.project(B_M_i, B_i)
                
                # Store the new set and the previous one for convergence check
                self.agent_sets[agent_id]['previous'] = self.agent_sets[agent_id]['current']
                self.agent_sets[agent_id]['current'] = new_set
            
            # Check for convergence
            converged = self._check_convergence()
            iteration += 1
            
            print(f"Completed iteration {iteration}, converged: {converged}")
        
        # Return the final sets
        return {agent_id: data['current'] for agent_id, data in self.agent_sets.items()}
    
    def _check_convergence(self) -> bool:
        """
        Check if the algorithm has converged.
        
        The algorithm has converged if for all agents, their set has not changed
        significantly from the previous iteration.
        
        Returns:
            True if converged, False otherwise
        """
        for agent_id, data in self.agent_sets.items():
            current = data['current']
            previous = data['previous']
            
            # If no previous set (first iteration), not converged
            if previous is None:
                return False
            
            # If the sets are equal, continue checking other agents
            if current == previous:
                continue
                
            # Check if the sets are approximately equal based on volume
            # This is a simple approach; more sophisticated methods could be used
            try:
                current_vol = current.volume()
                prev_vol = previous.volume()
                
                # If volumes are significantly different, not converged
                if abs(current_vol - prev_vol) > self.convergence_tol * max(current_vol, prev_vol):
                    return False
                
                # Alternative: check if the current set is contained in the previous set
                # This checks if S_i(κ+1) ⊆ S_i(κ) as per Lemma 1
                if not self._is_subset(current, previous):
                    return False
                
            except Exception as e:
                # If volume calculation fails, use a more direct comparison
                print(f"Warning: Volume calculation failed for agent {agent_id}: {e}")
                return False
        
        # If we get here, all agents have converged
        return True
    
    def _is_subset(self, set1: Polytope, set2: Polytope) -> bool:
        """
        Check if set1 is a subset of set2.
        
        Args:
            set1: First polytope
            set2: Second polytope
            
        Returns:
            True if set1 ⊆ set2, False otherwise
        """
        # For H-representation (Ax ≤ b), set1 ⊆ set2 if all constraints of set2
        # are satisfied by all vertices of set1
        
        # Get vertices of set1
        vertices = set1.get_generators()[0]
        
        # Get constraints of set2
        A2, b2 = set2.get_constraints()
        
        # Check if all vertices of set1 satisfy all constraints of set2
        for vertex in vertices:
            # Check if Ax ≤ b for all constraints
            if not np.all(A2 @ vertex <= b2 + self.convergence_tol):
                return False
        
        return True


def compute_distributed_reachability(network: Network, 
                                    target_sets: Dict[int, Polytope], 
                                    axis_sets: Dict[str, AxisSet],
                                    horizon: int) -> Dict[int, Polytope]:
    """
    Implement Algorithm 2 from the paper: the complete distributed backward reachability.
    
    This function orchestrates the entire distributed reachability computation:
    1. Solve local reachability problems for each agent
    2. Run the distributed algorithm to compute the projections
    3. Extract the backward reachable sets and admissible control sequences
    
    Args:
        network: Network of agents
        target_sets: Dictionary mapping agent IDs to their target sets
        axis_sets: Dictionary of axis sets
        horizon: Time horizon H
        
    Returns:
        Dictionary mapping agent IDs to their backward reachable sets
    """
    from pyReach.operations.solve_local_reachability import solve_local_reachability
    
    # Step 1: Solve the local reachability problem for each agent
    local_solutions = {}
    for agent_id in network.get_agent_ids():
        local_solution = solve_local_reachability(
            network, agent_id, target_sets[agent_id], axis_sets, horizon
        )
        local_solutions[agent_id] = local_solution
    
    # Step 2: Initialize and run the distributed algorithm
    distributed_alg = DistributedExtrusion(network, axis_sets)
    distributed_alg.initialize(local_solutions)
    final_solutions = distributed_alg.run()
    
    # Step 3: Extract the backward reachable sets for each agent
    backward_reachable_sets = {}
    for agent_id in network.get_agent_ids():
        # Get the axis sets for this agent
        B_H_i = axis_sets[f'B_H_{agent_id}']
        B_x_0_i = axis_sets[f'B_x_0_{agent_id}']
        
        # Extract the backward reachable set by projecting the solution
        backward_reachable_sets[agent_id] = final_solutions[agent_id].project(B_H_i, B_x_0_i)
    
    return backward_reachable_sets


def compute_admissible_control_sequences(network: Network,
                                         final_solutions: Dict[int, Polytope],
                                         axis_sets: Dict[str, AxisSet]) -> Dict[int, Polytope]:
    """
    Extract the admissible control sequences for each agent.
    
    This implements the extraction of Φ̄_kh,i as per Theorem 5 of the paper.
    
    Args:
        network: Network of agents
        final_solutions: Dictionary mapping agent IDs to their final solutions
        axis_sets: Dictionary of axis sets
        
    Returns:
        Dictionary mapping agent IDs to their admissible control sequences
    """
    admissible_controls = {}
    
    for agent_id in network.get_agent_ids():
        # Get the axis sets for this agent
        B_H_i = axis_sets[f'B_H_{agent_id}']
        B_x_0_i = axis_sets[f'B_x_0_{agent_id}']
        B_H_u_i = axis_sets[f'B_H_u_{agent_id}']
        
        # Create the combined axis set for initial state and input sequence
        combined_axes = B_x_0_i.union(B_H_u_i)
        
        # Extract the admissible control sequences by projecting the solution
        admissible_controls[agent_id] = final_solutions[agent_id].project(B_H_i, combined_axes)
    
    return admissible_controls