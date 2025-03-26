import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Set, Tuple, Optional, Union, Mapping

# from core import AxisSet, Polytope
from pyReach.core.AxisSet import AxisSet
from pyReach.core.Polytope import Polytope

from pyReach.dynamics import LinearAffineSystem


class Reachability:
    def __init__(self, system: LinearAffineSystem, state_constraints: Polytope, 
                 input_constraints: Polytope, target_set: Polytope, horizon: int):
        self.system = system
        self.state_constraints = state_constraints
        self.input_constraints = input_constraints
        self.target_set = target_set
        self.horizon = horizon
    
    def solve(self) -> Polytope:
        """Solve local backward reachability problem for a linear affine system"""
        # Get dimensions
        state_dim = self.system.state_dim
        input_dim = self.system.input_dim
        
        # Generate constraint matrices for the linear program
        A_dyn, b_dyn = self._generate_dynamics_constraints()
        A_state, b_state = self._generate_state_constraints()
        A_input, b_input = self._generate_input_constraints()
        A_target, b_target = self._generate_target_constraints()
        
        # Combine all constraints
        A = np.vstack([A_dyn, A_state, A_input, A_target])
        b = np.concatenate([b_dyn, b_state, b_input, b_target])
        
        # Create the full polytope representing all constraints
        full_polytope = Polytope(A, b)
        
        # Project onto the initial state space
        initial_vars = set(range(state_dim))
        backward_reachable_set = full_polytope.project(initial_vars)
        
        return backward_reachable_set
    
    def _generate_dynamics_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constraints from system dynamics"""
        state_dim = self.system.state_dim
        input_dim = self.system.input_dim
        total_vars = state_dim * (self.horizon + 1) + input_dim * self.horizon
        
        # For linear affine system: x(t+1) = A*x(t) + B*u(t) + K
        # This becomes: -x(t+1) + A*x(t) + B*u(t) + K = 0
        
        # Number of equality constraints: state_dim * horizon (one per state per time step)
        A_dyn = np.zeros((state_dim * self.horizon, total_vars))
        b_dyn = np.zeros(state_dim * self.horizon)
        
        for t in range(self.horizon):
            # For each time step, add constraints for next state
            # x(t+1) indices
            next_state_idx = np.arange(state_dim) + state_dim * (t + 1)
            # x(t) indices
            curr_state_idx = np.arange(state_dim) + state_dim * t
            # u(t) indices
            curr_input_idx = np.arange(input_dim) + state_dim * (self.horizon + 1) + input_dim * t
            
            # Row indices for the current time step constraints
            row_idx = np.arange(state_dim) + state_dim * t
            
            # -I for x(t+1)
            A_dyn[row_idx[:, np.newaxis], next_state_idx] = -np.eye(state_dim)
            
            # A matrix for x(t)
            A_dyn[row_idx[:, np.newaxis], curr_state_idx] = self.system.A_ii
            
            # B matrix for u(t)
            A_dyn[row_idx[:, np.newaxis], curr_input_idx] = self.system.B_ii
            
            # Add constant term K
            b_dyn[row_idx] = -self.system.K_i
        
        return A_dyn, b_dyn
    
    def _generate_state_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constraints for state bounds"""
        state_dim = self.system.state_dim
        input_dim = self.system.input_dim
        total_vars = state_dim * (self.horizon + 1) + input_dim * self.horizon
        
        # State constraints apply at each time step
        constraints_per_step = self.state_constraints.A.shape[0]
        total_constraints = constraints_per_step * (self.horizon + 1)
        
        A_state = np.zeros((total_constraints, total_vars))
        b_state = np.zeros(total_constraints)
        
        for t in range(self.horizon + 1):
            # State indices for time t
            state_idx = np.arange(state_dim) + state_dim * t
            
            # Constraint rows for time t
            row_idx = np.arange(constraints_per_step) + constraints_per_step * t
            
            # Copy the state constraints for this time step
            for i, idx in enumerate(row_idx):
                A_state[idx, state_idx] = self.state_constraints.A[i]
                b_state[idx] = self.state_constraints.b[i]
        
        return A_state, b_state
    
    def _generate_input_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constraints for input bounds"""
        state_dim = self.system.state_dim
        input_dim = self.system.input_dim
        total_vars = state_dim * (self.horizon + 1) + input_dim * self.horizon
        
        # Input constraints apply at each time step
        constraints_per_step = self.input_constraints.A.shape[0]
        total_constraints = constraints_per_step * self.horizon
        
        A_input = np.zeros((total_constraints, total_vars))
        b_input = np.zeros(total_constraints)
        
        for t in range(self.horizon):
            # Input indices for time t
            input_idx = np.arange(input_dim) + state_dim * (self.horizon + 1) + input_dim * t
            
            # Constraint rows for time t
            row_idx = np.arange(constraints_per_step) + constraints_per_step * t
            
            # Copy the input constraints for this time step
            for i, idx in enumerate(row_idx):
                A_input[idx, input_idx] = self.input_constraints.A[i]
                b_input[idx] = self.input_constraints.b[i]
        
        return A_input, b_input
    
    def _generate_target_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constraints for target set at final time step"""
        state_dim = self.system.state_dim
        input_dim = self.system.input_dim
        total_vars = state_dim * (self.horizon + 1) + input_dim * self.horizon
        
        # Target constraints only apply at final time step
        constraints = self.target_set.A.shape[0]
        
        A_target = np.zeros((constraints, total_vars))
        b_target = np.zeros(constraints)
        
        # Final state indices
        final_state_idx = np.arange(state_dim) + state_dim * self.horizon
        
        # Copy the target constraints
        for i in range(constraints):
            A_target[i, final_state_idx] = self.target_set.A[i]
            b_target[i] = self.target_set.b[i]
        
        return A_target, b_target
    
    def iterate(self, neighbor_sets: Dict[int, Polytope]) -> Dict[int, Polytope]:
        """Perform one iteration of the distributed algorithm"""
        # Assuming each agent has already computed its local reachable set
        # This function implements equation (17) from the paper for linear affine systems
        
        # Get the axes that define the neighborhood
        agent_id = self.system.id
        B_H_Mi = self._get_neighborhood_axes(neighbor_sets.keys())
        
        # Extrude each neighbor's set to the neighborhood axes
        extruded_sets = []
        for neighbor_id, neighbor_set in neighbor_sets.items():
            # Get the axes for this neighbor
            B_H_j = self._get_agent_axes(neighbor_id)
            
            # Extrude the neighbor's set to the neighborhood axes
            extruded_set = neighbor_set.extrude(B_H_j, len(B_H_Mi))
            extruded_sets.append(extruded_set)
        
        # Intersect all extruded sets
        if not extruded_sets:
            return {}
        
        intersection = extruded_sets[0]
        for ext_set in extruded_sets[1:]:
            intersection = intersection.intersection(ext_set)
        
        # Project back to each neighbor's axes
        result = {}
        for neighbor_id in neighbor_sets.keys():
            B_H_j = self._get_agent_axes(neighbor_id)
            result[neighbor_id] = intersection.project(B_H_j)
        
        return result
    
    def _get_agent_axes(self, agent_id: int) -> Set[int]:
        """Get the axis set for an agent"""
        # This is a simplified implementation - in practice, this would be
        # determined based on the system structure as defined in Section 3.3
        state_dim = self.system.state_dim
        input_dim = self.system.input_dim
        
        # For simplicity, assume consecutive indices for each agent
        state_indices = set(range(state_dim * (agent_id - 1), state_dim * agent_id))
        input_indices = set()
        
        return state_indices.union(input_indices)
    
    def _get_neighborhood_axes(self, neighbor_ids: Set[int]) -> Set[int]:
        """Get the combined axis set for a neighborhood"""
        axes = set()
        for neighbor_id in neighbor_ids:
            axes = axes.union(self._get_agent_axes(neighbor_id))
        return axes
    
    def checkConvergence(self, old_sets: Dict[int, Polytope], 
                         new_sets: Dict[int, Polytope], 
                         tolerance: float = 1e-6) -> bool:
        """Check if the algorithm has converged"""
        for agent_id in old_sets:
            if agent_id not in new_sets:
                return False
            
            old_set = old_sets[agent_id]
            new_set = new_sets[agent_id]
            
            # Check if new_set is a subset of old_set
            if not new_set.is_subset(old_set):
                return False
            
            # Check if the sets are nearly equal (within tolerance)
            old_volume = old_set.volume()
            new_volume = new_set.volume()
            
            if old_volume - new_volume > tolerance * old_volume:
                return False
        
        return True