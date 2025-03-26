import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union


class LinearAffineSystem:
    def __init__(self, id: int, A_ij: Dict[int, np.ndarray], B_ij: Dict[int, np.ndarray], 
                 K_i: np.ndarray):
        self.id = id
        self.A_ij = A_ij
        self.B_ij = B_ij
        self.K_i = K_i
        
        # Determine dimensions from the matrices
        self.state_dim = A_ij[id].shape[0]
        self.input_dim = B_ij[id].shape[1]
        
        # Store the A_ii and B_ii matrices for convenience
        self.A_ii = A_ij[id]
        self.B_ii = B_ij[id]
        
        # Identify neighbors based on dynamics
        self.neighbors = set()
        for j in A_ij:
            if j != id and not np.allclose(A_ij[j], 0):
                self.neighbors.add(j)
        for j in B_ij:
            if j != id and not np.allclose(B_ij[j], 0):
                self.neighbors.add(j)
    
    def nextState(self, x: Dict[int, np.ndarray], u: Dict[int, np.ndarray]) -> np.ndarray:
        """Compute the next state for this system given current states and inputs of all systems"""
        next_x = np.copy(self.K_i)
        
        # Add contribution from all states
        for j, A_matrix in self.A_ij.items():
            if j in x:
                next_x += A_matrix @ x[j]
        
        # Add contribution from all inputs
        for j, B_matrix in self.B_ij.items():
            if j in u:
                next_x += B_matrix @ u[j]
        
        return next_x
    
    def generateConstraintMatrices(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constraint matrices for the linear program representing system dynamics"""
        # For a horizon of H steps, the decision variables are:
        # [x_i(0), x_i(1), ..., x_i(H), u_i(0), u_i(1), ..., u_i(H-1)]
        
        total_vars = self.state_dim * (horizon + 1) + self.input_dim * horizon
        
        # Each time step introduces state_dim equality constraints
        num_constraints = self.state_dim * horizon
        
        A = np.zeros((num_constraints, total_vars))
        b = np.zeros(num_constraints)
        
        for t in range(horizon):
            # Indices for variables at time t and t+1
            x_t_indices = slice(t * self.state_dim, (t + 1) * self.state_dim)
            x_t1_indices = slice((t + 1) * self.state_dim, (t + 2) * self.state_dim)
            u_t_indices = slice(
                (horizon + 1) * self.state_dim + t * self.input_dim,
                (horizon + 1) * self.state_dim + (t + 1) * self.input_dim
            )
            
            # Constraint indices for time step t
            c_indices = slice(t * self.state_dim, (t + 1) * self.state_dim)
            
            # Dynamics: x(t+1) = A*x(t) + B*u(t) + K
            # In standard form: -x(t+1) + A*x(t) + B*u(t) + K = 0
            
            # Coefficient for x(t+1): -I
            A[c_indices, x_t1_indices] = -np.eye(self.state_dim)
            
            # Coefficient for x(t): A
            A[c_indices, x_t_indices] = self.A_ii
            
            # Coefficient for u(t): B
            A[c_indices, u_t_indices] = self.B_ii
            
            # Right-hand side: -K
            b[c_indices] = -self.K_i
        
        return A, b
    
    def generateConstraintsForNeighbors(self, neighbors: Set[int], horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constraints including neighbor systems"""
        # Calculate total number of variables
        state_dims = {self.id: self.state_dim}
        input_dims = {self.id: self.input_dim}
        
        for j in neighbors:
            # In practice, these dimensions would be known or communicated
            # Here we'll infer them from the matrices
            state_dims[j] = self.A_ij[j].shape[1]
            input_dims[j] = self.B_ij[j].shape[1] if j in self.B_ij else 0
        
        # Total variables for each system across the horizon
        total_vars_per_system = {}
        for j in list(neighbors) + [self.id]:
            total_vars_per_system[j] = state_dims[j] * (horizon + 1) + input_dims[j] * horizon
        
        # Total variables across all systems
        total_vars = sum(total_vars_per_system.values())
        
        # Constraints for this system's dynamics
        num_constraints = self.state_dim * horizon
        
        A = np.zeros((num_constraints, total_vars))
        b = np.zeros(num_constraints)
        
        # Create a mapping of variable indices for each system
        var_indices = {}
        offset = 0
        for j in list(neighbors) + [self.id]:
            var_indices[j] = offset
            offset += total_vars_per_system[j]
        
        for t in range(horizon):
            # Constraint indices for time step t
            c_indices = slice(t * self.state_dim, (t + 1) * self.state_dim)
            
            # Indices for this system's variables at time t and t+1
            self_offset = var_indices[self.id]
            x_t_indices = slice(
                self_offset + t * self.state_dim, 
                self_offset + (t + 1) * self.state_dim
            )
            x_t1_indices = slice(
                self_offset + (t + 1) * self.state_dim, 
                self_offset + (t + 2) * self.state_dim
            )
            u_t_indices = slice(
                self_offset + (horizon + 1) * self.state_dim + t * self.input_dim,
                self_offset + (horizon + 1) * self.state_dim + (t + 1) * self.input_dim
            )
            
            # Coefficient for x_i(t+1): -I
            A[c_indices, x_t1_indices] = -np.eye(self.state_dim)
            
            # Coefficient for x_i(t): A_ii
            A[c_indices, x_t_indices] = self.A_ii
            
            # Coefficient for u_i(t): B_ii
            A[c_indices, u_t_indices] = self.B_ii
            
            # Add neighbor contributions
            for j in neighbors:
                neighbor_offset = var_indices[j]
                
                # Indices for neighbor's state at time t
                xj_t_indices = slice(
                    neighbor_offset + t * state_dims[j],
                    neighbor_offset + (t + 1) * state_dims[j]
                )
                
                # Coefficient for x_j(t): A_ij
                if j in self.A_ij and not np.allclose(self.A_ij[j], 0):
                    A[c_indices, xj_t_indices] = self.A_ij[j]
                
                # Indices for neighbor's input at time t
                if input_dims[j] > 0:
                    uj_t_indices = slice(
                        neighbor_offset + (horizon + 1) * state_dims[j] + t * input_dims[j],
                        neighbor_offset + (horizon + 1) * state_dims[j] + (t + 1) * input_dims[j]
                    )
                    
                    # Coefficient for u_j(t): B_ij
                    if j in self.B_ij and not np.allclose(self.B_ij[j], 0):
                        A[c_indices, uj_t_indices] = self.B_ij[j]
            
            # Right-hand side: -K_i
            b[c_indices] = -self.K_i
        
        return A, b
    
    def __str__(self) -> str:
        return f"LinearAffineSystem(id={self.id}, state_dim={self.state_dim}, input_dim={self.input_dim}, neighbors={self.neighbors})"
    
    def __repr__(self) -> str:
        return str(self)