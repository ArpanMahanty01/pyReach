# agent.py

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union, Any


class Agent:
    """
    Represents an agent in a networked system for distributed reachability analysis.
    
    This class can represent different types of systems, including linear affine systems
    as described in the paper.
    """
    
    def __init__(self, 
                 agent_id: int,
                 state_dim: int,
                 input_dim: int,
                 system_type: str = 'linear_affine',
                 disturbance_dim: int = 0):
        """
        Initialize an agent in the networked system.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the agent's state
            input_dim: Dimension of the agent's input
            system_type: Type of system ('linear_affine', 'nonlinear', etc.)
            disturbance_dim: Dimension of the agent's disturbance (if any)
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.system_type = system_type
        self.disturbance_dim = disturbance_dim
        
        # Initialize system-specific parameters
        if system_type == 'linear_affine':
            # Dynamics matrices for linear affine systems
            self.A_matrices: Dict[int, np.ndarray] = {}  # A_ij matrices for each neighbor j
            self.B_matrices: Dict[int, np.ndarray] = {}  # B_ij matrices for each neighbor j
            self.K = np.zeros((state_dim, 1))            # Affine term
            self.E = np.zeros((state_dim, disturbance_dim)) if disturbance_dim > 0 else None
        else:
            #! handle these cases
            pass
        
        # Constraints
        self.state_constraints = None  # Polytope to represent X_i
        self.input_constraints = None  # Polytope to represent U_i
        self.disturbance_set = None    # Polytope to represent D_i
        self.joint_constraints = []    # List of joint constraints with other agents
    
    def set_linear_dynamics(self, 
                           neighbor_id: int, 
                           A_matrix: np.ndarray, 
                           B_matrix: np.ndarray):
        """
        Set the linear dynamics matrices for a neighbor.
        
        This is used for linear affine systems to set the A_ij and B_ij matrices
        as described in eq. (20) of the paper.
        
        Args:
            neighbor_id: ID of the neighboring agent
            A_matrix: State transition matrix A_ij
            B_matrix: Input matrix B_ij
        """
        if self.system_type != 'linear_affine':
            raise ValueError("This method is only for linear affine systems")
        
        self.A_matrices[neighbor_id] = A_matrix
        self.B_matrices[neighbor_id] = B_matrix
    
    def set_affine_term(self, K: np.ndarray):
        """
        Set the affine term K_i in the dynamics.
        
        Args:
            K: Affine term vector
        """
        if self.system_type != 'linear_affine':
            raise ValueError("This method is only for linear affine systems")
        
        self.K = K
    
    def set_disturbance_matrix(self, E: np.ndarray):
        """
        Set the disturbance matrix E_i in the dynamics.
        
        Args:
            E: Disturbance matrix
        """
        if self.system_type != 'linear_affine':
            raise ValueError("This method is only for linear affine systems")
        
        if self.disturbance_dim == 0:
            raise ValueError("Agent has no disturbance dimension")
        
        self.E = E
    
    def set_dynamics_function(self, dynamics_function):
        """
        Set a custom dynamics function for non-linear systems.
        
        Args:
            dynamics_function: Function that defines the dynamics
        """
        if self.system_type == 'linear_affine':
            raise ValueError("For linear affine systems, use set_linear_dynamics instead")
        
        self.dynamics_function = dynamics_function
    
    def set_state_constraints(self, constraints):
        """
        Set the state constraints X_i.
        
        Args:
            constraints: Polytope representing the state constraints
        """
        self.state_constraints = constraints
    
    def set_input_constraints(self, constraints):
        """
        Set the input constraints U_i.
        
        Args:
            constraints: Polytope representing the input constraints
        """
        self.input_constraints = constraints
    
    def set_disturbance_set(self, disturbance_set):
        """
        Set the disturbance set D_i.
        
        Args:
            disturbance_set: Polytope representing the disturbance set
        """
        if self.disturbance_dim == 0:
            raise ValueError("Agent has no disturbance dimension")
        
        self.disturbance_set = disturbance_set
    
    def add_joint_constraint(self, constraint):
        """
        Add a joint constraint with other agents.
        
        Args:
            constraint: Constraint object representing a joint constraint
        """
        self.joint_constraints.append(constraint)
    
    def compute_next_state(self, 
                          current_state: np.ndarray, 
                          current_input: np.ndarray,
                          neighbor_states: Dict[int, np.ndarray] = None,
                          neighbor_inputs: Dict[int, np.ndarray] = None,
                          disturbance: np.ndarray = None) -> np.ndarray:
        """
        Compute the next state based on the agent's dynamics.
        
        For linear affine systems, this implements eq. (20) of the paper:
        x_i(t+1) = sum_j A_ij x_j(t) + sum_j B_ij u_j(t) + K_i + E_i d_i(t)
        
        Args:
            current_state: Current state of the agent
            current_input: Current input of the agent
            neighbor_states: Dictionary mapping neighbor IDs to their states
            neighbor_inputs: Dictionary mapping neighbor IDs to their inputs
            disturbance: Disturbance input (if any)
            
        Returns:
            Next state of the agent
        """
        if self.system_type == 'linear_affine':
            if neighbor_states is None:
                neighbor_states = {}
            if neighbor_inputs is None:
                neighbor_inputs = {}
            
            # Initialize next state with the affine term
            next_state = self.K.copy().flatten()
            
            # Add contribution from this agent
            if self.agent_id in self.A_matrices:
                next_state += self.A_matrices[self.agent_id] @ current_state
            if self.agent_id in self.B_matrices:
                next_state += self.B_matrices[self.agent_id] @ current_input
            
            # Add contributions from neighbors
            for neighbor_id, neighbor_state in neighbor_states.items():
                if neighbor_id in self.A_matrices:
                    next_state += self.A_matrices[neighbor_id] @ neighbor_state
            
            for neighbor_id, neighbor_input in neighbor_inputs.items():
                if neighbor_id in self.B_matrices:
                    next_state += self.B_matrices[neighbor_id] @ neighbor_input
            
            # Add disturbance if present
            if disturbance is not None and self.E is not None:
                next_state += self.E @ disturbance
            
            return next_state
        else:
            # For non-linear systems, use the dynamics function
            if self.dynamics_function is None:
                raise ValueError("Dynamics function is not set")
            
            return self.dynamics_function(current_state, current_input, 
                                          neighbor_states, neighbor_inputs, 
                                          disturbance)
    
    def check_state_constraints(self, state: np.ndarray) -> bool:
        """
        Check if a state satisfies the agent's state constraints.
        
        Args:
            state: State to check
            
        Returns:
            True if the state satisfies the constraints, False otherwise
        """
        if self.state_constraints is None:
            # No constraints defined, so any state is valid
            return True
        
        return self.state_constraints.contains(state)
    
    def check_input_constraints(self, input_: np.ndarray) -> bool:
        """
        Check if an input satisfies the agent's input constraints.
        
        Args:
            input_: Input to check
            
        Returns:
            True if the input satisfies the constraints, False otherwise
        """
        if self.input_constraints is None:
            # No constraints defined, so any input is valid
            return True
        
        return self.input_constraints.contains(input_)
    
    def check_joint_constraints(self, 
                               state: np.ndarray, 
                               input_: np.ndarray,
                               neighbor_states: Dict[int, np.ndarray] = None,
                               neighbor_inputs: Dict[int, np.ndarray] = None) -> bool:
        """
        Check if a state-input pair satisfies the agent's joint constraints with neighbors.
        
        Args:
            state: State to check
            input_: Input to check
            neighbor_states: Dictionary mapping neighbor IDs to their states
            neighbor_inputs: Dictionary mapping neighbor IDs to their inputs
            
        Returns:
            True if the state-input pair satisfies all joint constraints, False otherwise
        """
        if not self.joint_constraints:
            # No joint constraints defined
            return True
        
        # Check each joint constraint
        for constraint in self.joint_constraints:
            if not constraint.is_satisfied(state, input_, neighbor_states, neighbor_inputs):
                return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"Agent {self.agent_id} ({self.system_type}): state_dim={self.state_dim}, input_dim={self.input_dim}"
    
    def __repr__(self) -> str:
        """Formal string representation of the agent."""
        return self.__str__()