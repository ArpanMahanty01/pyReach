# polytope.py

import numpy as np
from typing import List, Tuple, Optional, Union
import polytope as pc 

from .AxisSet import AxisSet


class Polytope:
    """
    Extended polytope class for reachability analysis.
    
    This class extends the functionality of the base polytope package
    with additional operations needed for distributed reachability.
    """
    
    def __init__(self, polytope: pc.Polytope):
        """
        Initialize with a base polytope.
        
        Args:
            polytope: Base polytope object
        """
        self.polytope = polytope
    
    @classmethod
    def from_hyperplanes(cls, A: np.ndarray, b: np.ndarray) -> 'Polytope':
        """
        Create a polytope from hyperplane representation (Ax <= b).
        
        Args:
            A: Matrix of hyperplane normals
            b: Vector of hyperplane offsets
            
        Returns:
            Polytope instance
        """
        return cls(pc.Polytope(A, b))
    
    @classmethod
    def from_vertices(cls, vertices: np.ndarray) -> 'Polytope':
        """
        Create a polytope from vertex representation.
        
        Args:
            vertices: Array where each row is a vertex
            
        Returns:
            Polytope instance
        """
        # The polytope package may have different ways to construct from vertices
        # This is a common approach, but check the documentation of your chosen package
        return cls(pc.qhull(vertices))
    
    def project(self, from_axes: AxisSet, to_axes: AxisSet) -> 'Polytope':
        """
        Project the polytope from a higher-dimensional space to a lower-dimensional space.
        
        This implementation is optimized for linear affine systems and uses Fourier-Motzkin
        elimination for efficient projection computation.
        
        Args:
            from_axes: Original axis set
            to_axes: Target axis set (must be a subset of from_axes)
            
        Returns:
            Projected polytope
        """
        if not all(axis in from_axes.indices for axis in to_axes.indices):
            raise ValueError("Target axes must be a subset of original axes")
        
        # Get the H-representation (Ax <= b) of the polytope
        A, b = self.get_constraints()
        
        # If the polytope is empty, return an empty polytope
        if self.is_empty():
            return Polytope.from_hyperplanes(np.zeros((0, len(to_axes))), np.zeros(0))
        
        # Create mapping from original dimensions to projected dimensions
        projection_indices = []
        elimination_indices = []
        
        # Identify which indices to keep and which to eliminate
        for i, axis in enumerate(from_axes.indices):
            if axis in to_axes.indices:
                projection_indices.append(i)
            else:
                elimination_indices.append(i)
        
        # If no dimensions to eliminate, just extract the relevant coordinates
        if not elimination_indices:
            A_proj = A[:, projection_indices]
            return Polytope.from_hyperplanes(A_proj, b)
        
        # Apply Fourier-Motzkin elimination for each dimension to eliminate
        A_current, b_current = A, b
        
        for elim_idx in elimination_indices:
            # Adjust index to account for already eliminated dimensions
            adjusted_idx = elim_idx
            for prev_idx in elimination_indices:
                if prev_idx < elim_idx and prev_idx not in projection_indices:
                    adjusted_idx -= 1
            
            # Separate constraints into those with positive, negative, and zero coefficients
            # for the dimension being eliminated
            pos_rows = np.where(A_current[:, adjusted_idx] > 0)[0]
            neg_rows = np.where(A_current[:, adjusted_idx] < 0)[0]
            zero_rows = np.where(A_current[:, adjusted_idx] == 0)[0]
            
            # Initialize new arrays for the projected constraints
            num_new_constraints = len(pos_rows) * len(neg_rows) + len(zero_rows)
            A_new = np.zeros((num_new_constraints, A_current.shape[1] - 1))
            b_new = np.zeros(num_new_constraints)
            
            # Keep constraints where the coefficient is zero
            constraint_idx = 0
            for i in zero_rows:
                A_new[constraint_idx, :adjusted_idx] = A_current[i, :adjusted_idx]
                A_new[constraint_idx, adjusted_idx:] = A_current[i, (adjusted_idx+1):]
                b_new[constraint_idx] = b_current[i]
                constraint_idx += 1
            
            # Combine positive and negative constraints to eliminate the variable
            for i in pos_rows:
                for j in neg_rows:
                    # Normalize constraints so the coefficients of the eliminated variable are 1 and -1
                    alpha = A_current[i, adjusted_idx]
                    beta = -A_current[j, adjusted_idx]
                    
                    # Combine constraints: alpha * (row_j) + beta * (row_i)
                    A_new[constraint_idx, :adjusted_idx] = (beta * A_current[i, :adjusted_idx] + 
                                                        alpha * A_current[j, :adjusted_idx])
                    A_new[constraint_idx, adjusted_idx:] = (beta * A_current[i, (adjusted_idx+1):] + 
                                                        alpha * A_current[j, (adjusted_idx+1):])
                    b_new[constraint_idx] = beta * b_current[i] + alpha * b_current[j]
                    constraint_idx += 1
            
            # Update for the next iteration
            A_current = A_new
            b_current = b_new
        
        # Remove redundant constraints if possible
        # This is optional but can significantly reduce the number of constraints
        A_current, b_current = self._remove_redundant_constraints(A_current, b_current)
        
        return Polytope.from_hyperplanes(A_current, b_current)

    def _remove_redundant_constraints(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove redundant constraints from the H-representation of a polytope.
        
        Args:
            A: Matrix of constraint coefficients
            b: Vector of constraint constants
            
        Returns:
            Tuple of (A, b) with redundant constraints removed
        """
        # Simple implementation: check each constraint by solving an LP
        # For each constraint, maximize its violation subject to all other constraints
        # If the maximum violation is <= 0, the constraint is redundant
        
        if A.shape[0] == 0:
            return A, b
        
        non_redundant = []
        
        for i in range(A.shape[0]):
            # Create the constraints without the current one
            A_others = np.delete(A, i, axis=0)
            b_others = np.delete(b, i)
            
            # Set up the LP to maximize violation of constraint i
            # Maximize c^T x subject to A_others x <= b_others
            # where c = A[i] (we're maximizing the left side of A[i] x <= b[i])
            c = A[i]
            
            # Use scipy's linprog to solve the LP
            # We negate c because linprog minimizes by default
            from scipy.optimize import linprog
            res = linprog(-c, A_eq=None, b_eq=None, A_ub=A_others, b_ub=b_others, 
                        bounds=None, method='highs')
            
            # If the problem is infeasible, keep the constraint (degenerate case)
            # If the optimal value <= b[i], the constraint is redundant
            if not res.success or (-res.fun > b[i]):
                non_redundant.append(i)
        
        return A[non_redundant], b[non_redundant]
    
    def extrude(self, from_axes: AxisSet, to_axes: AxisSet) -> 'Polytope':
        """
        Extrude the polytope from a lower-dimensional space to a higher-dimensional space.
        
        This method extends a polytope by adding dimensions where constraints are unbounded.
        For linear affine systems, this is crucial for the distributed reachability algorithm.
        
        Args:
            from_axes: Original axis set (lower dimension)
            to_axes: Target axis set (higher dimension, must include from_axes)
            
        Returns:
            Extruded polytope in the higher-dimensional space
        """
        if not all(axis in to_axes.indices for axis in from_axes.indices):
            raise ValueError("Original axes must be a subset of target axes")
        
        # Get the H-representation (Ax <= b) of the original polytope
        A_orig, b_orig = self.get_constraints()
        
        # If the original polytope is empty, return an empty polytope in the target dimension
        if self.is_empty():
            return Polytope.from_hyperplanes(np.zeros((0, len(to_axes))), np.zeros(0))
        
        # Create mapping from original dimensions to target dimensions
        axis_mapping = {}
        for i, axis in enumerate(from_axes.indices):
            to_idx = to_axes.indices.index(axis)
            axis_mapping[i] = to_idx
        
        # Create the new constraint matrix for the higher-dimensional space
        A_new = np.zeros((A_orig.shape[0], len(to_axes)))
        
        # Map constraints to the appropriate dimensions
        for i in range(A_orig.shape[0]):
            for j in range(A_orig.shape[1]):
                if j in axis_mapping:
                    A_new[i, axis_mapping[j]] = A_orig[i, j]
        
        # The b vector remains unchanged
        b_new = b_orig.copy()
        
        # Add constraints to bound the polytope in the new dimensions if needed
        # This is optional and depends on the specific application
        # For distributed reachability, we typically want unbounded extrusion
        
        return Polytope.from_hyperplanes(A_new, b_new)
    
    def intersect(self, other: 'Polytope') -> 'Polytope':
        """
        Compute the intersection with another polytope.
        
        Args:
            other: Another polytope
            
        Returns:
            Intersection polytope
        """
        intersection = pc.intersect(self.polytope, other.polytope)
        return Polytope(intersection)
    
    @staticmethod
    def intersect_many(polytopes: List['Polytope']) -> 'Polytope':
        """
        Compute the intersection of multiple polytopes.
        
        Args:
            polytopes: List of polytopes to intersect
            
        Returns:
            Intersection polytope
        """
        if not polytopes:
            raise ValueError("Empty list of polytopes")
        
        result = polytopes[0]
        for poly in polytopes[1:]:
            result = result.intersect(poly)
        
        return result
    
    def is_empty(self) -> bool:
        """
        Check if the polytope is empty.
        
        Returns:
            True if empty, False otherwise
        """
        A,b = self.get_constraints()
        return A.shape[0] > 0 and A.shape[1] == 0
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if the polytope contains a point.
        
        Args:
            point: Point to check
            
        Returns:
            True if the polytope contains the point, False otherwise
        """
        return self.polytope.contains(point)
    
    def __eq__(self, other: 'Polytope') -> bool:
        """
        Check if two polytopes are equal.
        
        Args:
            other: Another polytope
            
        Returns:
            True if equal, False otherwise
        """
        # This depends on the implementation in the base package
        # A common approach is to check if both contain each other
        return (self.polytope <= other.polytope) and (other.polytope <= self.polytope)
    
    def volume(self) -> float:
        """
        Compute the volume of the polytope.
        
        Returns:
            Volume of the polytope
        """
        return self.polytope.volume
    
    def get_generators(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the generators (vertices) of the polytope.
        
        Returns:
            Tuple containing vertices and rays
        """
        # This depends on the implementation in the base package
        return self.polytope.generators
    
    def get_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the constraints (A, b) of the polytope.
        
        Returns:
            Tuple of (A, b) representing the constraints Ax <= b
        """
        return self.polytope.A, self.polytope.b
    
    def __str__(self) -> str:
        """String representation of the polytope."""
        return str(self.polytope)
    
    def __repr__(self) -> str:
        """Formal string representation of the polytope."""
        return repr(self.polytope)