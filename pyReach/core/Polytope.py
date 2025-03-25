import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import warnings
from typing import List, Tuple, Optional, Union, Set


class Polytope:
    """
    A class representing a polytope in the form Ax <= b.
    Designed for operations related to distributed backward reachability for linear affine systems.
    """
    
    def __init__(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-10):
        """
        Initialize a polytope with its constraint matrices.
        
        Parameters:
        -----------
        A : numpy.ndarray
            Coefficient matrix of shape (num_constraints, dimension)
        b : numpy.ndarray
            Right-hand side vector of shape (num_constraints,)
        tol : float, optional
            Numerical tolerance for various operations
        """
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimensions mismatch: A has {A.shape[0]} rows, b has {b.shape[0]} elements")
        
        # Remove redundant constraints to simplify the representation
        self.A, self.b = self._remove_redundant_constraints(A, b)
        self.dim = self.A.shape[1]  # Dimension of the space
        self.tol = tol
    
    def _remove_redundant_constraints(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove redundant constraints from the polytope representation.
        
        This function identifies and removes constraints that are redundant (implied by other constraints).
        For linear affine systems, this helps keep the representation minimal which speeds up projections.
        
        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray]
            Simplified A matrix and b vector
        """
        if A.shape[0] == 0:  # Empty constraint set
            return A, b
            
        non_redundant_rows = []
        for i in range(A.shape[0]):
            # Create a linear program to check if constraint i is redundant
            # Maximize A[i]·x - b[i] subject to all other constraints
            c = -A[i]  # Negative because we're maximizing
            A_others = np.delete(A, i, axis=0)
            b_others = np.delete(b, i)
            
            # We need to handle potential infeasibility
            try:
                # Use the simplex method as it's more robust for this task
                result = linprog(c, A_ub=A_others, b_ub=b_others, method='highs')
                
                # If the solution's negative objective value is less than -b[i] - tol, constraint i is not redundant
                # Note: LP maximizes -A[i]·x, so result.fun is -max(A[i]·x)
                if result.success and -result.fun <= b[i] + self.tol:
                    # Constraint i is redundant (implied by other constraints)
                    continue
                non_redundant_rows.append(i)
            except:
                # If the LP fails, conservatively keep the constraint
                non_redundant_rows.append(i)
        
        # Return the non-redundant constraints
        if non_redundant_rows:
            return A[non_redundant_rows], b[non_redundant_rows]
        # If all constraints are redundant, the polytope is unbounded - return an empty constraint set
        return np.zeros((0, A.shape[1])), np.zeros(0)
    
    def is_empty(self) -> bool:
        """
        Check if the polytope is empty (has no feasible points).
        
        Returns:
        --------
        bool
            True if the polytope is empty, False otherwise
        """
        if self.A.shape[0] == 0:
            return False  # No constraints means the polytope is the entire space
            
        # Create a linear program to find any feasible point
        # We minimize a dummy objective (zeros) just to get a feasible point
        c = np.zeros(self.dim)
        try:
            result = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
            return not result.success
        except:
            # If the LP solver fails, conservatively say the polytope is not empty
            warnings.warn("LP solver failed to determine if polytope is empty")
            return False
    
    def contains(self, point: np.ndarray, tol: Optional[float] = None) -> bool:
        """
        Check if a point is contained in the polytope.
        
        Parameters:
        -----------
        point : numpy.ndarray
            The point to check, shape (dimension,)
        tol : float, optional
            Tolerance for numerical comparisons, defaults to self.tol
            
        Returns:
        --------
        bool
            True if the point is in the polytope, False otherwise
        """
        if tol is None:
            tol = self.tol
            
        if len(point) != self.dim:
            raise ValueError(f"Point dimension {len(point)} does not match polytope dimension {self.dim}")
            
        # Check if all constraints are satisfied: Ax <= b
        return np.all(self.A @ point <= self.b + tol)
    
    def project(self, axis_set: Set[int]) -> 'Polytope':
        """
        Project the polytope onto the subspace defined by the given axes.
        
        Parameters:
        -----------
        axis_set : Set[int]
            Set of coordinate indices to project onto (0-indexed)
            
        Returns:
        --------
        Polytope
            The projected polytope
        """
        if not axis_set:
            return Polytope(np.zeros((0, 0)), np.zeros(0))
            
        if max(axis_set) >= self.dim or min(axis_set) < 0:
            raise ValueError(f"Invalid axis indices in {axis_set}, must be between 0 and {self.dim-1}")
            
        # For projection, we use Fourier-Motzkin elimination to project out variables not in axis_set
        # This is efficient for linear affine systems
        
        # Get the indices to eliminate
        to_eliminate = set(range(self.dim)) - set(axis_set)
        
        # Start with the current polytope
        A_proj = self.A.copy()
        b_proj = self.b.copy()
        
        # Eliminate variables one by one
        for var_idx in sorted(to_eliminate, reverse=True):
            # Partition constraints based on coefficient sign for the variable to eliminate
            pos_rows = np.where(A_proj[:, var_idx] > self.tol)[0]
            neg_rows = np.where(A_proj[:, var_idx] < -self.tol)[0]
            zero_rows = np.where(np.abs(A_proj[:, var_idx]) <= self.tol)[0]
            
            # Create new constraints by combining positive and negative constraints
            new_constraints = []
            new_rhs = []
            
            # Keep constraints not involving this variable
            if len(zero_rows) > 0:
                new_constraints.append(A_proj[zero_rows])
                new_rhs.append(b_proj[zero_rows])
            
            # Combine positive and negative constraints to eliminate the variable
            for i in pos_rows:
                for j in neg_rows:
                    # Scale constraints so the coefficients for var_idx cancel out
                    scale_i = -A_proj[j, var_idx]
                    scale_j = A_proj[i, var_idx]
                    
                    # Combine the constraints
                    new_constraint = scale_i * A_proj[i] + scale_j * A_proj[j]
                    new_right_hand = scale_i * b_proj[i] + scale_j * b_proj[j]
                    
                    # Skip if the constraint becomes numerically unstable
                    if np.any(np.abs(new_constraint) > 1e10):
                        continue
                    
                    new_constraints.append(new_constraint.reshape(1, -1))
                    new_rhs.append(np.array([new_right_hand]))
            
            # Combine all new constraints
            if new_constraints:
                A_proj = np.vstack(new_constraints)
                b_proj = np.concatenate(new_rhs)
            else:
                # If no constraints left, the projection is the entire space
                return Polytope(np.zeros((0, len(axis_set))), np.zeros(0))
        
        # Extract columns corresponding to kept variables and create the projected polytope
        kept_cols = sorted(axis_set)
        A_final = A_proj[:, kept_cols]
        return Polytope(A_final, b_proj)
    
    def extrude(self, axis_set: Set[int], full_space_dim: int) -> 'Polytope':
        """
        Extrude the polytope into a higher dimensional space.
        
        Parameters:
        -----------
        axis_set : Set[int]
            Set of coordinate indices that the polytope lives in (0-indexed)
        full_space_dim : int
            Dimension of the full space to extrude into
            
        Returns:
        --------
        Polytope
            The extruded polytope
        """
        if not axis_set:
            # Empty axis set means the polytope is unconstrained in all dimensions
            return Polytope(np.zeros((0, full_space_dim)), np.zeros(0))
            
        if max(axis_set) >= full_space_dim or min(axis_set) < 0:
            raise ValueError(f"Invalid axis indices in {axis_set}, must be between 0 and {full_space_dim-1}")
            
        if len(axis_set) != self.dim:
            raise ValueError(f"Axis set size {len(axis_set)} doesn't match polytope dimension {self.dim}")
        
        # Create a mapping from original indices to the indices in the full space
        axis_list = sorted(axis_set)
        
        # Create the new constraint matrix
        A_extruded = np.zeros((self.A.shape[0], full_space_dim))
        
        # Map constraints from the original space to the full space
        for i, orig_idx in enumerate(axis_list):
            A_extruded[:, orig_idx] = self.A[:, i]
        
        return Polytope(A_extruded, self.b)
    
    def intersection(self, other: 'Polytope') -> 'Polytope':
        """
        Compute the intersection of this polytope with another.
        
        Parameters:
        -----------
        other : Polytope
            Another polytope to intersect with
            
        Returns:
        --------
        Polytope
            The intersection polytope
        """
        if self.dim != other.dim:
            raise ValueError(f"Cannot intersect polytopes of different dimensions: {self.dim} and {other.dim}")
        
        # Intersect by simply stacking the constraints
        A_combined = np.vstack((self.A, other.A))
        b_combined = np.concatenate((self.b, other.b))
        
        return Polytope(A_combined, b_combined)
    
    def get_chebyshev_center(self) -> Optional[np.ndarray]:
        """
        Find the Chebyshev center of the polytope - the center of the largest inscribed ball.
        This is useful for finding a "central" point inside the polytope.
        
        Returns:
        --------
        numpy.ndarray or None
            The Chebyshev center if the polytope is bounded, None otherwise
        """
        if self.A.shape[0] == 0:
            # If no constraints, polytope is unbounded
            return None
            
        # Normalize rows of A to have unit norm
        row_norms = np.linalg.norm(self.A, axis=1)
        A_normalized = self.A / row_norms[:, np.newaxis]
        b_normalized = self.b / row_norms
        
        # Set up LP to maximize the radius of the inscribed ball
        # Variables: [x_1, ..., x_n, r] where r is the radius
        c = np.zeros(self.dim + 1)
        c[-1] = -1  # Maximize r
        
        # Constraints: a_i · x + r||a_i|| ≤ b_i  for all i
        A_lp = np.hstack((A_normalized, np.ones((A_normalized.shape[0], 1))))
        
        try:
            result = linprog(c, A_ub=A_lp, b_ub=b_normalized, method='highs')
            if result.success:
                return result.x[:-1]  # Return the center (exclude the radius)
            return None
        except:
            # If LP fails, try a simpler approach
            warnings.warn("Chebyshev center computation failed, returning a feasible point instead")
            return self._find_feasible_point()
    
    def _find_feasible_point(self) -> Optional[np.ndarray]:
        """
        Find any feasible point inside the polytope.
        
        Returns:
        --------
        numpy.ndarray or None
            A feasible point if one exists, None otherwise
        """
        c = np.zeros(self.dim)  # We don't care about optimization, just feasibility
        try:
            result = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
            if result.success:
                return result.x
            return None
        except:
            return None
    
    def get_vertices(self, max_vertices: int = 1000) -> Optional[np.ndarray]:
        """
        Compute the vertices of the polytope using a vertex enumeration algorithm.
        
        Parameters:
        -----------
        max_vertices : int, optional
            Maximum number of vertices to compute before giving up (for efficiency)
            
        Returns:
        --------
        numpy.ndarray or None
            Array of shape (num_vertices, dimension) containing the vertices,
            or None if the polytope is unbounded or has too many vertices
        """
        if self.A.shape[0] == 0:
            return None  # Unbounded polytope
            
        # For high-dimensional polytopes, this can be computationally expensive
        # So we'll implement a simple approach using linear programming
        
        # First, check if the polytope is bounded
        for i in range(self.dim):
            # Check if we can make coordinate i arbitrarily large positive
            c = np.zeros(self.dim)
            c[i] = 1
            result = linprog(-c, A_ub=self.A, b_ub=self.b, method='highs')
            if not result.success:
                return None  # Polytope is unbounded
        
        # Now, compute vertices via the "vertex enumeration" approach
        vertices = []
        
        # Generate a set of random objective directions
        np.random.seed(42)  # For reproducibility
        num_directions = min(max_vertices, self.dim * 10)
        directions = np.random.randn(num_directions, self.dim)
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        
        # Add axis-aligned directions
        for i in range(self.dim):
            vec = np.zeros(self.dim)
            vec[i] = 1
            directions = np.vstack((directions, vec, -vec))
        
        # Find extreme points in each direction
        for direction in directions:
            result = linprog(-direction, A_ub=self.A, b_ub=self.b, method='highs')
            if result.success:
                vertex = result.x
                # Check if this vertex is already in our list (up to tolerance)
                is_new = all(np.linalg.norm(vertex - v) > self.tol for v in vertices)
                if is_new:
                    vertices.append(vertex)
                    if len(vertices) > max_vertices:
                        # Too many vertices, likely a numerical issue
                        return None
        
        if not vertices:
            # If no vertices found, try to find a single feasible point
            point = self._find_feasible_point()
            if point is not None:
                return np.array([point])
            return None
        
        return np.array(vertices)
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two polytopes are equal.
        
        Parameters:
        -----------
        other : object
            Another object to compare with
            
        Returns:
        --------
        bool
            True if the polytopes are equal, False otherwise
        """
        if not isinstance(other, Polytope):
            return False
            
        if self.dim != other.dim:
            return False
            
        # Check if one is a subset of the other and vice versa
        return self.is_subset(other) and other.is_subset(self)
    
    def is_subset(self, other: 'Polytope') -> bool:
        """
        Check if this polytope is a subset of another polytope.
        
        Parameters:
        -----------
        other : Polytope
            Another polytope to compare with
            
        Returns:
        --------
        bool
            True if this polytope is a subset of the other, False otherwise
        """
        if self.dim != other.dim:
            raise ValueError(f"Cannot compare polytopes of different dimensions: {self.dim} and {other.dim}")
            
        # A polytope P1 is a subset of P2 if and only if all vertices of P1 are in P2
        # and all of P1's constraints are implied by P2's constraints
        
        # First, check if this polytope is empty
        if self.is_empty():
            return True  # Empty set is a subset of any set
            
        # Get vertices of this polytope
        vertices = self.get_vertices()
        if vertices is None:
            # If we can't get vertices (e.g., unbounded polytope), use a more expensive approach
            for i in range(other.A.shape[0]):
                # For each constraint in other, check if it's redundant in the context of self
                # Maximize other.A[i]·x - other.b[i] subject to self's constraints
                c = -other.A[i]
                try:
                    result = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
                    if result.success and -result.fun > other.b[i] + self.tol:
                        # Found a point in self that violates a constraint in other
                        return False
                except:
                    # If LP fails, conservatively say not a subset
                    return False
            return True
                
        # Check if all vertices of this polytope are contained in the other polytope
        for vertex in vertices:
            if not other.contains(vertex):
                return False
                
        return True
    
    def volume(self) -> float:
        """
        Compute the volume of the polytope.
        
        Returns:
        --------
        float
            The volume of the polytope
        """
        # Get vertices of the polytope
        vertices = self.get_vertices()
        if vertices is None or len(vertices) <= self.dim:
            return 0.0  # Unbounded or lower-dimensional polytope
            
        try:
            # Compute convex hull to get a proper triangulation
            hull = ConvexHull(vertices)
            return hull.volume
        except:
            # If computation fails, return 0
            warnings.warn("Volume computation failed")
            return 0.0
    
    def __str__(self) -> str:
        """String representation of the polytope."""
        if self.A.shape[0] == 0:
            return f"Polytope(Dimension={self.dim}, Constraints=None)"
        return f"Polytope(Dimension={self.dim}, Constraints={self.A.shape[0]})"
    
    def __repr__(self) -> str:
        """Detailed representation of the polytope."""
        return str(self)