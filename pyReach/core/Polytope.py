import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import warnings
from typing import List, Tuple, Optional, Union, Set


class Polytope:
    def __init__(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-10):
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimensions mismatch: A has {A.shape[0]} rows, b has {b.shape[0]} elements")
        self.A, self.b = self._remove_redundant_constraints(A, b)
        self.dim = self.A.shape[1]
        self.tol = tol
    
    def _remove_redundant_constraints(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if A.shape[0] == 0:
            return A, b
            
        non_redundant_rows = []
        for i in range(A.shape[0]):
            c = -A[i]
            A_others = np.delete(A, i, axis=0)
            b_others = np.delete(b, i)
            
            try:
                result = linprog(c, A_ub=A_others, b_ub=b_others, method='highs')
                if result.success and -result.fun <= b[i] + self.tol:
                    continue
                non_redundant_rows.append(i)
            except:
                non_redundant_rows.append(i)
        
        if non_redundant_rows:
            return A[non_redundant_rows], b[non_redundant_rows]
        return np.zeros((0, A.shape[1])), np.zeros(0)
    
    def is_empty(self) -> bool:
        if self.A.shape[0] == 0:
            return False
            
        c = np.zeros(self.dim)
        try:
            result = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
            return not result.success
        except:
            return False
    
    def contains(self, point: np.ndarray, tol: Optional[float] = None) -> bool:
        if tol is None:
            tol = self.tol
            
        if len(point) != self.dim:
            raise ValueError(f"Point dimension {len(point)} does not match polytope dimension {self.dim}")
            
        return np.all(self.A @ point <= self.b + tol)
    
    def project(self, axis_set: Set[int]) -> 'Polytope':
        if not axis_set:
            return Polytope(np.zeros((0, 0)), np.zeros(0))
            
        if max(axis_set) >= self.dim or min(axis_set) < 0:
            raise ValueError(f"Invalid axis indices in {axis_set}, must be between 0 and {self.dim-1}")

        to_eliminate = set(range(self.dim)) - set(axis_set)
        
        A_proj = self.A.copy()
        b_proj = self.b.copy()
        
        for var_idx in sorted(to_eliminate, reverse=True):
            pos_rows = np.where(A_proj[:, var_idx] > self.tol)[0]
            neg_rows = np.where(A_proj[:, var_idx] < -self.tol)[0]
            zero_rows = np.where(np.abs(A_proj[:, var_idx]) <= self.tol)[0]
            
            new_constraints = []
            new_rhs = []
            
            if len(zero_rows) > 0:
                new_constraints.append(A_proj[zero_rows])
                new_rhs.append(b_proj[zero_rows])
            
            for i in pos_rows:
                for j in neg_rows:
                    scale_i = -A_proj[j, var_idx]
                    scale_j = A_proj[i, var_idx]
                    
                    new_constraint = scale_i * A_proj[i] + scale_j * A_proj[j]
                    new_right_hand = scale_i * b_proj[i] + scale_j * b_proj[j]
                    
                    if np.any(np.abs(new_constraint) > 1e10):
                        continue
                    
                    new_constraints.append(new_constraint.reshape(1, -1))
                    new_rhs.append(np.array([new_right_hand]))
            
            if new_constraints:
                A_proj = np.vstack(new_constraints)
                b_proj = np.concatenate(new_rhs)
            else:
                return Polytope(np.zeros((0, len(axis_set))), np.zeros(0))
        
        kept_cols = sorted(axis_set)
        A_final = A_proj[:, kept_cols]
        return Polytope(A_final, b_proj)
    
    def extrude(self, axis_set: Set[int], full_space_dim: int) -> 'Polytope':
        if not axis_set:
            return Polytope(np.zeros((0, full_space_dim)), np.zeros(0))
            
        if max(axis_set) >= full_space_dim or min(axis_set) < 0:
            raise ValueError(f"Invalid axis indices in {axis_set}, must be between 0 and {full_space_dim-1}")
            
        if len(axis_set) != self.dim:
            raise ValueError(f"Axis set size {len(axis_set)} doesn't match polytope dimension {self.dim}")
        
        axis_list = sorted(axis_set)
        
        A_extruded = np.zeros((self.A.shape[0], full_space_dim))
        
        for i, orig_idx in enumerate(axis_list):
            A_extruded[:, orig_idx] = self.A[:, i]
        
        return Polytope(A_extruded, self.b)
    
    def intersection(self, other: 'Polytope') -> 'Polytope':
        if self.dim != other.dim:
            raise ValueError(f"Cannot intersect polytopes of different dimensions: {self.dim} and {other.dim}")
        
        A_combined = np.vstack((self.A, other.A))
        b_combined = np.concatenate((self.b, other.b))
        
        return Polytope(A_combined, b_combined)
    
    def get_chebyshev_center(self) -> Optional[np.ndarray]:
        if self.A.shape[0] == 0:
            return None
            
        row_norms = np.linalg.norm(self.A, axis=1)
        A_normalized = self.A / row_norms[:, np.newaxis]
        b_normalized = self.b / row_norms
        
        c = np.zeros(self.dim + 1)
        c[-1] = -1  # Maximize r
        
        A_lp = np.hstack((A_normalized, np.ones((A_normalized.shape[0], 1))))
        
        try:
            result = linprog(c, A_ub=A_lp, b_ub=b_normalized, method='highs')
            if result.success:
                return result.x[:-1]
            return None
        except:
            warnings.warn("Chebyshev center computation failed, returning a feasible point instead")
            return self._find_feasible_point()
    
    def _find_feasible_point(self) -> Optional[np.ndarray]:
        c = np.zeros(self.dim)
        try:
            result = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
            if result.success:
                return result.x
            return None
        except:
            return None
    
    def get_vertices(self, max_vertices: int = 1000) -> Optional[np.ndarray]:
        if self.A.shape[0] == 0:
            return None

        for i in range(self.dim):
            c = np.zeros(self.dim)
            c[i] = 1
            result = linprog(-c, A_ub=self.A, b_ub=self.b, method='highs')
            if not result.success:
                return None
        
        vertices = []
        

        np.random.seed(42)
        num_directions = min(max_vertices, self.dim * 10)
        directions = np.random.randn(num_directions, self.dim)
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        
        for i in range(self.dim):
            vec = np.zeros(self.dim)
            vec[i] = 1
            directions = np.vstack((directions, vec, -vec))
        
        for direction in directions:
            result = linprog(-direction, A_ub=self.A, b_ub=self.b, method='highs')
            if result.success:
                vertex = result.x
                is_new = all(np.linalg.norm(vertex - v) > self.tol for v in vertices)
                if is_new:
                    vertices.append(vertex)
                    if len(vertices) > max_vertices:
                        return None
        
        if not vertices:
            point = self._find_feasible_point()
            if point is not None:
                return np.array([point])
            return None
        
        return np.array(vertices)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polytope):
            return False
            
        if self.dim != other.dim:
            return False
            
        return self.is_subset(other) and other.is_subset(self)
    
    def is_subset(self, other: 'Polytope') -> bool:
        if self.dim != other.dim:
            raise ValueError(f"Cannot compare polytopes of different dimensions: {self.dim} and {other.dim}")
            
        if self.is_empty():
            return True
            
        vertices = self.get_vertices()
        if vertices is None:
            for i in range(other.A.shape[0]):
                c = -other.A[i]
                try:
                    result = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
                    if result.success and -result.fun > other.b[i] + self.tol:
                        return False
                except:
                    return False
            return True
                
        for vertex in vertices:
            if not other.contains(vertex):
                return False
                
        return True
    
    def volume(self) -> float:
        vertices = self.get_vertices()
        if vertices is None or len(vertices) <= self.dim:
            return 0.0
            
        try:
            hull = ConvexHull(vertices)
            return hull.volume
        except:
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