from pyReach.core.ContSet import contSet
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class Polytope(contSet):
    """Polytope representation using H-representation: {x | Ax <= b} or V-representation (vertices)."""
    
    def __init__(self, A=None, b=None, vertices=None):
        """
        Initialize a polytope using either H-representation or V-representation.
        
        Args:
            A (array-like, optional): Matrix A in H-representation Ax <= b
            b (array-like, optional): Vector b in H-representation Ax <= b
            vertices (array-like, optional): Matrix of vertices (one vertex per row)
        
        Raises:
            ValueError: If neither (A,b) nor vertices are provided
        """
        if A is not None and b is not None:
            self.A = np.array(A, dtype=float)
            self.b = np.array(b, dtype=float).flatten()
            self._vertices = None
            self._rep_type = 'H'
            
            # Validate dimensions
            if len(self.b) != self.A.shape[0]:
                raise ValueError(f"Dimension mismatch: A has {self.A.shape[0]} rows, but b has {len(self.b)} elements")
                
        elif vertices is not None:
            self._vertices = np.array(vertices, dtype=float)
            if len(self._vertices.shape) == 1:
                # Single vertex, reshape to 2D
                self._vertices = self._vertices.reshape(1, -1)
            self.A, self.b = None, None
            self._rep_type = 'V'
            
            # Compute H-representation from vertices if more than dimension+1 vertices
            if len(self._vertices) > self.dimension() + 1:
                try:
                    self._compute_h_rep_from_vertices()
                except:
                    # Keep V-representation if conversion fails
                    pass
        else:
            raise ValueError("Either (A,b) or vertices must be provided")
    
    def _compute_h_rep_from_vertices(self):
        """Compute H-representation from vertices using convex hull."""
        if self._vertices is None or len(self._vertices) <= self.dimension():
            return
        
        try:
            # Compute convex hull
            hull = ConvexHull(self._vertices)
            
            # Extract halfspaces from the convex hull
            self.A = hull.equations[:, :-1]
            self.b = -hull.equations[:, -1]
            self._rep_type = 'H+V'  # Both representations are available
        except Exception as e:
            # Conversion failed, keep V-representation
            print(f"Warning: H-representation computation failed: {e}")
    
    def _compute_vertices_from_h_rep(self):
        """Compute vertices from H-representation."""
        if self.A is None or self.b is None:
            return
        
        try:
            # For 2D, use vertex enumeration
            if self.dimension() == 2:
                self._vertices = self._compute_vertices_2d()
                self._rep_type = 'H+V'  # Both representations are available
                return
            
            # For higher dimensions, use more advanced methods
            # Find a point inside the polytope
            interior_point = self._find_interior_point()
            if interior_point is None:
                # Polytope might be empty
                self._vertices = np.zeros((0, self.dimension()))
                return
            
            # Use halfspace intersection
            halfspaces = np.column_stack([self.A, -self.b])
            hs_intersection = HalfspaceIntersection(halfspaces, interior_point)
            
            # Extract vertices
            self._vertices = hs_intersection.intersections
            self._rep_type = 'H+V'  # Both representations are available
        except Exception as e:
            # Conversion failed, keep H-representation
            print(f"Warning: Vertex computation failed: {e}")
    
    def _compute_vertices_2d(self):
        """Compute vertices for a 2D polytope from H-representation."""
        if self.dimension() != 2:
            raise ValueError("This method only works for 2D polytopes")
        
        A, b = self.A, self.b
        n_constraints = A.shape[0]
        vertices = []
        
        # For each pair of constraints, find their intersection
        for i in range(n_constraints):
            for j in range(i+1, n_constraints):
                # Solve the 2x2 system: A_i * x = b_i, A_j * x = b_j
                try:
                    A_sub = np.vstack([A[i], A[j]])
                    b_sub = np.array([b[i], b[j]])
                    vertex = np.linalg.solve(A_sub, b_sub)
                    
                    # Check if the vertex satisfies all constraints
                    if np.all(A @ vertex <= b + 1e-10):
                        vertices.append(vertex)
                except np.linalg.LinAlgError:
                    # Parallel constraints
                    continue
        
        # Sort vertices counterclockwise
        if vertices:
            vertices = np.array(vertices)
            # Find centroid
            centroid = np.mean(vertices, axis=0)
            # Sort by angle
            angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            return vertices[sorted_indices]
        else:
            return np.zeros((0, 2))
    
    def _find_interior_point(self):
        """Find a point inside the polytope using linear programming."""
        # Use Chebyshev center method
        A, b = self.A, self.b
        n_constraints, n_vars = A.shape
        
        if n_constraints == 0:
            return np.zeros(n_vars)
        
        # Normalize rows of A to have unit norm
        norms = np.sqrt(np.sum(A * A, axis=1))
        if np.any(norms == 0):
            # Handle zero rows
            valid_rows = norms > 0
            A = A[valid_rows]
            b = b[valid_rows]
            norms = norms[valid_rows]
            n_constraints = A.shape[0]
            
            if n_constraints == 0:
                return np.zeros(n_vars)
        
        A_normalized = A / norms.reshape(-1, 1)
        b_normalized = b / norms
        
        # Set up LP to maximize the radius of a sphere inside the polytope
        # max r s.t. a_i^T x + r ||a_i|| <= b_i for all i
        # Rewritten as: max r s.t. a_i^T x + r <= b_i / ||a_i|| for all i
        
        # Objective: [-r, 0, 0, ..., 0] to maximize r
        c = np.zeros(n_vars + 1)
        c[0] = -1  # Maximize r
        
        # Constraints: [||a_i||, a_i] [r; x] <= b_i
        A_lp = np.column_stack([np.ones(n_constraints), A_normalized])
        b_lp = b_normalized
        
        # Solve LP
        result = linprog(c, A_ub=A_lp, b_ub=b_lp, method='highs')
        
        if result.success:
            # Extract Chebyshev center
            return result.x[1:]
        else:
            # Polytope might be empty or unbounded
            return None
    
    def dimension(self):
        """Return the dimension of the polytope."""
        if self.A is not None:
            return self.A.shape[1]
        elif self._vertices is not None:
            return self._vertices.shape[1]
        else:
            raise ValueError("Polytope has neither H-representation nor V-representation")
    
    def is_empty(self):
        """Check if the polytope is empty."""
        if self._vertices is not None:
            return len(self._vertices) == 0
        
        # Check if H-representation defines an empty polytope
        # Try to find a feasible point
        A, b = self.A, self.b
        n_vars = A.shape[1]
        
        # Objective function (any will do since we only care about feasibility)
        c = np.zeros(n_vars)
        
        # Solve LP
        result = linprog(c, A_ub=A, b_ub=b, method='highs')
        
        return not result.success
    
    def contains(self, point, method='exact'):
        """
        Check if the polytope contains a point.
        
        Args:
            point (array-like): The point to check
            method (str): Method to use (ignored, always exact)
        
        Returns:
            bool: True if the polytope contains the point, False otherwise
        """
        point = np.array(point, dtype=float).flatten()
        
        if len(point) != self.dimension():
            raise ValueError(f"Dimension mismatch: polytope has dimension {self.dimension()}, "
                             f"but point has dimension {len(point)}")
        
        # If we have H-representation, check if Ax <= b
        if self.A is not None and self.b is not None:
            return np.all(self.A @ point <= self.b + 1e-10)
        
        # If only V-representation is available, check if point is in convex hull
        # This is more complicated - use an approximate method
        if self._vertices is not None:
            # For 2D polytopes
            if self.dimension() == 2:
                # Use winding number algorithm for 2D
                return self._point_in_polygon_2d(point)
            
            # For higher dimensions, try to use barycentric coordinates
            try:
                # Add a dimension of ones for affine combination
                vertices_homog = np.column_stack([self._vertices, np.ones(len(self._vertices))])
                point_homog = np.append(point, 1)
                
                # Solve for barycentric coordinates
                # V * lambda = p, sum(lambda) = 1
                coeffs = np.linalg.lstsq(vertices_homog, point_homog, rcond=None)[0]
                
                # Check if all coefficients are positive (within tolerance)
                return np.all(coeffs >= -1e-10) and abs(np.sum(coeffs) - 1) < 1e-10
            except:
                # Fall back to computing H-representation
                self._compute_h_rep_from_vertices()
                if self.A is not None and self.b is not None:
                    return np.all(self.A @ point <= self.b + 1e-10)
                
                # If all else fails, return False
                return False
        
        return False
    
    def _point_in_polygon_2d(self, point):
        """
        Check if a point is inside a 2D polygon using the crossing number algorithm.
        
        Args:
            point (array-like): The point to check [x, y]
        
        Returns:
            bool: True if the point is inside the polygon, False otherwise
        """
        if self.dimension() != 2 or self._vertices is None:
            raise ValueError("This method only works for 2D polytopes with V-representation")
        
        if len(self._vertices) < 3:
            # Not a proper polygon
            return False
        
        # Crossing number algorithm
        x, y = point
        n_crossings = 0
        
        for i in range(len(self._vertices)):
            x1, y1 = self._vertices[i]
            x2, y2 = self._vertices[(i + 1) % len(self._vertices)]
            
            # Check if ray from point crosses this edge
            if ((y1 <= y < y2) or (y2 <= y < y1)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                n_crossings += 1
        
        # Odd number of crossings means point is inside
        return n_crossings % 2 == 1
    
    def is_intersecting(self, other_set, method='exact'):
        """
        Check if this polytope intersects with another set.
        
        Args:
            other_set (Set): The other set to check intersection with
            method (str): Method to use for intersection check
        
        Returns:
            bool: True if the sets intersect, False otherwise
        """
        if isinstance(other_set, Polytope):
            # For two polytopes, check if there exists a point that satisfies both sets of constraints
            
            if self._rep_type.startswith('H') and other_set._rep_type.startswith('H'):
                # Both have H-representation, combine constraints
                A_combined = np.vstack([self.A, other_set.A])
                b_combined = np.hstack([self.b, other_set.b])
                
                # Check if the combined system has a feasible point
                n_vars = self.dimension()
                c = np.zeros(n_vars)  # Any objective function will do
                
                result = linprog(c, A_ub=A_combined, b_ub=b_combined, method='highs')
                
                return result.success
            
            # If one or both don't have H-representation, try other methods
            
            # Check if any vertex of one polytope is in the other
            if self._vertices is None:
                self._compute_vertices_from_h_rep()
            
            if other_set._vertices is None:
                other_set._compute_vertices_from_h_rep()
            
            if self._vertices is not None and len(self._vertices) > 0:
                for vertex in self._vertices:
                    if other_set.contains(vertex):
                        return True
            
            if other_set._vertices is not None and len(other_set._vertices) > 0:
                for vertex in other_set._vertices:
                    if self.contains(vertex):
                        return True
            
            # If still no intersection found, check for edge intersections (only for 2D)
            if self.dimension() == 2:
                if self._vertices is None or other_set._vertices is None:
                    return False
                
                # Check for edge intersections
                for i in range(len(self._vertices)):
                    v1 = self._vertices[i]
                    v2 = self._vertices[(i + 1) % len(self._vertices)]
                    
                    for j in range(len(other_set._vertices)):
                        v3 = other_set._vertices[j]
                        v4 = other_set._vertices[(j + 1) % len(other_set._vertices)]
                        
                        if self._line_segments_intersect(v1, v2, v3, v4):
                            return True
            
            # No intersection found
            return False
        
        # For other set types, try to use the other set's intersection method
        try:
            return other_set.is_intersecting(self, method)
        except:
            # Fall back to checking if any vertex of this polytope is in the other set
            if self._vertices is None:
                self._compute_vertices_from_h_rep()
            
            if self._vertices is not None and len(self._vertices) > 0:
                for vertex in self._vertices:
                    if other_set.contains(vertex):
                        return True
            
            # As a last resort, check if the other set's center is in this polytope
            try:
                if self.contains(other_set.center()):
                    return True
            except:
                pass
            
            return False
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """
        Check if two line segments intersect.
        
        Args:
            p1, p2: First line segment endpoints
            p3, p4: Second line segment endpoints
        
        Returns:
            bool: True if the line segments intersect, False otherwise
        """
        # Convert points to vectors
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        p4 = np.array(p4)
        
        # Compute direction vectors
        d1 = p2 - p1
        d2 = p4 - p3
        
        # Compute cross product
        cross = np.cross(d1, d2)
        
        # If cross product is zero, lines are parallel
        if abs(cross) < 1e-10:
            # Check if they overlap
            if abs(np.cross(p3 - p1, d1)) < 1e-10:
                # Lines are collinear, check if segments overlap
                t1 = np.dot(p3 - p1, d1) / np.dot(d1, d1)
                t2 = np.dot(p4 - p1, d1) / np.dot(d1, d1)
                return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 <= 0 and t2 >= 1) or (t2 <= 0 and t1 >= 1)
            return False
        
        # Compute parameters for intersection point
        t1 = np.cross(p3 - p1, d2) / cross
        t2 = np.cross(p3 - p1, d1) / cross
        
        # Check if intersection point is within both segments
        return (0 <= t1 <= 1) and (0 <= t2 <= 1)
    
    def plot(self, dims=None, **kwargs):
        """
        Plot a 2D projection of the polytope.
        
        Args:
            dims (list, optional): List of two dimensions to project onto. Default is [0, 1].
            **kwargs: Additional keyword arguments to pass to matplotlib.
        
        Returns:
            matplotlib.axes.Axes: The axes object with the plot.
        """
        if dims is None:
            dims = [0, 1]
        
        if len(dims) != 2:
            raise ValueError("dims must be a list of two dimensions")
        
        # Make sure we have vertices
        if self._vertices is None:
            self._compute_vertices_from_h_rep()
        
        # If we still don't have vertices or there are too few, try to compute them specially
        if self._vertices is None or len(self._vertices) < 3:
            if self.A is not None and self.b is not None:
                # For 2D, compute a grid of points and find those inside the polytope
                if self.dimension() == 2:
                    # Find bounds for the grid
                    interior_point = self._find_interior_point()
                    if interior_point is None:
                        # Polytope might be empty
                        ax = kwargs.pop('ax', plt.gca())
                        return ax
                    
                    # Create a grid around the interior point
                    margin = 10  # Arbitrary margin
                    x_min, x_max = interior_point[0] - margin, interior_point[0] + margin
                    y_min, y_max = interior_point[1] - margin, interior_point[1] + margin
                    
                    n_points = 100
                    x = np.linspace(x_min, x_max, n_points)
                    y = np.linspace(y_min, y_max, n_points)
                    
                    # Create meshgrid
                    xx, yy = np.meshgrid(x, y)
                    points = np.column_stack([xx.ravel(), yy.ravel()])
                    
                    # Check which points are inside the polytope
                    inside = np.all(self.A @ points.T <= self.b.reshape(-1, 1) + 1e-10, axis=0)
                    
                    # If no points are inside, the polytope might be empty or very small
                    if not np.any(inside):
                        ax = kwargs.pop('ax', plt.gca())
                        return ax
                    
                    # Use convex hull to find the boundary
                    try:
                        inside_points = points[inside]
                        hull = ConvexHull(inside_points)
                        self._vertices = inside_points[hull.vertices]
                    except:
                        # If convex hull fails, just plot the inside points
                        ax = kwargs.pop('ax', plt.gca())
                        ax.scatter(points[inside, 0], points[inside, 1], **kwargs)
                        return ax
                else:
                    # For higher dimensions, we can't easily visualize without vertices
                    ax = kwargs.pop('ax', plt.gca())
                    return ax
            else:
                # If we have neither H-representation nor enough vertices, just return
                ax = kwargs.pop('ax', plt.gca())
                return ax
        
        # Project vertices to the specified dimensions
        if self._vertices is not None and len(self._vertices) > 0:
            vertices_2d = self._vertices[:, dims]
            
            # Create a figure if needed
            ax = kwargs.pop('ax', plt.gca())
            
            # Plot the polytope as a filled polygon
            fc = kwargs.pop('facecolor', kwargs.pop('fc', 'blue'))
            ec = kwargs.pop('edgecolor', kwargs.pop('ec', 'black'))
            alpha = kwargs.pop('alpha', 0.5)
            
            # For 2D, plot as a polygon
            if len(vertices_2d) >= 3:
                polygon = Polygon(vertices_2d, facecolor=fc, edgecolor=ec, alpha=alpha, **kwargs)
                ax.add_patch(polygon)
            else:
                # Not enough vertices for a polygon, plot as points
                ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1], color=fc, **kwargs)
            
            # Update limits
            margin = 0.1
            min_vals = np.min(vertices_2d, axis=0) - margin
            max_vals = np.max(vertices_2d, axis=0) + margin
            
            ax.set_xlim(min_vals[0], max_vals[0])
            ax.set_ylim(min_vals[1], max_vals[1])
            
            # Set equal aspect ratio for better visualization
            ax.set_aspect('equal')
            
            return ax
        
        # If we get here, we don't have a way to visualize the polytope
        ax = kwargs.pop('ax', plt.gca())
        return ax
    
    def center(self):
        """
        Return the center of the polytope.
        
        Returns:
            numpy.ndarray: The center point of the polytope
        """
        # If we have vertices, use their average
        if self._vertices is not None and len(self._vertices) > 0:
            return np.mean(self._vertices, axis=0)
        
        # Otherwise, find a point inside using the Chebyshev center or analytic center
        if self.A is not None and self.b is not None:
            interior_point = self._find_interior_point()
            if interior_point is not None:
                return interior_point
        
        # If all else fails, return the origin
        return np.zeros(self.dimension())
    
    def volume(self):
        """
        Calculate the volume of the polytope.
        
        Returns:
            float: The volume of the polytope
        """
        # Make sure we have vertices
        if self._vertices is None:
            self._compute_vertices_from_h_rep()
        
        if self._vertices is None or len(self._vertices) == 0:
            return 0
        
        # For low dimensions, use scipy's ConvexHull
        if self.dimension() <= 3:
            try:
                hull = ConvexHull(self._vertices)
                return hull.volume
            except:
                return 0
        
        # For higher dimensions, calculating the exact volume is challenging
        # Return an approximation using the determinant of the covariance matrix
        try:
            # Center the vertices
            centered = self._vertices - np.mean(self._vertices, axis=0)
            
            # Compute covariance matrix
            cov = centered.T @ centered / len(centered)
            
            # Volume is proportional to sqrt(det(cov))
            return np.sqrt(np.linalg.det(cov)) * np.pi**(self.dimension()/2) / np.math.gamma(self.dimension()/2 + 1)
        except:
            return 0
    
    def support_function(self, direction, lower=False):
        """
        Compute the support function of the polytope in the given direction.
        
        Args:
            direction (numpy.ndarray): The direction vector
            lower (bool): If True, compute the lower bound rather than upper bound
            
        Returns:
            float: The value of the support function
        """
        direction = np.array(direction, dtype=float).flatten()
        
        if len(direction) != self.dimension():
            raise ValueError(f"Dimension mismatch: polytope has dimension {self.dimension()}, "
                            f"but direction has dimension {len(direction)}")
        
        # If the direction is near zero, return 0
        if np.allclose(direction, 0):
            return 0
        
        # For V-representation, compute the dot product with all vertices
        if self._vertices is not None and len(self._vertices) > 0:
            projections = self._vertices @ direction
            if lower:
                return np.min(projections)
            else:
                return np.max(projections)
        
        # For H-representation, solve a linear program
        if self.A is not None and self.b is not None:
            # Set up the LP: max c^T x s.t. Ax <= b
            c = direction if not lower else -direction
            
            # Solve LP
            result = linprog(-c, A_ub=self.A, b_ub=self.b, method='highs')
            
            if result.success:
                return c @ result.x
            else:
                # Polytope might be empty or unbounded
                return float('inf') if not lower else float('-inf')
        
        # If neither representation is available, return NaN
        return float('nan')
    
    def vertices(self):
        """
        Get the vertices of the polytope.
        
        Returns:
            numpy.ndarray: The vertices of the polytope
        """
        if self._vertices is None:
            self._compute_vertices_from_h_rep()
        
        if self._vertices is None:
            return np.zeros((0, self.dimension()))
        
        return self._vertices
    
    def __and__(self, other):
        """
        Compute the intersection of this polytope with another set.
        
        Args:
            other (Set): The other set
            
        Returns:
            Polytope: The intersection
        """
        if isinstance(other, Polytope):
            # Combine the constraints from both polytopes
            if self._rep_type.startswith('H') and other._rep_type.startswith('H'):
                A_combined = np.vstack([self.A, other.A])
                b_combined = np.hstack([self.b, other.b])
                
                # Check if the intersection is empty
                n_vars = self.dimension()
                c = np.zeros(n_vars)  # Any objective function will do
                
                result = linprog(c, A_ub=A_combined, b_ub=b_combined, method='highs')
                
                if not result.success:
                    # Empty intersection
                    return Polytope(np.eye(self.dimension()), np.zeros(self.dimension()) - 1)
                
                return Polytope(A_combined, b_combined)
            
            # If we don't have H-representation for both, compute it
            if not self._rep_type.startswith('H'):
                self._compute_h_rep_from_vertices()
            
            if not other._rep_type.startswith('H'):
                other._compute_h_rep_from_vertices()
            
            # Try again
            return self.__and__(other)
        
        # For other set types, try to use the other set's intersection method
        try:
            return other.__and__(self)
        except:
            raise NotImplementedError(f"Intersection with {type(other)} not implemented")
    
    def __or__(self, other):
        """
        Compute the convex hull of this polytope and another set.
        
        Args:
            other (Set): The other set
            
        Returns:
            Polytope: The convex hull
        """
        if isinstance(other, Polytope):
            # Get vertices of both polytopes
            if self._vertices is None:
                self._compute_vertices_from_h_rep()
            
            if other._vertices is None:
                other._compute_vertices_from_h_rep()
            
            if self._vertices is not None and other._vertices is not None:
                # Combine vertices and compute convex hull
                combined_vertices = np.vstack([self._vertices, other._vertices])
                
                # Remove duplicate vertices
                combined_vertices = np.unique(combined_vertices, axis=0)
                
                return Polytope(vertices=combined_vertices)
            
            # If we don't have vertices for one or both, use H-representation
            if self._rep_type.startswith('H') and other._rep_type.startswith('H'):
                # TODO: Implement convex hull for H-representation
                # This is challenging, so fall back to V-representation
                self._compute_vertices_from_h_rep()
                other._compute_vertices_from_h_rep()
                return self.__or__(other)
            
            # If we still don't have what we need, raise an error
            raise ValueError("Could not compute convex hull: missing required representation")
        
        # For other set types, try to use the other set's union method
        try:
            return other.__or__(self)
        except:
            raise NotImplementedError(f"Union with {type(other)} not implemented")
    
    def project(self, dims):
        """
        Project the polytope onto specified dimensions.
        
        Args:
            dims (list): The dimensions to project onto
            
        Returns:
            Polytope: The projected polytope
        """
        dims = np.array(dims, dtype=int)
        
        if np.any(dims < 0) or np.any(dims >= self.dimension()):
            raise ValueError(f"Invalid dimensions: {dims}. Must be between 0 and {self.dimension()-1}")
        
        # If we have vertices, project them directly
        if self._vertices is not None:
            projected_vertices = self._vertices[:, dims]
            
            # Remove duplicate vertices
            projected_vertices = np.unique(projected_vertices, axis=0)
            
            return Polytope(vertices=projected_vertices)
        
        # If we have H-representation, the projection is more complex
        # For now, compute vertices, then project them
        self._compute_vertices_from_h_rep()
        return self.project(dims)
    
    def linear_map(self, matrix):
        """
        Apply a linear transformation to the polytope.
        
        Args:
            matrix (numpy.ndarray): The transformation matrix
            
        Returns:
            Polytope: The transformed polytope
        """
        matrix = np.array(matrix, dtype=float)
        
        if matrix.shape[1] != self.dimension():
            raise ValueError(f"Dimension mismatch: matrix has input dimension {matrix.shape[1]}, "
                           f"but polytope has dimension {self.dimension()}")
        
        # For V-representation, transform the vertices
        # For V-representation, transform the vertices
        if self._vertices is not None:
            transformed_vertices = self._vertices @ matrix.T
            return Polytope(vertices=transformed_vertices)
        
        # For H-representation, the transformation is more complex
        # If the matrix is invertible, we can transform the constraints
        if self.A is not None and self.b is not None:
            try:
                if matrix.shape[0] == matrix.shape[1]:  # Square matrix
                    # Check if matrix is invertible
                    inv_matrix = np.linalg.inv(matrix)
                    
                    # Transform H-representation: Ax <= b becomes A*inv(M)*M*x <= b
                    transformed_A = self.A @ inv_matrix
                    transformed_b = self.b.copy()
                    
                    return Polytope(transformed_A, transformed_b)
            except np.linalg.LinAlgError:
                # Matrix is not invertible
                pass
        
        # Otherwise, compute vertices and transform them
        self._compute_vertices_from_h_rep()
        return self.linear_map(matrix)
    
    def minkowski_sum(self, other_set):
        """
        Compute the Minkowski sum of this polytope and another set.
        
        Args:
            other_set: The other set
            
        Returns:
            Set: The Minkowski sum
        """
        if isinstance(other_set, Polytope):
            # Get vertices of both polytopes
            if self._vertices is None:
                self._compute_vertices_from_h_rep()
            
            if other_set._vertices is None:
                other_set._compute_vertices_from_h_rep()
            
            if self._vertices is not None and other_set._vertices is not None:
                # Compute all possible sums of vertices
                result_vertices = []
                for v1 in self._vertices:
                    for v2 in other_set._vertices:
                        result_vertices.append(v1 + v2)
                
                # Compute convex hull of all sums
                result_vertices = np.array(result_vertices)
                
                # We only need the vertices of the convex hull
                try:
                    hull = ConvexHull(result_vertices)
                    hull_vertices = result_vertices[hull.vertices]
                    return Polytope(vertices=hull_vertices)
                except:
                    # If convex hull fails, use all vertices
                    return Polytope(vertices=result_vertices)
            
            # If we don't have vertices for one or both, use H-representation
            # This is more complex, so for now, compute vertices first
            self._compute_vertices_from_h_rep()
            other_set._compute_vertices_from_h_rep()
            return self.minkowski_sum(other_set)
        
        # For other set types, try to use the other set's Minkowski sum method
        try:
            return other_set.minkowski_sum(self)
        except:
            raise NotImplementedError(f"Minkowski sum with {type(other_set)} not implemented")
    
    def __add__(self, other):
        """
        Overloaded + operator for Minkowski sum.
        
        Args:
            other: The other set or point
            
        Returns:
            Set: The Minkowski sum
        """
        if isinstance(other, (list, np.ndarray)):
            # Treat as a point/vector
            other_vec = np.array(other, dtype=float).flatten()
            
            if len(other_vec) != self.dimension():
                raise ValueError(f"Dimension mismatch: polytope has dimension {self.dimension()}, "
                               f"but vector has dimension {len(other_vec)}")
            
            # For V-representation, add the vector to each vertex
            if self._vertices is not None:
                shifted_vertices = self._vertices + other_vec
                return Polytope(vertices=shifted_vertices)
            
            # For H-representation, shift the right-hand side
            if self.A is not None and self.b is not None:
                shifted_b = self.b - self.A @ other_vec
                return Polytope(self.A.copy(), shifted_b)
            
            raise ValueError("Polytope has neither H-representation nor V-representation")
        
        return self.minkowski_sum(other)
    
    def __mul__(self, other):
        """
        Overloaded * operator for linear transformation or scaling.
        
        Args:
            other: Either a matrix for linear transformation or a scalar for scaling
            
        Returns:
            Polytope: The transformed polytope
        """
        if isinstance(other, (int, float)):
            # Scaling by a scalar
            if other == 0:
                # Return a single point at the origin
                return Polytope(vertices=np.zeros((1, self.dimension())))
            
            # For V-representation, scale the vertices
            if self._vertices is not None:
                scaled_vertices = other * self._vertices
                return Polytope(vertices=scaled_vertices)
            
            # For H-representation, scale the right-hand side and inverse-scale the matrix
            if self.A is not None and self.b is not None:
                if other > 0:
                    # Positive scaling: scale b and divide A by scalar
                    scaled_A = self.A / other
                    scaled_b = self.b.copy()
                else:
                    # Negative scaling: scale b and divide A by scalar, then negate both
                    scaled_A = -self.A / (-other)
                    scaled_b = -self.b.copy()
                
                return Polytope(scaled_A, scaled_b)
            
            raise ValueError("Polytope has neither H-representation nor V-representation")
        else:
            # Linear transformation
            return self.linear_map(other)
    
    def __rmul__(self, other):
        """
        Right multiplication for scalar scaling.
        
        Args:
            other: A scalar for scaling
            
        Returns:
            Polytope: The scaled polytope
        """
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise TypeError(f"Unsupported operand type for *: {type(other)}")
    
    def enlarge(self, factor):
        """
        Enlarge the polytope by scaling it around its center.
        
        Args:
            factor (float): The scaling factor
            
        Returns:
            Polytope: The enlarged polytope
        """
        if factor < 0:
            raise ValueError(f"Enlargement factor must be non-negative, got {factor}")
        
        # Get the center
        c = self.center()
        
        # Translate to origin, scale, and translate back
        return (self + (-c)) * factor + c
    
    def boundary(self):
        """
        Compute the boundary of the polytope.
        
        Note: This returns the vertices of the polytope, which define its boundary.
        
        Returns:
            numpy.ndarray: The vertices of the polytope
        """
        return self.vertices()
    
    def is_full_dim(self):
        """
        Check if the polytope is full-dimensional.
        
        Returns:
            bool: True if the polytope is full-dimensional, False otherwise
        """
        # Check if H-representation defines a full-dimensional polytope
        if self.A is not None and self.b is not None:
            # Compute rank of A
            rank = np.linalg.matrix_rank(self.A)
            return rank == self.dimension()
        
        # Check if V-representation defines a full-dimensional polytope
        if self._vertices is not None:
            # Check if vertices span the full space
            if len(self._vertices) <= self.dimension():
                return False
            
            # Compute the affine hull
            centered = self._vertices - self._vertices[0]
            rank = np.linalg.matrix_rank(centered)
            return rank == self.dimension()
        
        return False
    
    def cartesian_product(self, other_set):
        """
        Compute the Cartesian product of this polytope and another set.
        
        Args:
            other_set (Set): The other set
            
        Returns:
            Set: The Cartesian product
        """
        if isinstance(other_set, Polytope):
            n1 = self.dimension()
            n2 = other_set.dimension()
            
            # For H-representation, combine the constraints
            if self._rep_type.startswith('H') and other_set._rep_type.startswith('H'):
                # New constraint matrices have block diagonal structure
                A1 = self.A
                b1 = self.b
                A2 = other_set.A
                b2 = other_set.b
                
                # Build combined constraints
                A_combined = np.zeros((A1.shape[0] + A2.shape[0], n1 + n2))
                A_combined[:A1.shape[0], :n1] = A1
                A_combined[A1.shape[0]:, n1:] = A2
                
                b_combined = np.hstack([b1, b2])
                
                return Polytope(A_combined, b_combined)
            
            # For V-representation, compute the Cartesian product of vertices
            if self._vertices is not None and other_set._vertices is not None:
                # Create all combinations of vertices
                v1 = self._vertices
                v2 = other_set._vertices
                
                # Cartesian product of vertices
                v_prod = np.zeros((len(v1) * len(v2), n1 + n2))
                
                idx = 0
                for i in range(len(v1)):
                    for j in range(len(v2)):
                        v_prod[idx, :n1] = v1[i]
                        v_prod[idx, n1:] = v2[j]
                        idx += 1
                
                return Polytope(vertices=v_prod)
            
            # If we don't have consistent representations, compute them
            if self._vertices is None:
                self._compute_vertices_from_h_rep()
            
            if other_set._vertices is None:
                other_set._compute_vertices_from_h_rep()
            
            # Try again
            return self.cartesian_product(other_set)
        
        # For other set types, use a generic approach
        try:
            if isinstance(other_set, contSet):
                # Create a projection matrix for each dimension
                P1 = np.eye(self.dimension() + other_set.dimension(), self.dimension())
                P2 = np.hstack([np.zeros((other_set.dimension(), self.dimension())), 
                               np.eye(other_set.dimension())])
                
                # Compute the intersection of two half-spaces
                # x = P1*z ∈ self and P2*z ∈ other_set
                # This requires more specialized implementation for each set type
                # For now, use a simpler approach for known set types
                raise NotImplementedError("Generic Cartesian product not implemented")
            else:
                raise TypeError(f"Expected a Set, got {type(other_set)}")
        except:
            raise NotImplementedError(f"Cartesian product with {type(other_set)} not implemented")
    
    def convex_hull(self, other_set):
        """
        Compute the convex hull of this polytope and another set.
        
        Args:
            other_set (Set): The other set
            
        Returns:
            Set: The convex hull
        """
        return self.__or__(other_set)
    
    def randPoint(self, n=1, type='standard'):
        """
        Generate random points inside the polytope.
        
        Args:
            n (int): Number of points to generate
            type (str): Type of random points:
                'standard': Uniform random points
                'extreme': Points near the boundary
            
        Returns:
            numpy.ndarray: Matrix of random points (one per row)
        """
        if n <= 0:
            raise ValueError(f"Number of points must be positive, got {n}")
        
        # For polytopes, generating truly uniform random points is challenging
        # Use a simple rejection sampling method for now
        
        # Get a bounding box
        if self._vertices is None:
            self._compute_vertices_from_h_rep()
        
        if self._vertices is None or len(self._vertices) == 0:
            # Empty polytope or no vertices
            return np.zeros((n, self.dimension()))
        
        # Compute bounding box
        min_corner = np.min(self._vertices, axis=0)
        max_corner = np.max(self._vertices, axis=0)
        
        # Generate points using rejection sampling
        points = []
        max_attempts = 1000 * n  # Limit the number of attempts
        
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            # Generate a random point in the bounding box
            point = min_corner + np.random.rand(self.dimension()) * (max_corner - min_corner)
            
            # Check if the point is inside the polytope
            if self.contains(point):
                points.append(point)
            
            attempts += 1
        
        # If we couldn't generate enough points, duplicate the last one
        while len(points) < n:
            if len(points) > 0:
                points.append(points[-1])
            else:
                # If we couldn't generate any points, use the center
                points.append(self.center())
        
        return np.array(points)
    
    @staticmethod
    def generateRandom(dimension=None, n_halfspaces=None):
        """
        Generate a random polytope.
        
        Args:
            dimension (int, optional): Dimension of the polytope
            n_halfspaces (int, optional): Number of halfspaces
            
        Returns:
            Polytope: A random polytope
        """
        if dimension is None:
            dimension = np.random.randint(1, 6)
        
        if n_halfspaces is None:
            n_halfspaces = np.random.randint(dimension + 1, 2 * dimension + 1)
        
        # First, generate a unit ball and then add random halfspaces
        # Get random normal vectors for the halfspaces
        normals = np.random.randn(n_halfspaces, dimension)
        
        # Normalize the normal vectors
        norms = np.sqrt(np.sum(normals * normals, axis=1)).reshape(-1, 1)
        normals = normals / norms
        
        # Generate points at random distances from the origin
        distances = 1 + 0.5 * np.random.rand(n_halfspaces)
        
        # Create the halfspace representation
        A = normals
        b = distances
        
        return Polytope(A, b)
    
    def __str__(self):
        """String representation of the polytope."""
        n_constraints = 0 if self.A is None else self.A.shape[0]
        n_vertices = 0 if self._vertices is None else len(self._vertices)
        
        if self._rep_type == 'H':
            return f"Polytope(dim={self.dimension()}, constraints={n_constraints})"
        elif self._rep_type == 'V':
            return f"Polytope(dim={self.dimension()}, vertices={n_vertices})"
        else:
            return f"Polytope(dim={self.dimension()}, constraints={n_constraints}, vertices={n_vertices})"
    
    def __repr__(self):
        """Detailed string representation of the polytope."""
        return self.__str__()
    
    def copy(self):
        """Create a deep copy of the polytope."""
        if self._rep_type.startswith('H'):
            return Polytope(self.A.copy(), self.b.copy())
        elif self._rep_type == 'V':
            return Polytope(vertices=self._vertices.copy())
        else:
            # Both representations are available
            result = Polytope(self.A.copy(), self.b.copy())
            result._vertices = self._vertices.copy()
            result._rep_type = self._rep_type
            return result