from typing import List, Set


class AxisSet:
    """
    Representation of an axis set as defined in Definition 1 of the paper.
    
    An axis set is a finite set B ⊂ N+ that can be written as B = {β₁, β₂, ..., βq}
    such that βᵢ < βⱼ iff i < j.
    """
    
    def __init__(self, indices: List[int]):
        """
        Initialize an axis set with a list of indices.
        
        Args:
            indices: List of indices in ascending order
        """
        # Ensure indices are sorted
        if not all(indices[i] < indices[i+1] for i in range(len(indices)-1)):
            indices.sort()
        self.indices = indices
    
    def __str__(self) -> str:
        """String representation of the axis set."""
        return f"AxisSet({self.indices})"
    
    def __repr__(self) -> str:
        """Formal string representation of the axis set."""
        return f"AxisSet({self.indices})"
    
    def __len__(self) -> int:
        """Get the number of elements in the axis set."""
        return len(self.indices)
    
    def __contains__(self, index: int) -> bool:
        """Check if an index is in the axis set."""
        return index in self.indices
    
    def __iter__(self):
        """Iterator for the axis set."""
        return iter(self.indices)
    
    def union(self, other: 'AxisSet') -> 'AxisSet':
        """
        Compute the union of this axis set with another axis set.
        
        Args:
            other: Another axis set
            
        Returns:
            A new axis set that is the union of the two
        """
        return AxisSet(sorted(set(self.indices).union(set(other.indices))))
    
    def intersection(self, other: 'AxisSet') -> 'AxisSet':
        """
        Compute the intersection of this axis set with another axis set.
        
        Args:
            other: Another axis set
            
        Returns:
            A new axis set that is the intersection of the two
        """
        return AxisSet(sorted(set(self.indices).intersection(set(other.indices))))
    
    def is_empty(self) -> bool:
        """Check if the axis set is empty."""
        return len(self.indices) == 0
    
    @property
    def dimension(self) -> int:
        """Get the dimension (cardinality) of the axis set."""
        return len(self.indices)