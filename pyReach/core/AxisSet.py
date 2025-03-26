import numpy as np
from typing import Set, List, Optional, Union, Tuple


class AxisSet:
    def __init__(self, indices: Set[int]):
        self.indices = set(indices)
        self.dimension = len(self.indices)
        self._sorted_indices = None
    
    @property
    def sorted_indices(self) -> List[int]:
        if self._sorted_indices is None:
            self._sorted_indices = sorted(self.indices)
        return self._sorted_indices
    
    def union(self, other: 'AxisSet') -> 'AxisSet':
        return AxisSet(self.indices.union(other.indices))
    
    def intersection(self, other: 'AxisSet') -> 'AxisSet':
        return AxisSet(self.indices.intersection(other.indices))
    
    def difference(self, other: 'AxisSet') -> 'AxisSet':
        return AxisSet(self.indices.difference(other.indices))
    
    def isSubset(self, other: 'AxisSet') -> bool:
        return self.indices.issubset(other.indices)
    
    def isSuperset(self, other: 'AxisSet') -> bool:
        return self.indices.issuperset(other.indices)
    
    def isEmpty(self) -> bool:
        return len(self.indices) == 0
    
    def to_mask(self, full_dimension: int) -> np.ndarray:
        mask = np.zeros(full_dimension, dtype=bool)
        for idx in self.indices:
            if 0 <= idx < full_dimension:
                mask[idx] = True
        return mask
    
    def get_projection_matrix(self, full_dimension: int) -> np.ndarray:
        if self.isEmpty():
            return np.zeros((0, full_dimension))
        
        matrix = np.zeros((self.dimension, full_dimension))
        for i, idx in enumerate(self.sorted_indices):
            if 0 <= idx < full_dimension:
                matrix[i, idx] = 1
        return matrix
    
    def get_coordinate_map(self, other: 'AxisSet') -> dict:
        if not self.isSubset(other):
            raise ValueError("Current AxisSet must be a subset of the other AxisSet")
        
        other_sorted = other.sorted_indices
        map_dict = {}
        
        for i, idx in enumerate(self.sorted_indices):
            j = other_sorted.index(idx)
            map_dict[i] = j
            
        return map_dict
    
    def get_indices_in_range(self, start: int, end: int) -> 'AxisSet':
        return AxisSet({idx for idx in self.indices if start <= idx < end})
    
    def shift_indices(self, offset: int) -> 'AxisSet':
        return AxisSet({idx + offset for idx in self.indices})
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AxisSet):
            return False
        return self.indices == other.indices
    
    def __hash__(self) -> int:
        return hash(frozenset(self.indices))
    
    def __str__(self) -> str:
        return f"AxisSet({self.sorted_indices})"
    
    def __repr__(self) -> str:
        return str(self)