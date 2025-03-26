import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union, Mapping

from pyReach.core.AxisSet import AxisSet
from pyReach.core.Polytope import Polytope
from pyReach.dynamics import LinearAffineSystem
from pyReach.core import Reachability


class Agent:
    def __init__(self, agent_id: int, neighbors: Set[int], state_axes: AxisSet, input_axes: AxisSet,
                 system: LinearAffineSystem, state_constraints: Polytope, input_constraints: Polytope,
                 target_set: Polytope):
        self.id = agent_id
        self.neighbors = neighbors
        self.state_axes = state_axes
        self.input_axes = input_axes
        self.system = system
        self.state_constraints = state_constraints
        self.input_constraints = input_constraints
        self.target_set = target_set
        
        self.reachability_problem = None
        self.local_reachable_set = None
        self.backward_reachable_set = None
        self.admissible_control_sequence = None
        
        self.neighbor_reachable_sets = {}
    
    def setupReachabilityProblem(self, problem: Reachability):
        self.reachability_problem = problem
    
    def solveLocalReachability(self) -> Polytope:
        if not self.reachability_problem:
            raise ValueError(f"Agent {self.id}: Reachability problem not set up")
        
        self.local_reachable_set = self.reachability_problem.solve()
        return self.local_reachable_set
    
    def processIteration(self, neighbor_sets: Dict[int, Polytope]) -> Polytope:
        if not self.reachability_problem:
            raise ValueError(f"Agent {self.id}: Reachability problem not set up")
        
        self.neighbor_reachable_sets = neighbor_sets
        updated_sets = self.reachability_problem.iterate(neighbor_sets)
        
        if self.id in updated_sets:
            self.local_reachable_set = updated_sets[self.id]
        
        return self.local_reachable_set
    
    def extractBackwardReachableSet(self) -> Polytope:
        if self.local_reachable_set is None:
            raise ValueError(f"Agent {self.id}: Local reachable set not computed")
        
        self.backward_reachable_set = self.local_reachable_set.project(self.state_axes.indices)
        return self.backward_reachable_set
    
    def extractAdmissibleControlSequence(self) -> Polytope:
        if self.local_reachable_set is None:
            raise ValueError(f"Agent {self.id}: Local reachable set not computed")
        
        combined_axes = self.state_axes.union(self.input_axes)
        self.admissible_control_sequence = self.local_reachable_set.project(combined_axes.indices)
        return self.admissible_control_sequence
    
    def sendDataToNeighbors(self) -> Dict[int, Polytope]:
        return {neighbor: self.local_reachable_set for neighbor in self.neighbors}
    
    def receiveDataFromNeighbors(self, data: Dict[int, Polytope]):
        self.neighbor_reachable_sets.update(data)
    
    def __str__(self) -> str:
        return f"Agent(id={self.id}, neighbors={self.neighbors})"
    
    def __repr__(self) -> str:
        return str(self)