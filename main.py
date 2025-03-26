import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple

from pyReach.core.AxisSet import AxisSet
from pyReach.core.Polytope import Polytope
from pyReach.core.Reachability import Reachability
from pyReach.dynamics.LinearAffineSystem import LinearAffineSystem
from pyReach.network.Agent import Agent
from pyReach.network.NetworkManager import NetworkManager

def main():

    A11 = np.array([[0.8]])
    A12 = np.array([[0.1]])
    A13 = np.array([[0.0]])
    B11 = np.array([[1.0]])
    B12 = np.array([[0.0]])
    B13 = np.array([[0.0]])
    K1 = np.array([0.0])
    
    A_dict1 = {1: A11, 2: A12, 3: A13}
    B_dict1 = {1: B11, 2: B12, 3: B13}
    system1 = LinearAffineSystem(1, A_dict1, B_dict1, K1)
    
    A21 = np.array([[0.1]])
    A22 = np.array([[0.7]])
    A23 = np.array([[0.1]])
    B21 = np.array([[0.0]])
    B22 = np.array([[1.0]])
    B23 = np.array([[0.0]])
    K2 = np.array([0.0])
    
    A_dict2 = {1: A21, 2: A22, 3: A23}
    B_dict2 = {1: B21, 2: B22, 3: B23}
    system2 = LinearAffineSystem(2, A_dict2, B_dict2, K2)
    
    A31 = np.array([[0.0]])
    A32 = np.array([[0.1]])
    A33 = np.array([[0.8]])
    B31 = np.array([[0.0]])
    B32 = np.array([[0.0]])
    B33 = np.array([[1.0]])
    K3 = np.array([0.0])
    
    A_dict3 = {1: A31, 2: A32, 3: A33}
    B_dict3 = {1: B31, 2: B32, 3: B33}
    system3 = LinearAffineSystem(3, A_dict3, B_dict3, K3)
    
    A_state = np.array([[1.0], [-1.0]])
    b_state = np.array([5.0, 5.0])
    state_constraints = Polytope(A_state, b_state)
    
    # Input constraints: -1 ≤ u ≤ 1
    A_input = np.array([[1.0], [-1.0]])
    b_input = np.array([1.0, 1.0])
    input_constraints = Polytope(A_input, b_input)
    
    # Target sets: -0.5 ≤ x ≤ 0.5
    A_target = np.array([[1.0], [-1.0]])
    b_target = np.array([0.5, 0.5])
    target_set = Polytope(A_target, b_target)
    
    state_axis1 = AxisSet({0})
    state_axis2 = AxisSet({0}) 
    state_axis3 = AxisSet({0})
    
    input_axis1 = AxisSet({0})
    input_axis2 = AxisSet({0})
    input_axis3 = AxisSet({0})
    
    class ModifiedReachability(Reachability):
        def iterate(self, neighbor_sets: Dict[int, Polytope]) -> Dict[int, Polytope]:
            return {self.system.id: self.local_reachable_set}
            return {self.system.id: self.local_reachable_set}
    
    class ModifiedAgent(Agent):
        def extractBackwardReachableSet(self) -> Polytope:
            self.backward_reachable_set = self.local_reachable_set
            return self.backward_reachable_set
        
        def extractAdmissibleControlSequence(self) -> Polytope:
            self.admissible_control_sequence = self.local_reachable_set
            return self.admissible_control_sequence
    
    agent1 = ModifiedAgent(
        agent_id=1,
        neighbors={2},
        state_axes=state_axis1,
        input_axes=input_axis1,
        system=system1,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        target_set=target_set
    )
    
    agent2 = ModifiedAgent(
        agent_id=2,
        neighbors={1, 3},
        state_axes=state_axis2,
        input_axes=input_axis2,
        system=system2,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        target_set=target_set
    )
    
    agent3 = ModifiedAgent(
        agent_id=3,
        neighbors={2},
        state_axes=state_axis3,
        input_axes=input_axis3,
        system=system3,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        target_set=target_set
    )
    
    horizon = 3
    
    reachability1 = ModifiedReachability(
        system=system1,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        target_set=target_set,
        horizon=horizon
    )
    reachability1.local_reachable_set = reachability1.solve()
    
    reachability2 = ModifiedReachability(
        system=system2,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        target_set=target_set,
        horizon=horizon
    )
    reachability2.local_reachable_set = reachability2.solve()
    
    reachability3 = ModifiedReachability(
        system=system3,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        target_set=target_set,
        horizon=horizon
    )
    reachability3.local_reachable_set = reachability3.solve()
    
    agent1.setupReachabilityProblem(reachability1)
    agent2.setupReachabilityProblem(reachability2)
    agent3.setupReachabilityProblem(reachability3)
    
    agents = {1: agent1, 2: agent2, 3: agent3}
    network = NetworkManager(agents)
    network.setupConnections()
    
    max_iterations = 10
    print(f"Running distributed backward reachability algorithm with horizon {horizon}...")
    network.runAlgorithm(max_iterations)
    
    backward_reachable_sets = network.getResults()
    
    print(f"\nAlgorithm converged after {network.iteration_count} iterations.")
    print("\nBackward Reachable Sets:")
    for agent_id, reachable_set in backward_reachable_sets.items():
        print(f"Agent {agent_id}: {reachable_set}")
    
    x1_points = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(15, 5))
    
    for i, (agent_id, reachable_set) in enumerate(backward_reachable_sets.items()):
        plt.subplot(1, 3, i+1)
        
        constraints = reachable_set.A
        bounds = reachable_set.b
        
        if constraints.shape[0] > 1 and constraints.shape[1] > 0:
            lower_bound = -bounds[1] if constraints[1, 0] < 0 else -float('inf')
            upper_bound = bounds[0] if constraints[0, 0] > 0 else float('inf')
        else:
            lower_bound = -5
            upper_bound = 5
        
        target_lower = -0.5
        target_upper = 0.5
        plt.axvspan(target_lower, target_upper, alpha=0.3, color='green', label='Target Set')
        
        plt.axvspan(lower_bound, upper_bound, alpha=0.3, color='blue', label='Backward Reachable Set')
        
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Agent {agent_id}')
        plt.xlim(-5, 5)
        plt.ylim(-0.1, 1.1)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('backward_reachable_sets.png')
    plt.show()
    
    print("Results saved to 'backward_reachable_sets.png'")

if __name__ == "__main__":
    main()