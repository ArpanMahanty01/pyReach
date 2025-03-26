import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union, Mapping

from pyReach.core.AxisSet import AxisSet
from pyReach.core.Polytope import Polytope
from pyReach.core.Reachability import Reachability
from pyReach.dynamics.LinearAffineSystem import LinearAffineSystem
from pyReach.network.Agent import Agent


class NetworkManager:
    def __init__(self, agents: Dict[int, Agent]):
        self.agents = agents
        self.communication_graph = {}
        self.iteration_count = 0
        self.converged = False
    
    def setupConnections(self):
        for agent_id, agent in self.agents.items():
            self.communication_graph[agent_id] = agent.neighbors
    
    def runAlgorithm(self, max_iterations: int):
        self.iteration_count = 0
        self.converged = False
        
        # Solve local reachability problems
        for agent in self.agents.values():
            agent.solveLocalReachability()
        
        # Initialize old sets
        old_sets = {agent_id: agent.local_reachable_set for agent_id, agent in self.agents.items()}
        
        # Iterate until convergence or max iterations
        for iteration in range(max_iterations):
            self.iteration_count += 1
            
            # Exchange data between neighbors
            exchange_data = {}
            for agent_id, agent in self.agents.items():
                exchange_data[agent_id] = agent.sendDataToNeighbors()
            
            # Process received data
            for agent_id, agent in self.agents.items():
                received_data = {}
                for neighbor_id in agent.neighbors:
                    if neighbor_id in exchange_data and agent_id in exchange_data[neighbor_id]:
                        received_data[neighbor_id] = exchange_data[neighbor_id][agent_id]
                agent.receiveDataFromNeighbors(received_data)
            
            # Process iteration
            for agent in self.agents.values():
                agent.processIteration(agent.neighbor_reachable_sets)
            
            # Check for convergence
            new_sets = {agent_id: agent.local_reachable_set for agent_id, agent in self.agents.items()}
            
            if any(agent.reachability_problem.checkConvergence(old_sets, new_sets) for agent in self.agents.values()):
                self.converged = True
                break
            
            old_sets = new_sets
        
        # Extract final results
        for agent in self.agents.values():
            agent.extractBackwardReachableSet()
            agent.extractAdmissibleControlSequence()
    
    def getResults(self) -> Dict[int, Polytope]:
        return {agent_id: agent.backward_reachable_set for agent_id, agent in self.agents.items()}
    
    def getAdmissibleControlSequences(self) -> Dict[int, Polytope]:
        return {agent_id: agent.admissible_control_sequence for agent_id, agent in self.agents.items()}
    
    def __str__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return f"NetworkManager(agents={len(self.agents)}, iterations={self.iteration_count}, status={status})"
    
    def __repr__(self) -> str:
        return str(self)