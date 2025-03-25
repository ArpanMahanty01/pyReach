# network.py

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx


class Network:
    """
    Represents a network of interconnected agents for distributed reachability analysis.
    
    This class manages the network structure, agent connections, and provides methods
    for computing communication neighborhoods.
    """
    
    def __init__(self):
        """Initialize an empty network."""
        self.agents = {}  # Dictionary mapping agent_id to Agent object
        self.dynamics_graph = nx.DiGraph()  # Directed graph for dynamics connections
        self.constraint_graph = nx.DiGraph()  # Directed graph for constraint connections
        self.communication_graph = nx.Graph()  # Undirected graph for communication
    
    def add_agent(self, agent_id: int, agent: Any) -> None:
        """
        Add an agent to the network.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent object
        """
        if agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent_id} already exists in the network")
        
        self.agents[agent_id] = agent
        self.dynamics_graph.add_node(agent_id)
        self.constraint_graph.add_node(agent_id)
        self.communication_graph.add_node(agent_id)
    
    def add_dynamics_connection(self, from_agent: int, to_agent: int) -> None:
        """
        Add a dynamics connection between agents (from_agent influences to_agent).
        
        Args:
            from_agent: ID of the influencing agent
            to_agent: ID of the influenced agent
        """
        if from_agent not in self.agents or to_agent not in self.agents:
            raise ValueError("Both agents must be in the network")
        
        self.dynamics_graph.add_edge(from_agent, to_agent)
        
        # Update the communication graph
        self.communication_graph.add_edge(from_agent, to_agent)
    
    def add_constraint_connection(self, from_agent: int, to_agent: int) -> None:
        """
        Add a constraint connection between agents (from_agent constrains to_agent).
        
        Args:
            from_agent: ID of the constraining agent
            to_agent: ID of the constrained agent
        """
        if from_agent not in self.agents or to_agent not in self.agents:
            raise ValueError("Both agents must be in the network")
        
        self.constraint_graph.add_edge(from_agent, to_agent)
        
        # Update the communication graph
        self.communication_graph.add_edge(from_agent, to_agent)
    
    def get_dynamics_neighbors(self, agent_id: int) -> Set[int]:
        """
        Get the neighbors of an agent in the dynamics graph (N_X,i in the paper).
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Set of agent IDs that influence the dynamics of the specified agent
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} does not exist in the network")
        
        # Get the predecessors (agents that influence this agent)
        return set(self.dynamics_graph.predecessors(agent_id))
    
    def get_constraint_neighbors(self, agent_id: int) -> Set[int]:
        """
        Get the neighbors of an agent in the constraint graph (N_I,i in the paper).
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Set of agent IDs that constrain the specified agent
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} does not exist in the network")
        
        # Get the predecessors (agents that constrain this agent)
        return set(self.constraint_graph.predecessors(agent_id))
    
    def get_neighborhood(self, agent_id: int) -> Set[int]:
        """
        Get the complete neighborhood of an agent (N_i in the paper).
        
        The neighborhood includes all agents that influence the dynamics (N_X,i),
        all agents involved in constraints (N_I,i), and the agent itself.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Set of agent IDs in the neighborhood
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} does not exist in the network")
        
        dynamics_neighbors = self.get_dynamics_neighbors(agent_id)
        constraint_neighbors = self.get_constraint_neighbors(agent_id)
        
        # Combine all neighbors and add the agent itself
        return dynamics_neighbors.union(constraint_neighbors).union({agent_id})
    
    def get_communication_neighbors(self, agent_id: int) -> Set[int]:
        """
        Get the neighbors of an agent in the communication graph (M_i in the paper).
        
        The communication graph is undirected and includes an edge between two agents
        if either one influences the other through dynamics or constraints.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Set of agent IDs that are neighbors in the communication graph
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} does not exist in the network")
        
        # Get neighbors in the undirected communication graph
        # Also include the agent itself (i ∈ M_i)
        return set(self.communication_graph.neighbors(agent_id)).union({agent_id})
    
    def get_agent(self, agent_id: int) -> Any:
        """
        Get an agent from the network.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent object
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} does not exist in the network")
        
        return self.agents[agent_id]
    
    def get_agents(self) -> Dict[int, Any]:
        """
        Get all agents in the network.
        
        Returns:
            Dictionary mapping agent IDs to Agent objects
        """
        return self.agents
    
    def get_agent_ids(self) -> List[int]:
        """
        Get the IDs of all agents in the network.
        
        Returns:
            List of agent IDs
        """
        return list(self.agents.keys())
    
    def get_num_agents(self) -> int:
        """
        Get the number of agents in the network.
        
        Returns:
            Number of agents
        """
        return len(self.agents)
    
    def get_overall_state_dimension(self) -> int:
        """
        Get the overall state dimension of the network.
        
        Returns:
            Total state dimension (sum of all agent state dimensions)
        """
        return sum(agent.state_dim for agent in self.agents.values())
    
    def get_overall_input_dimension(self) -> int:
        """
        Get the overall input dimension of the network.
        
        Returns:
            Total input dimension (sum of all agent input dimensions)
        """
        return sum(agent.input_dim for agent in self.agents.values())
    
    def get_overall_dimensions(self) -> Tuple[int, int]:
        """
        Get the overall dimensions of the network.
        
        Returns:
            Tuple of (state_dimension, input_dimension)
        """
        return (self.get_overall_state_dimension(), self.get_overall_input_dimension())
    
    def __str__(self) -> str:
        """String representation of the network."""
        return f"Network with {self.get_num_agents()} agents"
    
    def __repr__(self) -> str:
        """Formal string representation of the network."""
        return self.__str__()