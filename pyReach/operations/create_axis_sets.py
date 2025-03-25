# axis_operations.py

from typing import Dict, List, Set, Tuple
import numpy as np
from pyReach.core.AxisSet import AxisSet
from pyReach.core.Network import Network


def create_axis_sets(network: Network, horizon: int) -> Dict[str, AxisSet]:
    """
    Create the axis sets defined in Section 3.3 of the paper for a networked system.
    
    This function generates all the axis sets needed for distributed reachability analysis:
    - Time-specific axis sets for each agent's states and inputs
    - Neighborhood axis sets
    - Horizon-specific axis sets
    - Global axis sets
    
    Args:
        network: Network of agents
        horizon: Time horizon H
        
    Returns:
        Dictionary mapping names to axis sets
    """
    axis_sets = {}
    agents = network.get_agent_ids()
    
    # Compute the cumulative dimensions for indexing
    cum_state_dims = [0]
    cum_input_dims = [0]
    
    for agent_id in agents:
        agent = network.get_agent(agent_id)
        cum_state_dims.append(cum_state_dims[-1] + agent.state_dim)
        cum_input_dims.append(cum_input_dims[-1] + agent.input_dim)
    
    # Create the time-step specific axis sets for each agent
    for t in range(horizon + 1):
        for i, agent_id in enumerate(agents):
            agent = network.get_agent(agent_id)
            
            # B̃_x,t,i - Axis set for agent i's state at time t
            B_x_t_i = AxisSet([
                t * (cum_state_dims[-1] + cum_input_dims[-1]) + cum_state_dims[i] + j + 1
                for j in range(agent.state_dim)
            ])
            axis_sets[f'B_x_{t}_{agent_id}'] = B_x_t_i
            
            # B̃_u,t,i - Axis set for agent i's input at time t
            B_u_t_i = AxisSet([
                t * (cum_state_dims[-1] + cum_input_dims[-1]) + cum_state_dims[-1] + cum_input_dims[i] + j + 1
                for j in range(agent.input_dim)
            ])
            axis_sets[f'B_u_{t}_{agent_id}'] = B_u_t_i
            
            # B̃_t,i - Combined state-input axis set for agent i at time t
            axis_sets[f'B_{t}_{agent_id}'] = B_x_t_i.union(B_u_t_i)
    
    # Create the neighborhood axis sets for each time step
    for t in range(horizon + 1):
        for agent_id in agents:
            # Get communication neighbors
            M_i = network.get_communication_neighbors(agent_id)
            
            # B_x,t,i - State axis set for agent i and its neighbors at time t
            B_x_t_M_i = AxisSet([])
            for j in M_i:
                B_x_t_M_i = B_x_t_M_i.union(axis_sets[f'B_x_{t}_{j}'])
            axis_sets[f'B_x_{t}_M_{agent_id}'] = B_x_t_M_i
            
            # B_u,t,i - Input axis set for agent i and its neighbors at time t
            B_u_t_M_i = AxisSet([])
            for j in M_i:
                B_u_t_M_i = B_u_t_M_i.union(axis_sets[f'B_u_{t}_{j}'])
            axis_sets[f'B_u_{t}_M_{agent_id}'] = B_u_t_M_i
            
            # B_t,i - Combined state-input axis set for agent i and its neighbors at time t
            axis_sets[f'B_{t}_M_{agent_id}'] = B_x_t_M_i.union(B_u_t_M_i)
    
    # Create the horizon-specific axis sets
    for agent_id in agents:
        # B^H_x,i - State axis set for agent i over horizon
        B_H_x_i = AxisSet([])
        for t in range(horizon + 1):
            B_H_x_i = B_H_x_i.union(axis_sets[f'B_x_{t}_{agent_id}'])
        axis_sets[f'B_H_x_{agent_id}'] = B_H_x_i
        
        # B^H_u,i - Input axis set for agent i over horizon
        B_H_u_i = AxisSet([])
        for t in range(horizon + 1):
            B_H_u_i = B_H_u_i.union(axis_sets[f'B_u_{t}_{agent_id}'])
        axis_sets[f'B_H_u_{agent_id}'] = B_H_u_i
        
        # B^H_i - Combined state-input axis set for agent i over horizon
        axis_sets[f'B_H_{agent_id}'] = B_H_x_i.union(B_H_u_i)
        
        # Get communication neighbors
        M_i = network.get_communication_neighbors(agent_id)
        
        # B^H_u,M_i - Input axis set for neighbors over horizon
        B_H_u_M_i = AxisSet([])
        for t in range(horizon + 1):
            B_H_u_M_i = B_H_u_M_i.union(axis_sets[f'B_u_{t}_M_{agent_id}'])
        axis_sets[f'B_H_u_M_{agent_id}'] = B_H_u_M_i
        
        # B^H_M_i - Combined state-input axis set for neighbors over horizon
        B_H_M_i = AxisSet([])
        for t in range(horizon + 1):
            B_H_M_i = B_H_M_i.union(axis_sets[f'B_{t}_M_{agent_id}'])
        axis_sets[f'B_H_M_{agent_id}'] = B_H_M_i
    
    # Create the global axis sets
    
    # B̄^H - Global axis set for all variables over horizon
    B_H = AxisSet([])
    for agent_id in agents:
        B_H = B_H.union(axis_sets[f'B_H_{agent_id}'])
    axis_sets['B_H'] = B_H
    
    # B̄_x,t - Global state axis set at time t
    for t in range(horizon + 1):
        B_x_t = AxisSet([])
        for agent_id in agents:
            B_x_t = B_x_t.union(axis_sets[f'B_x_{t}_{agent_id}'])
        axis_sets[f'B_x_{t}'] = B_x_t
    
    # B̄^H_u - Global input axis set over horizon
    B_H_u = AxisSet([])
    for agent_id in agents:
        B_H_u = B_H_u.union(axis_sets[f'B_H_u_{agent_id}'])
    axis_sets['B_H_u'] = B_H_u
    
    return axis_sets