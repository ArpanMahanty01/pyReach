from abc import ABC,abstractmethod

class ContDynamics(ABC):
    """
    % contDynamics - basic class for continuous dynamics
%
% Syntax:
%    sys = contDynamics()
%    sys = contDynamics(name)
%    sys = contDynamics(name,states)
%    sys = contDynamics(name,states,inputs)
%    sys = contDynamics(name,states,inputs,outputs)
%    sys = contDynamics(name,states,inputs,outputs,dists)
%    sys = contDynamics(name,states,inputs,outputs,dists,noises)
%
% Inputs:
%    name - system name
%    states - number of states
%    inputs - number of inputs
%    outputs - number of outputs
%    dists - number of disturbances
%    noises - number of disturbances on output
%
% Outputs:
%    sys - generated contDynamics object
%
% Example:
%    sys = contDynamics('system',2,1,1);
    """
    def __init__(self,name,states,inputs,outputs,dists,noises):
        self.name = name
        self.states = states
        self.inputs = inputs
        self.outputs = outputs
        self.dists = dists
        self.noises = noises
