from pyReach.core.ContDynamics.ContDynamics import ContDynamics
import numpy as np

class linParamSys(ContDynamics):
    """
    % linParamSys class (linear parametric system)
%
% Syntax:
%    obj = linParamSys(A,B)
%    obj = linParamSys(A,B,type)
%    obj = linParamSys(name,A,B)
%    obj = linParamSys(name,A,B,type)
%
% Inputs:
%    name - name of the system
%    A - system matrix
%    B - input matrix
%    type - constant/time-varying parameters
%           - 'constParam' (constant parameters, default)
%           - 'varParam' (time-varying parameters)
%
% Outputs:
%    obj - generated linParamSys object
%
% Example:
%    Ac = [-2 0; 1.5 -3]; Aw = [0 0; 0.5 0];
%    A = intervalMatrix(Ac,Aw);
%    B = [1; 1];
%    sys = linParamSys(A,B,'varParam')
    """
    def __init__(self,name,A,B,type):
        if not self._check_input_args(name,A,B,type):
            raise ValueError("Invalid input arguments")
        states,inputs,outputs,dists,noises = self._compute_properties(A,B,type)
        super().__init__(name,states,inputs,outputs,dists,noises)
        

    def _check_input_args(name,A,B,type):
        return True
    
    def _compute_properties(self):
        pass 
    #! TODO: Implement this function

