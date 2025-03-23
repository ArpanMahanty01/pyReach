from pyReach.core.ContDynamics.ContDynamics import ContDynamics

class linProbSys(ContDynamics):
    """
    % linProbSys - class constructor for linear probabilistic systems
%
% Description:
%    Generates a linear stochastic differential system, also known as the
%    multivariate Ornstein-Uhlenbeck process:
%       x'(t) = A x(t) + B u(t) + C xi(t),
%    where xi(t) is white Gaussian noise.
%
% Syntax:
%    obj = linProbSys(A,B)
%    obj = linProbSys(A,B,C)
%    obj = linProbSys(name,A,B)
%    obj = linProbSys(name,A,B,C)
%
% Inputs:
%    name - name of system
%    A - state matrix
%    B - input matrix
%    C - noise matrix
%
% Outputs:
%    obj - generated linProbSys object
    """

    def __init__(self,name,A,B,C):
        states,inputs,outputs,dists,noises = self._compute_properties(A,B,C)
        super().__init__(name,states,inputs,outputs,dists,noises)
        
    def _compute_properties(self,A,B,C):
        pass 
    #! TODO: Implement this function