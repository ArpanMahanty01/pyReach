from pyReach.core.ContDynamics.ContDynamics import ContDynamics

class LinearSysDT(ContDynamics):
    """
    % linearSysDT - object constructor for linear discrete-time systems
%    
% Description:
%    Generates a discrete-time linear system object according to the 
%    following first-order difference equations:
%       x(k+1) = A x(k) + B u(k) + c + E w(k)
%       y(k)   = C x(k) + D u(k) + k + F v(k)
%
% Syntax:
%    linsysDT = linearSysDT(A,B,dt)
%    linsysDT = linearSysDT(A,B,c,dt)
%    linsysDT = linearSysDT(A,B,c,C,dt)
%    linsysDT = linearSysDT(A,B,c,C,D,dt)
%    linsysDT = linearSysDT(A,B,c,C,D,k,dt)
%    linsysDT = linearSysDT(A,B,c,C,D,k,E,dt)
%    linsysDT = linearSysDT(A,B,c,C,D,k,E,F,dt)
%    linsysDT = linearSysDT(name,A,B,dt)
%    linsysDT = linearSysDT(name,A,B,c,dt)
%    linsysDT = linearSysDT(name,A,B,c,C,dt)
%    linsysDT = linearSysDT(name,A,B,c,C,D,dt)
%    linsysDT = linearSysDT(name,A,B,c,C,D,k,dt)
%    linsysDT = linearSysDT(name,A,B,c,C,D,k,E,dt)
%    linsysDT = linearSysDT(name,A,B,c,C,D,k,E,F,dt)
%
% Inputs:
%    name - name of system
%    A - state matrix
%    B - input matrix
%    c - constant input
%    C - output matrix
%    D - feedthrough matrix
%    k - output offset
%    E - disturbance matrix
%    F - output disturbance matrix
%    dt - sampling time
%
% Outputs:
%    linsysDT - generated linearSysDT object
    """
    def __init__(self,name,A,B,c,C,D,k,E,F,dt):
        states,inputs,outputs,dists,noises = self._compute_properties(A,B,c,C,D,k,E,F,dt)
        super().__init__(name,states,inputs,outputs,dists,noises)
        
    def _compute_properties(self,A,B,c,C,D,k,E,F,dt):
        pass 
    #! TODO: Implement this function
