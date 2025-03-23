from pyReach.core.ContDynamics.ContDynamics import ContDynamics
import numpy as np

class NonlinearSys(ContDynamics):
    """Nonlinear continuous-time system: dx/dt = f(x, u)."""
    
    def __init__(self, f, n=None, m=None, output_func=None, p=None):
        self.f = f
        
        # Determine dimensions if not provided
        if n is None or m is None:
            # Try to infer dimensions
            x_test = np.zeros(1)
            u_test = np.zeros(1)
            try:
                result = f(x_test, u_test)
                n = len(result)
                m = 1
            except:
                raise ValueError("Could not determine dimensions, please provide n and m")
        
        self.n = n  # state dimension
        self.m = m  # input dimension
        
        if output_func is None:
            self.output_func = lambda x, u: x
        else:
            self.output_func = output_func
            
        if p is None:
            self.p = n  # output dimension
        else:
            self.p = p
    
    def dimension(self):
        return self.n
    
    def evaluate(self, x, u=None, t=None):
        if u is None:
            u = np.zeros(self.m)
        return self.f(x, u)