from pyReach.core.ContDynamics import linParamSys,linearSysDT
from pyReach.core.ContDynamics import linearSys

class ReachabilityAnalyser():
    def __init__(self,sys,params,options):
        self.sys = sys
        self.params = params
        self.options = options

    def compute(self):
        """
        computes and returns the reachable set of the system
        """

        if isinstance(self.sys,linearSys):
            if self.options.get('algorithm') == 'standard':
                return self.sys.__reach_standard()
            
    def check_violation(self,reachSet,spec):
        """
        checks if the reachable set violates the specification
        """
        
    

        
                
        

        
        