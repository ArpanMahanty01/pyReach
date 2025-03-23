class ReachOptions:
    """Options for reachability analysis."""
    
    def __init__(self):
        # General options
        self.time_step = 0.01
        self.taylor_terms = 4
        self.zonotope_order = 10
        self.intermediate_order = 20
        self.error_order = 5
        self.alg = 'lin'  # options: 'lin', 'poly', etc.
        self.tensor_order = 2
        self.reduction_technique = 'girard'
        
        # Options for hybrid systems
        self.guard_intersect = 'polytope'
        self.enclose = ['box', 'pca']