class Location:
    """Location of a hybrid automaton."""
    
    def __init__(self, invariant, transitions, dynamics, name=None):
        self.invariant = invariant  # ContSet object
        self.transitions = transitions  # List of Transition objects
        self.dynamics = dynamics  # ContDynamics object
        self.name = name
    
    def is_valid(self, x):
        """Check if a state satisfies the invariant."""
        return self.invariant.contains(x)
    
    def get_enabled_transitions(self, x):
        """Get all enabled transitions for a given state."""
        return [t for t in self.transitions if t.is_enabled(x)]