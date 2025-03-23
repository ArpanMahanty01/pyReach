class Transition:
    """Transition between locations in a hybrid automaton."""
    
    def __init__(self, guard, reset, target_idx, label=None):
        self.guard = guard  # ContSet object
        self.reset = reset  # Reset function
        self.target_idx = target_idx  # Target location index
        self.label = label  # Synchronization label
    
    def is_enabled(self, x):
        """Check if the transition is enabled for a given state."""
        return self.guard.contains(x)
    
    def execute(self, x):
        """Execute the reset function on a given state."""
        return self.reset(x)