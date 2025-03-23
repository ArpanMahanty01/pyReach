class ReachSet:
    """Container for reachable sets."""
    
    def __init__(self):
        self.time_points = []
        self.time_intervals = []
        self.sets_at_time_points = []
        self.sets_at_time_intervals = []
    
    def add(self, time, reachable_set, is_time_point=True):
        if is_time_point:
            self.time_points.append(time)
            self.sets_at_time_points.append(reachable_set)
        else:
            self.time_intervals.append(time)
            self.sets_at_time_intervals.append(reachable_set)
    
    def plot(self, dims=None, **kwargs):
        # Implementation for plotting reachable sets
        pass