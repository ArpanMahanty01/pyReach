import numpy as np

class Interval:
    lower_bound: np.array
    upper_bound: np.array

    def __init__(self,lower_bound,upper_bound):
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)

        if self.lower_bound.shape != self.upper_bound.shape:
            raise ValueError("Lower and upper bounds must have the same shape")
        
        if isinstance(lower_bound,np.ndarray) and isinstance(upper_bound,np.ndarray):
            if np.any(self.lower_bound > self.upper_bound):
                raise ValueError("Lower bound must be less than upper bound")
            
    @property
    def radius(self):
        return 0.5 * np.sum(self.upper_bound - self.lower_bound)
    
    def contains(self,X,tolerance=1e-9):
        assert isinstance(X,np.ndarray), "X must be a numpy array"
        assert X.shape == self.lower_bound.shape, "X must have the same number of rows as the lower and upper bounds"

        return np.all(X + tolerance >= self.lower_bound) and np.all(X - tolerance <= self.upper_bound)
    