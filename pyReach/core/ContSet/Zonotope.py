import numpy as np

class Zonotope:
    Z: np.ndarray
    halfspace: np.ndarray

    def __init__(self,center,generator):
        center = np.array(center)
        generator = np.array(generator)
        
        if center.shape[0] != generator.shape[0]:
            raise ValueError("Center and generator must have the same number of rows")
        
        
        
        
            