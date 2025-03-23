import numpy as np
from scipy.linalg import expm

class TaylorLinSys:
    """
    Helper class for storing auxiliary values used in the reachability analysis 
    of linear continuous-time systems.
    Note: Do not use outside of built-in reachability algorithms.
    
    Args:
        A (np.ndarray): State matrix (must be square, numeric, finite, and real-valued)
    """
    
    def __init__(self, A):
        # Input validation
        if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1] \
            or not np.all(np.isfinite(A)) or not np.all(np.isreal(A)):
            raise ValueError(
                "A must be numeric, finite, 2D, square, and real-valued"
            )
        
        # Basic properties
        self.A = A
        self.A_abs = np.abs(A)
        # Ainv is computed only when requested (might be singular)
        
        # Initialize lists for powers
        self.Apower = [A]  # A^1
        self.Apower_abs = [np.abs(A)]  # |A|^1
        self.Apos = [np.where(A >= 0, A, 0)]  # Positive elements of A^1
        self.Aneg = [np.where(A < 0, A, 0)]  # Negative elements of A^1
        
        # Initialize empty lists for time-step dependent properties
        self.timeStep = []
        self.eAdt = []  # Propagation matrix e^(A*dt)
        self.E = []     # Remainder of exponential matrix
        self.F = []     # Correction matrix for state
        self.G = []     # Correction matrix for input
        self.dtoverfac = []  # dt^i/i!
        self.Apower_dt_fact = []  # (A*dt)^i/i!
        self.Apower_abs_dt_fact = []  # (|A|*dt)^i/i!
        
    def compute_field(self, name, options):
        """
        Compute requested field based on provided options.
        
        Args:
            name (str): Name of the field to compute
            options (dict): Dictionary with 'timeStep' and/or 'ithpower'
        """
        if name == 'Ainv':
            return self.compute_Ainv()
        elif name == 'eAdt':
            return self.compute_eAdt(options['timeStep'])
        elif name == 'Apower':
            return self.compute_Apower(options['ithpower'])
        elif name == 'Apower_abs':
            return self.compute_Apower_abs(options['ithpower'])
        elif name == 'Apower_dt_fact':
            return self.compute_Apower_dt_fact(options['timeStep'], options['ithpower'])
        elif name == 'Apower_abs_dt_fact':
            return self.compute_Apower_abs_dt_fact(options['timeStep'], options['ithpower'])
        elif name == 'Apos':
            return self.compute_Apos(options['ithpower'])
        elif name == 'Aneg':
            return self.compute_Aneg(options['ithpower'])
        elif name == 'dtoverfac':
            return self.compute_dtoverfac(options['timeStep'], options['ithpower'])
        else:
            raise ValueError(
                f"Field must be one of: 'Ainv', 'eAdt', 'Apower', 'Apower_abs', "
                f"'Apower_dt_fact', 'Apower_abs_dt_fact', 'Apos', 'Aneg', 'dtoverfac'"
            )

    def compute_Apower(self, ithpower):
        """Compute A^i up to the requested power."""
        idx = ithpower - 1
        while len(self.Apower) < ithpower:
            prev = self.Apower[-1]
            self.Apower.append(prev @ self.A)
        return self.Apower[idx]

    def compute_Apower_abs(self, ithpower):
        """Compute |A|^i up to the requested power."""
        idx = ithpower - 1
        while len(self.Apower_abs) < ithpower:
            prev = self.Apower_abs[-1]
            self.Apower_abs.append(prev @ self.A_abs)
        return self.Apower_abs[idx]

    def compute_Apower_dt_fact(self, timeStep, ithpower):
        """Compute (A*dt)^i/i! for given time step and power."""
        timeStepIdx = self._get_index_for_time_step(timeStep)
        if timeStepIdx == -1:
            self._make_new_time_step(timeStep)
            timeStepIdx = len(self.timeStep) - 1
            self.Apower_dt_fact.append([])

        current = self.Apower_dt_fact[timeStepIdx]
        while len(current) < ithpower:
            if not current and len(current) == 0:  # First element
                current.append(self.A * timeStep)
            else:
                i = len(current) + 1
                current.append(current[-1] @ self.A * timeStep / i)
        return current[ithpower - 1]

    def compute_Apower_abs_dt_fact(self, timeStep, ithpower):
        """Compute (|A|*dt)^i/i! for given time step and power."""
        timeStepIdx = self._get_index_for_time_step(timeStep)
        if timeStepIdx == -1:
            self._make_new_time_step(timeStep)
            timeStepIdx = len(self.timeStep) - 1
            self.Apower_abs_dt_fact.append([])

        current = self.Apower_abs_dt_fact[timeStepIdx]
        while len(current) < ithpower:
            if not current and len(current) == 0:  # First element
                current.append(self.A_abs * timeStep)
            else:
                i = len(current) + 1
                current.append(current[-1] @ self.A_abs * timeStep / i)
        return current[ithpower - 1]

    def compute_Apos(self, ithpower):
        """Compute positive elements of A^i."""
        idx = ithpower - 1
        while len(self.Apos) < ithpower:
            Apower_i = self.compute_Apower(ithpower=len(self.Apos) + 1)
            self.Apos.append(np.where(Apower_i >= 0, Apower_i, 0))
        return self.Apos[idx]

    def compute_Aneg(self, ithpower):
        """Compute negative elements of A^i."""
        idx = ithpower - 1
        while len(self.Aneg) < ithpower:
            Apower_i = self.compute_Apower(ithpower=len(self.Aneg) + 1)
            self.Aneg.append(np.where(Apower_i < 0, Apower_i, 0))
        return self.Aneg[idx]

    def compute_dtoverfac(self, timeStep, ithpower):
        """Compute dt^i/i! for given time step and power."""
        idx = self._get_index_for_time_step(timeStep)
        if idx == -1:
            self._make_new_time_step(timeStep)
            idx = len(self.timeStep) - 1
            self.dtoverfac.append([])

        current = self.dtoverfac[idx]
        while len(current) < ithpower:
            if not current and len(current) == 0:  # First element
                current.append(timeStep)
            else:
                i = len(current) + 1
                current.append(current[-1] * timeStep / i)
        return current[ithpower - 1]

    def compute_eAdt(self, timeStep):
        """Compute matrix exponential e^(A*dt)."""
        idx = self._get_index_for_time_step(timeStep)
        if idx == -1:
            self._make_new_time_step(timeStep)
            idx = len(self.timeStep) - 1
            self.eAdt[idx] = expm(self.A * timeStep)
        elif not self.eAdt[idx]:
            self.eAdt[idx] = expm(self.A * timeStep)
        return self.eAdt[idx]

    def compute_Ainv(self):
        """Compute inverse of A if it exists."""
        if not hasattr(self, 'Ainv'):
            if np.linalg.matrix_rank(self.A) < self.A.shape[0]:
                self.Ainv = None
            else:
                self.Ainv = np.linalg.inv(self.A)
        return self.Ainv

    def _get_index_for_time_step(self, timeStep):
        """Get index of existing time step within tolerance."""
        if not self.timeStep:
            return -1
        idx = np.where(np.abs(np.array(self.timeStep) - timeStep) < 1e-10)[0]
        return idx[0] if len(idx) > 0 else -1

    def _make_new_time_step(self, timeStep):
        """Add new time step and initialize associated lists."""
        self.timeStep.append(timeStep)
        self.eAdt.append(None)
        self.E.append(None)
        self.F.append(None)
        self.G.append(None)
        self.dtoverfac.append([])
        self.Apower_dt_fact.append([])
        self.Apower_abs_dt_fact.append([])

    def insert_field_time_step(self, field, val, timeStep):
        """Insert value for a specific field at given time step."""
        idx = self._get_index_for_time_step(timeStep)
        if idx == -1:
            self._make_new_time_step(timeStep)
            idx = len(self.timeStep) - 1
        
        if field == 'E':
            self.E[idx] = val
        elif field == 'F':
            self.F[idx] = val
        elif field == 'G':
            self.G[idx] = val
        elif field == 'eAdt':
            self.eAdt[idx] = val
        else:
            raise ValueError("Field must be 'E', 'F', 'G', or 'eAdt'")
