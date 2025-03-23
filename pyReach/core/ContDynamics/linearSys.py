from pyReach.core.ContDynamics.ContDynamics import ContDynamics
from pyReach.core._global.taylorLinSys import TaylorLinSys
import numpy as np
from scipy.linalg import expm, inv

class LinearSys(ContDynamics):
    """Class representing linear time-invariant systems."""
    
    def __init__(self, *args):
        # Parse input arguments
        name, A, B, c, C, D, k, E, F = self._parse_input_args(args)
        
        # Compute properties
        name, A, B, c, C, D, k, E, F, states, inputs, outputs, dists, noises = \
            self._compute_properties(name, A, B, c, C, D, k, E, F)
        
        # Initialize parent class
        super().__init__(name, states, inputs, outputs, dists, noises)
        
        # Assign properties
        self.A = A
        self.B = B
        self.c = c
        self.C = C
        self.D = D
        self.k = k
        self.E = E
        self.F = F
        
        # Initialize helper properties
        self.taylor = None
        self.krylov = {}
    
    def _parse_input_args(self, args):
        """Parse input arguments."""
        def_name = 'linearSys'
        A, B, c, C, D, k, E, F = None, None, None, None, None, None, None, None
        
        if len(args) > 0 and isinstance(args[0], str):
            # First argument is name
            name = args[0]
            remaining_args = args[1:]
        else:
            name = def_name
            remaining_args = args
        
        # Assign remaining arguments
        if len(remaining_args) > 0:
            A = remaining_args[0]
        if len(remaining_args) > 1:
            B = remaining_args[1]
        if len(remaining_args) > 2:
            c = remaining_args[2]
        if len(remaining_args) > 3:
            C = remaining_args[3]
        if len(remaining_args) > 4:
            D = remaining_args[4]
        if len(remaining_args) > 5:
            k = remaining_args[5]
        if len(remaining_args) > 6:
            E = remaining_args[6]
        if len(remaining_args) > 7:
            F = remaining_args[7]
        
        return name, A, B, c, C, D, k, E, F
    
    def _compute_properties(self, name, A, B, c, C, D, k, E, F):
        """Compute system properties."""
        # Number of states
        states = A.shape[0] if A is not None else 0
        
        # Input matrix and number of inputs
        if A is not None and B is None:
            B = np.zeros((states, 1))
        
        inputs = states
        if B is not None and B.size > 1:  # Not scalar
            inputs = B.shape[1]
        
        # Constant offset
        if c is None:
            c = np.zeros((states, 1))
        elif len(c.shape) == 1:
            c = c.reshape(-1, 1)
        
        # Output matrix and number of outputs
        if C is None:
            C = 1
        
        outputs = states
        if hasattr(C, 'shape') and C.size > 1:  # Not scalar
            outputs = C.shape[0]
        
        # Feedthrough matrix
        if D is None:
            D = np.zeros((outputs, inputs))
        
        # Output offset
        if k is None:
            k = np.zeros((outputs, 1))
        elif len(k.shape) == 1:
            k = k.reshape(-1, 1)
        
        # Disturbance matrix and number of disturbances
        if E is None:
            E = np.zeros((states, 1))
        
        if hasattr(E, 'shape') and E.size > 1:  # Not scalar
            dists = E.shape[1]
        else:
            dists = states
        
        # Noise matrix and number of noises
        if F is None:
            F = np.zeros((outputs, 1))
        
        if hasattr(F, 'shape') and F.size > 1:  # Not scalar
            noises = F.shape[1]
        else:
            noises = outputs
        
        return name, A, B, c, C, D, k, E, F, states, inputs, outputs, dists, noises
    
    def get_taylor(self, name, **kwargs):
        """Wrapper function to compute and access Taylor series values."""
        if self.taylor is None:
            self.taylor = TaylorLinSys(self.A)
        
        return self._compute_field(self.taylor, name, **kwargs)
    
    def _compute_field(self, obj, name, **kwargs):
        """Compute a field for the taylor object."""
        if name == 'eAdt':
            timeStep = kwargs.get('timeStep')
            if timeStep in obj.eAdt:
                return obj.eAdt[timeStep]
            else:
                result = expm(self.A * timeStep)
                obj.eAdt[timeStep] = result
                return result
        
        elif name == 'Apower':
            timeStep = kwargs.get('timeStep')
            ithpower = kwargs.get('ithpower')
            
            # Ensure powers are computed up to ithpower
            if ithpower not in obj.powers:
                for i in range(1, ithpower + 1):
                    if i == 1:
                        obj.powers[i] = self.A
                    else:
                        obj.powers[i] = obj.powers[i-1] @ self.A / i
            
            return obj.powers[ithpower]
        
        elif name == 'Apower_abs':
            timeStep = kwargs.get('timeStep')
            ithpower = kwargs.get('ithpower')
            
            # Ensure powers_abs are computed up to ithpower
            if ithpower not in obj.powers_abs:
                A_abs = np.abs(self.A)
                for i in range(1, ithpower + 1):
                    if i == 1:
                        obj.powers_abs[i] = A_abs
                    else:
                        obj.powers_abs[i] = obj.powers_abs[i-1] @ A_abs
            
            return obj.powers_abs[ithpower]
        
        elif name == 'dtoverfac':
            timeStep = kwargs.get('timeStep')
            ithpower = kwargs.get('ithpower')
            
            # Compute dt^i / i!
            return timeStep**ithpower / np.math.factorial(ithpower)
        
        elif name == 'Ainv':
            # Try to compute inverse of A if possible
            try:
                return inv(self.A)
            except:
                return None
        
        return None
    
    def particularSolution_constant(self, U, timeStep, truncationOrder, blocks=None):
        """Compute particular solution for constant input."""
        # Initialize taylor helper if needed
        if self.taylor is None:
            self.taylor = TaylorLinSys(self.A)
        
        # Check if U is zero
        if np.all(np.abs(U) < np.finfo(float).eps):
            return np.zeros((self.nrOfDims, 1))
        
        # Check if inverse exists for fast computation
        Ainv = self.get_taylor('Ainv')
        if Ainv is not None:
            eAdt = self.get_taylor('eAdt', timeStep=timeStep)
            return Ainv @ (eAdt - np.eye(self.nrOfDims)) @ U
        
        # Compute by power series
        truncationOrderInf = np.isinf(truncationOrder)
        if truncationOrderInf:
            truncationOrder = 75
        
        # First term (eta = 0)
        Asum = timeStep * np.eye(self.nrOfDims)
        
        # Loop until convergence or max order
        for eta in range(1, truncationOrder):
            # Get A^eta
            Apower_mm = self.get_taylor('Apower', timeStep=timeStep, ithpower=eta)
            # Get dt^(eta+1)/(eta+1)!
            dtoverfac = self.get_taylor('dtoverfac', timeStep=timeStep, ithpower=eta+1)
            # Additional term
            addTerm = Apower_mm * dtoverfac
            
            # Adaptive truncation
            if truncationOrderInf:
                if np.any(np.isinf(addTerm)) or eta == truncationOrder - 1:
                    raise Exception("Time Step Size too big for computation of Pu.")
                if np.all(np.abs(addTerm) <= np.finfo(float).eps * np.abs(Asum)):
                    break
            
            # Add term to current Asum
            Asum = Asum + addTerm
        
        # Compute particular solution
        return Asum @ U
    
    def particularSolution_timeVarying(self, U, timeStep, truncationOrder, blocks=None):
        """Compute particular solution for time-varying input."""
        # Initialize taylor helper if needed
        if self.taylor is None:
            self.taylor = TaylorLinSys(self.A)
        
        # Check if U is zero
        if np.all(np.abs(U) < np.finfo(float).eps):
            return np.zeros((self.nrOfDims, 1))
        
        # First term (eta = 0: A^0*dt^1/1 * U = dt*U)
        Ptp = timeStep * U
        
        # Truncation order handling
        truncationOrderInf = np.isinf(truncationOrder)
        if truncationOrderInf:
            truncationOrder = 75
        
        # Loop until convergence or max order
        for eta in range(1, truncationOrder):
            # Get A^eta
            Apower_mm = self.get_taylor('Apower', timeStep=timeStep, ithpower=eta)
            # Get dt^(eta+1)/(eta+1)!
            dtoverfac = self.get_taylor('dtoverfac', timeStep=timeStep, ithpower=eta+1)
            # Additional term
            addTerm = Apower_mm * dtoverfac
            
            # Adaptive truncation
            if truncationOrderInf:
                if np.any(np.isinf(addTerm)) or eta == truncationOrder - 1:
                    raise Exception("Time Step Size too big for computation.")
                if np.all(np.abs(addTerm) <= np.finfo(float).eps):
                    break
            
            # Add term
            Ptp_eta = addTerm @ U
            Ptp = Ptp + Ptp_eta
        
        return Ptp
    
    def homogeneousSolution(self, X, timeStep, truncationOrder, blocks=None):
        """Compute homogeneous solution."""
        # Initialize taylor helper if needed
        if self.taylor is None:
            self.taylor = TaylorLinSys(self.A)
        
        # Get exponential matrix
        eAdt = self.get_taylor('eAdt', timeStep=timeStep)
        
        # Compute homogeneous time-point solution
        Htp = eAdt @ X
        
        return Htp
    
    def one_step(self, X, U, u, timeStep, truncationOrder, blocks=None):
        """Compute reachable set for one time step."""
        # Compute particular solutions
        PU = self.particularSolution_timeVarying(U, timeStep, truncationOrder)
        Pu = self.particularSolution_constant(u, timeStep, truncationOrder)
        
        # Compute homogeneous solution
        Htp = self.homogeneousSolution(X, timeStep, truncationOrder)
        
        # Extend to affine solution
        Htp = Htp + Pu
        
        # Compute time-interval solution
        Hti = self._enclose(X, Htp)
        
        # Compute reachable sets
        Rtp = Htp + PU
        Rti = Hti + PU
        
        return Rtp, Rti, Htp, Hti, PU, Pu
    
    def _enclose(self, X, Y):
        """Helper function to create envelope of two sets."""
        # This is a simplified version - in CORA this would depend on set representation
        # For now, we just return the union
        return np.concatenate([X, Y], axis=1)
    
    def canonicalForm(self, U, uVec, W, V, vVec):
        """Rewrite system into canonical form."""
        # Calculate centers
        centerU = np.mean(U, axis=1, keepdims=True)
        centerW = np.mean(W, axis=1, keepdims=True)
        centerV = np.mean(V, axis=1, keepdims=True)
        
        # Shift sets to center at origin
        U_shifted = U - centerU
        W_shifted = W - centerW 
        V_shifted = V - centerV
        
        # Offset for output
        if uVec.shape[1] == 1:
            v_ = self.D @ uVec + self.k + self.F @ (centerV + vVec)
        elif not np.any(self.D) and not np.any(self.k) and not np.any(self.F):
            v_ = np.zeros((self.nrOfOutputs, 1))
        else:
            v_ = self.D @ np.column_stack([uVec, np.zeros((self.nrOfInputs, 1))]) + self.k + self.F @ (centerV + vVec)
        
        # Simplify if zero
        if not np.any(v_):
            v_ = np.zeros((self.nrOfOutputs, 1))
        
        # Time-varying uncertainty on output
        V_ = self.D @ (U_shifted + centerU) + self.F @ V_shifted
        
        # Offset for state
        u_ = self.B @ uVec + self.B @ centerU + self.c + self.E @ centerW
        
        # Time-varying uncertainty for state
        U_ = self.B @ U_shifted + self.E @ W_shifted
        
        # Create canonical system
        n = self.nrOfDims
        r = self.nrOfOutputs
        linsys_ = LinearSys(
            self.name,
            self.A, np.eye(n), np.zeros((n, 1)),
            self.C, np.zeros((r, n)), np.zeros((r, 1)),
            np.zeros((n, n)), np.eye(r)
        )
        
        # Copy taylor
        linsys_.taylor = self.taylor
        
        return linsys_, U_, u_, V_, v_
    
    def is_equal(self, other, tol=np.finfo(float).eps):
        """Check if two systems are equal."""
        if not isinstance(other, LinearSys):
            return False
        
        if self.name != other.name:
            return False
        
        if (self.nrOfDims != other.nrOfDims or 
            self.nrOfInputs != other.nrOfInputs or 
            self.nrOfOutputs != other.nrOfOutputs):
            return False
        
        # Compare matrices
        if not np.allclose(self.A, other.A, atol=tol):
            return False
        
        if not np.allclose(self.B, other.B, atol=tol):
            return False
        
        if not np.allclose(self.c, other.c, atol=tol):
            return False
        
        if not np.allclose(self.C, other.C, atol=tol):
            return False
        
        if not np.allclose(self.D, other.D, atol=tol):
            return False
        
        if not np.allclose(self.k, other.k, atol=tol):
            return False
        
        if not np.allclose(self.E, other.E, atol=tol):
            return False
        
        if not np.allclose(self.F, other.F, atol=tol):
            return False
        
        return True
    
    def is_canonical_form(self):
        """Check if the system is in canonical form."""
        # Check dimensions
        if self.nrOfInputs != self.nrOfDims or self.nrOfNoises != self.nrOfOutputs:
            return False
        
        # Check offsets and matrices
        if (np.any(self.c) or np.any(self.k) or 
            np.any(self.E) or np.any(self.D)):
            return False
        
        # Check input/noise matrices are identity
        if (not np.array_equal(self.B, np.eye(self.nrOfDims)) or 
            not np.array_equal(self.F, np.eye(self.nrOfOutputs))):
            return False
        
        return True
    
    def to_nonlinear_sys(self):
        """Convert to a nonlinear system representation."""
        # This would create a nonlinear system with function handles
        # But since we don't have the NonlinearSys class implementation,
        # we'll just return a message
        if np.any(self.E) or np.any(self.F):
            raise ValueError("Only all-zero disturbance/noise matrices supported for conversion.")
        
        return "NonlinearSys object would be created here"
    
    def to_discrete_time(self, dt):
        """Convert to a discrete-time system."""
        # Compute discrete-time system matrix
        A_dt = expm(self.A * dt)
        
        # Compute discrete-time input matrix
        # B_dt = A^-1 * (e^A*dt - I) * B
        term_j = np.eye(self.nrOfDims) * dt
        T = term_j.copy()
        
        for j in range(2, 1001):
            term_j = term_j @ (dt/j * self.A)
            T = T + term_j
            if np.all(np.abs(term_j) < np.finfo(float).eps):
                break
        
        B_dt = T @ self.B
        E_dt = T @ self.E
        c_dt = T @ self.c
        
        # This would create a LinearSysDT object in CORA
        return f"LinearSysDT object would be created with A_dt, B_dt, etc."
    
    @staticmethod
    def generate_random(state_dim=None, input_dim=None, output_dim=None, 
                        real_interval=None, imaginary_interval=None):
        """Generate a random linear system."""
        if state_dim is None:
            state_dim = np.random.randint(4, 11)
        
        if input_dim is None:
            input_dim = np.random.randint(1, 4)
        
        if output_dim is None:
            output_dim = np.random.randint(1, 3)
        
        if real_interval is None:
            real_interval = [-1-10*np.random.rand(), -np.random.rand()]
        
        if imaginary_interval is None:
            imag_max = 10*np.random.rand()
            imaginary_interval = [-imag_max, imag_max]
        
        # Determine number of complex eigenvalues (conjugate pairs)
        n_conj_max = state_dim // 2
        n_conj = np.random.randint(0, n_conj_max + 1)
        n_real = state_dim - 2 * n_conj
        
        # Generate eigenvalues
        real_vals = np.random.uniform(real_interval[0], real_interval[1], n_real)
        real_parts = np.random.uniform(real_interval[0], real_interval[1], n_conj)
        imag_parts = np.random.uniform(0, min(abs(imaginary_interval[0]), imaginary_interval[1]), n_conj)
        
        # Construct Jordan form
        J_real = np.diag(real_vals)
        J_conj = np.zeros((2*n_conj, 2*n_conj))
        
        for i in range(n_conj):
            idx = 2*i
            J_conj[idx:idx+2, idx:idx+2] = np.array([
                [real_parts[i], -imag_parts[i]],
                [imag_parts[i], real_parts[i]]
            ])
        
        # Combine real and complex parts
        if n_real > 0 and n_conj > 0:
            J = np.block([[J_real, np.zeros((n_real, 2*n_conj))],
                           [np.zeros((2*n_conj, n_real)), J_conj]])
        elif n_real > 0:
            J = J_real
        else:
            J = J_conj
        
        # Random similarity transformation
        P = np.random.randn(state_dim, state_dim)
        A = P @ J @ np.linalg.inv(P)
        
        # Random input and output matrices
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        
        return LinearSys(A, B, None, C)
    
    def __eq__(self, other):
        return self.is_equal(other)
    
    def __ne__(self, other):
        return not self.is_equal(other)
    
    def _affine_solution(self,linsys_,Htp_start,u,timeStep,taylorTerms):
        """
        Computes the affine solution for linear systems.
        
        Args:
            linsys_: LinearSys object
        """
        pass
    
# Private methods
    def __reach_standard(self,params,options):
        """
    Computes the reachable set for linear systems using the standard reachability algorithm.
    
    Args:
        params: Dictionary containing model parameters:
            - R0: Initial reachable set
            - U: Input set
            - uTrans: Constant input vector or None
            - uTransVec: Time-varying input vector sequence or None
            - W: Disturbance set
            - V: Sensor noise set
            - tStart: Initial time
            - tFinal: Final time
        options: Dictionary containing options:
            - timeStep: Time step size
            - taylorTerms: Number of Taylor terms
            - reductionTechnique: Technique for order reduction
            - zonotopeOrder: Maximum zonotope order
            - verbose: Level of verbosity
            - specification: Safety specification (optional)
    
    Returns:
        timeInt: Time-interval output sets
        timePoint: Time-point output sets
        res: True if specifications are satisfied, False otherwise
        """
        # Time period and number of steps
        tVec = np.arange(params['tStart'], params['tFinal'] + options['timeStep'], options['timeStep'])
        steps = len(tVec) - 1
    
        # Put system into canonical form
        if 'uTransVec' in params:
            linsys_, U, u, V, v = self.canonicalForm(
                params['U'], params['uTransVec'], params['W'], params['V'], np.zeros((self.nrOfNoises, 1)))
        else:
            linsys_, U, u, V, v = self.canonicalForm(
                params['U'], params['uTrans'], params['W'], params['V'], np.zeros((self.nrOfNoises, 1)))
    
        # Check if input is zero
        isU = not np.all(np.abs(U) < np.finfo(float).eps)
    
        # Initialize output variables for reachable sets and output sets
        timeInt = {'set': [None] * steps, 'time': [None] * steps}
        timePoint = {'set': [None] * (steps + 1), 'time': tVec.tolist()}
    
        # Log information if verbose
        if options.get('verbose', 0) > 0:
            print(f"Step 1 of {steps}, time: {params['tStart']:.6f} to {params['tFinal']:.6f}")
    
    # Compute reachable sets for first step
        Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input = self.one_step(
            params['R0'], U, u[:, 0].reshape(-1, 1) if u.ndim > 1 else u, 
            options['timeStep'], options['taylorTerms'])
    
    # Read out propagation matrix and base particular solution
        eAdt = self.get_taylor('eAdt', timeStep=options['timeStep'])
        if not isU:
            PU_next = PU
    
    # Compute output set of start set and first time-interval solution
        timePoint['set'][0] = self._output_set_canonical_form(linsys_, params['R0'], V, v, 1)
        timeInt['set'][0] = self._output_set_canonical_form(linsys_, Rti, V, v, 1)
        timeInt['time'][0] = [tVec[0], tVec[1]]
        timePoint['set'][1] = self._output_set_canonical_form(linsys_, Rtp, V, v, 2)
    
    # Safety property check
        res = True
        if 'specification' in options:
            res, timeInt, timePoint = self._check_specification(
                options['specification'], Rti, timeInt, timePoint, 0)
            if not res:
                return timeInt, timePoint, res
    
    # Loop over all reachability steps
        for k in range(1, steps):
        # Method implemented from Algorithm 1 in [1]
        
        # Re-compute particular solution due to constant input if we have a
        # time-varying input trajectory
            if 'uTransVec' in params:
                Htp_start = Htp
                Htp, Pu, _, C_state, C_input = self._affine_solution(
                    linsys_, Htp_start, u[:, k].reshape(-1, 1) if u.ndim > 1 else u, 
                    options['timeStep'], options['taylorTerms'])
                Hti = self._enclose(Htp_start, Htp) + C_state
            else:
            # Homogeneous solution, including reduction
                Htp = eAdt @ Htp + Pu
                Hti = eAdt @ Hti + Pu
        
        # Reduction
            Htp = self._reduce(Htp, options['reductionTechnique'], options['zonotopeOrder'])
            Hti = self._reduce(Hti, options['reductionTechnique'], options['zonotopeOrder'])
        
            if not isU:
            # Propagate particular solution (time-varying, centered at zero)
                PU_next = eAdt @ PU_next
                PU = self._reduce(PU + PU_next, options['reductionTechnique'], options['zonotopeOrder'])
        
        # Compute reachable set
            Rti = Hti + PU + C_input
            Rtp = Htp + PU
        
        # Compute output set
            timeInt['set'][k] = self._output_set_canonical_form(linsys_, Rti, V, v, k)
            timeInt['time'][k] = [tVec[k], tVec[k + 1]]
        
        # Compute output set for start set of next step
            timePoint['set'][k + 1] = self._output_set_canonical_form(linsys_, Rtp, V, v, k + 1)
        
        # Safety property check
            if 'specification' in options:
                res, timeInt, timePoint = self._check_specification(
                    options['specification'], Rti, timeInt, timePoint, k)
                if not res:
                    return timeInt, timePoint, res
        
        # Log information if verbose
            if options.get('verbose', 0) > 0:
                print(f"Step {k+1} of {steps}, time: {tVec[k]:.6f} to {params['tStart']:.6f}/{params['tFinal']:.6f}")
    
    # Specification fulfilled
        return timeInt, timePoint, res
    
    