###############################################################################
#    Copyright (C) 2007 by Eike Welk                                          #
#    eike.welk@gmx.net                                                        #
#                                                                             #
#    Permission is hereby granted, free of charge, to any person obtaining    #
#    a copy of this software and associated documentation files (the          #
#    "Software"), to deal in the Software without restriction, including      #
#    without limitation the rights to use, copy, modify, merge, publish,      #
#    distribute, sublicense, and#or sell copies of the Software, and to       #
#    permit persons to whom the Software is furnished to do so, subject to    #
#    the following conditions:                                                #
#                                                                             #
#    The above copyright notice and this permission notice shall be           #
#    included in all copies or substantial portions of the Software.          #
#                                                                             #
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,          #
#    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF       #
#    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.   #
#    IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR        #
#    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,    #
#    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR    #
#    OTHER DEALINGS IN THE SOFTWARE.                                          #
###############################################################################


"""
Naive but fairly general implementation of a Kalman Filter, together with
some examples.


This program is based on the following paper:
    Greg Welsh and Garry Bishop. An Introduction to the Kalman Filter.
    http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

See also:
    http://en.wikipedia.org/wiki/Kalman_filter


A Kalman filter is an Algorithm to estimate the states of a linear system
from noisy measurements. It must know the system's parameters, as well as
the (noisy) inputs and outputs.

A linear system can be interpreted as a system of linear differential
equations, in a time discete form. Therfore the state variables of many
physical systems can be estimated by a Kalman Filter. But there does not
need to be an interpretation as a differential equation. Welsh and Bishop
(the authors of the paper above) use the Kalman Filter to predict the
position of moving objects in image sequences (IMHO).

System Equations:
            x[k] = A*x[k-1] + B*u[k] + w[k]                             (1)

            z[k] = H*x[k] + v[k]                                        (2)

with:
    x: State variables (vector)
    u: Inputs (vector)
    z: Outputs (vector)

    w,v: Random noise (vector)
    A,B,H: Parameter matrices

(For more details of the system see the documentation of the LinSys class.)

As one can see from equation (2) it is not necessary to measure the state
variables directly. A Kalman Filter can estimate the system states from a
linear combination of them, with added noise.

Example systems:
    - Estimate a constant value
    - Tank with inaccurate level measurement.
    - Spring and damper system

TODO:
    - Optional more accurate method for simulation (scipy, Runge-Kutta)
    Example systems:
        - 1D Vehicle
"""


from __future__ import division

import pylab

import numpy as N
from numpy import mat, array, c_, r_, vstack, ones, zeros, identity
from numpy.linalg import inv
from numpy.random import multivariate_normal


#------------------------------------------------------------------------------
#  Main classes that do the maths
#------------------------------------------------------------------------------

class LinSys(object):
    """
    General linear dynamic system. (Discrete ODE)

    The class can simulate the system either step by step, or do the whole
    simulation in one go. It stores all system parameters and it is therefore
    used to represent the system in the Kalman filter. This class is intended
    as a base class for more specialized simulator classes.

    Random noise is generated internally, when the simulation is started. For
    repeated computations the same random values are used.

    Do not use this class to seriously solve a (continuous) differential equation.
    The integration method used here is the very inaccurate Euler method.

    System Equations:
        Compute the state:
            x_k = A x_{k-1} + B u_k + w_k

        Compute the measurements:
            z_k = H x_k + v_k

    with:
        x_k, x_{k-1}:
            Current state vector, last state vector
        u_k:
            Control vector
        z_k:
            Measurement vector
        w_k, v_k:
            Independent random variables, with (multivariate) normal probability
            distributions. They represent process noise and measurement noise.

            w_k: Process noise:     mean = 0, covariance = Q
            v_k: Measurement noise: mean = 0, covariance = R

        A, B, H:
            Parameter matrices; system, control and measurement matrix.
        Q, R:
            Covariance matrices of process and measurement noise.

    Usage:
    Inherit from this class and write a problem specific __init__
    function, that accepts arguments that are easy to understand.
    The new __init__ must compute the matrices A, B, H, Q, R and give
    them to LinSys.__init__.
    For examples see: SysConstVal, SysTank, SysSpringPendulum

    You can call simulate(x_0, U) to compute multiple simulation steps at
    once:
    >>> tank = SysTank(q=0.002, r=2)          #Create system
    >>> U = vstack((ones(10)*0.1, zeros(10))) #Create inputs
    >>> X, Z = tank.simulate(mat('2.'), U)    #Simulate

    For more complex problems you can compute each simulation step
    separately:
    - At the beginning of the simulation call startSimulation(x_0) to supply
      the initial values.
    - For each time step call simulateStep(u_k). You must supply the input
      values.
    """

    def __init__(self, A=None, B=None, H=None, Q=None, R=None, nSteps=1000):
        """
        Parameter:
            A:      system matrix
            B:      control input matrix
            H:      measurement matrix
            Q:      process noise covariance
            R:      measurement noise covariance
            nSteps: number of steps for which noise will be computed
        """
        self.A = mat(A).copy()  # system matrix
        self.B = mat(B).copy()  # control input matrix
        self.H = mat(H).copy()  # measurement matrix
        self.Q = mat(Q).copy()  # process noise covariance
        self.R = mat(R).copy()  # measurement noise covariance
        self.x = mat(None)      # state
        self.k = None           # step count (for noise)
        self.W = None           # precomputed process noise.
        self.V = None           # precomputed measurement noise.
        self.computeNoise(nSteps) #pre-compute the noise
        self._checkMatrixCompatibility()

    def _checkMatrixCompatibility(self):
        assert self.A.shape[0] == self.A.shape[1], \
               'Matrix "A" must be square!'
        assert self.B.shape[0] == self.A.shape[0], \
               'Matrix "B" must have same number of rows as matrix "A"!'
        assert self.H.shape[1] == self.A.shape[1], \
               'Matrix "H" must have same number of columns as matrix "A"!'
        assert self.Q.shape[0] == self.Q.shape[1] and \
               self.Q.shape[0] == self.A.shape[0], \
               'Matrix "Q" must be square, ' \
               'and it must have the same dimensions as matrix "A"'
        assert self.R.shape[0] == self.R.shape[1]and \
               self.R.shape[0] == self.H.shape[0], \
               'Matrix "R" must be square, ' \
               'and it must have the same number of rows as matrix "H"'

    def computeNoise(self, nSteps):
        """
        Compute all process and measurement noise (self.V, self.W) in advance.

        Dimensions of self.V, self.W:
            The second dimension is the step number; so self.V[:, 2] is the
            measurement noise for step two.

        Parameter:
            nSteps: Number of steps that should be simulated.
        """
        matWidth = nSteps + 1 #just in case
        #compute process noise
        meanW = N.zeros(self.A.shape[0])
        W = multivariate_normal(meanW, self.Q, [matWidth])
        self.W = mat(W).T
        #compute measurement noise
        meanV = N.zeros(self.H.shape[0])
        V = multivariate_normal(meanV, self.R, [matWidth])
        self.V = mat(V).T

    def _procNoise(self):
        """Return process noise for current step."""
        return self.W[:, self.k]
    def _measNoise(self):
        """Return measurement noise current step."""
        return self.V[:, self.k]

    def startSimulation(self, x_0):
        """
        Set initial conditions

        Parameter:
            x_0:
                initial conditions: column vector (matrix with shape[1] == 1)
        Return:
            (x_0, z_k)
            x_0:
                state vector (initial conditions): column vector
                (matrix with shape[1] == 1)
            z_k:
                first measurements: column vector (matrix with shape[1] == 1)
         """
        assert(self.A.shape[1] == x_0.shape[0])
        assert(x_0.shape[1] == 1)
        #store initial initial conditions
        self.k = 0
        self.x = mat(x_0)
        #compute the first measurement
        z_k = self.H * self.x + self._measNoise()

        return self.x, z_k

    def simulateStep(self, u_k):
        """
        Compute one simulation step.

        Parameter:
            u_k:
                control input: column vector (matrix with shape[1] == 1)
        Return:
            (x_k, z_k)
            x_k:
                state vector: column vector (matrix with shape[1] == 1)
            z_k:
                new measurements: column vector (matrix with shape[1] == 1)
        """
        #compute new state
        x_new = self.A * self.x  + self.B * u_k + self._procNoise()
        #advance time and step count (only used for repeatable noise)
        #the new state values are the current values from now on
        self.k += 1
        self.x = x_new
        #compute new measurement
        z_k = self.H * self.x + self._measNoise()

        return self.x, z_k

    def simulate(self, x_0, U, computeNoiseAgain=False):
        """
        Compute multiple time steps. The simulation is run until the control
        inputs (U) are exhausted.

        Dimensions of input and outputs:
        The second dimension is the step number; so U[:,2] is the control
        input for step three.

        Parameter:
            x_0:
                initial conditions: 1D array or matrix. Will be converted to
                column vector.
            U:
                Control inputs: 2D Array. 2nd dimension is step number.
            computeNoiseAgain:
                If True: compute new random noise even if it has already been
                computed.
                If False: compute noise only if none exists.
                Default is False.
        Return:
            (X, Z)
            X:
                states: 2D Array; 2nd dimension is step number.
            Z:
                measurements: 2D Array; 2nd dimension is step number.
        """
        #convert x_o to column vector
        x_0 = mat(x_0)
        if x_0.shape[0] == 1:
            x_0 = x_0.T
        assert x_0.shape[1] == 1  #column vector
        assert x_0.shape[0] == self.A.shape[1] # A * x_0 possible
        #Prepare control inputs
        U = mat(U)
        assert U.shape[0] == self.B.shape[1]  # B * U[:,k] possible
        #create output arrays
        X = mat(N.zeros((x_0.shape[0], U.shape[1])))
        Z = mat(N.zeros((self.H.shape[0], U.shape[1])))
        #see if there is enough noise - 'number of steps' == U.shape[1]
        if self.V.shape[1] < U.shape[1] or self.W.shape[1] < U.shape[1] or \
           computeNoiseAgain:
            self.computeNoise(U.shape[1])
        #start simulation
        dummy, z_k = self.startSimulation(x_0)
        X[:, 0] = x_0
        Z[:, 0] = z_k
        #run simulation
        for k in range(1, U.shape[1]):
            #An old input U[:, k-1] is put into the filter to compute new
            #states X[:, k] and measurements Z[:, k]
            x_k, z_k = self.simulateStep(U[:, k-1])
            X[:, k] = x_k
            Z[:, k] = z_k

        return N.asarray(X), N.asarray(Z)


class KalmanFilter(object):
    """
    Kalman Filter for use together with LinSys objects.

    A Kalman filter estimates the states of a linear system from noisy
    measurements. It must know the system's parameters, and (noisy) inputs
    and outputs. The system parameters are taken from a LinSys instance.
    The inputs and outputs must be given at each step.

    As a convenience, the function estimate(...) can process multiple
    measurements and control inputs in one go.

    Usage:
    Create system and inputs; then simulate the system
    >>> tank = SysTank(q=0.002, r=2)          #Create system
    >>> U = vstack((ones(10)*0.1, zeros(10))) #Create inputs
    >>> X, Z = tank.simulate(mat('2.'), U)    #Simulate

    Create a Filter instance, and estimate system state from noisy measurements:
    >>> kFilt = KalmanFilter(tank)            #Create Filter
    >>> X_hat = kFilt.estimate(Z, U)          #Estimate

    For complex problems each estimation step can be done separately.
    - Start the estimation with a call to startEstimation()
    - Compute one estimate with estimateStep(...)
    In the loop that computes the estimation steps you will most likely also
    compute system states (simulateStep) and control values.
    """

    def __init__(self, linearSystem=None):
        """
        Parameter:
            linearSystem:
                The linear system: LinSys.
        """
        self.sys = linearSystem # the system
        self.x_hat = mat(None) # estimated state (a posteriori)
        self.P = mat(None) # error covariance of estimated state (a posteriori)

    def startEstimation(self, x_hat_0=None, P_0=None):
        """
        Determine start values for the the estimation algorithm.

        Parameter:
            x_hat_0:
                Start value for estimated system states. (the algorithm can
                guess this value).
            P_0:
                Start value for estimation error covariance matrix (the
                algorithm can guess this value).
        Return:
            (x_hat_0, P_0)
        """
        xLen = self.sys.A.shape[0] #number of variables in state vector
        # a posteriori estimated state - start value
        self.x_hat = mat(x_hat_0).copy() if x_hat_0 is not None else \
                     mat(N.zeros(xLen)).T
        # a posteriori estimate error covariance - start value
        self.P = mat(P_0).copy() if P_0 is not None else \
                 mat(N.identity(xLen))

        return self.x_hat, self.P

    def estimateStep(self, z_k, u_k1):
        """
        Perform one estimation step.

        Function takes new measurements and control values. From these inputs
        it computes new estimates for the linear system's state and for the
        estimation error covariance.
        (System state and covariance are both returned and stored as data
        attributes.)

        This is the complete algorithm of the Kalman filter.

        Parameter
            z_k : current measurements: vector (matrix with shape[1] == 1)
            u_k1: control values from last time step. They caused the current
                  measurements: vector (matrix with shape[1] == 1)

        Return
            (x_hat, P)
            x_hat: new estimated state of linear system
            P: new estimation error covariance
        """
        #constants that describe linear system - save a little typing
        A = self.sys.A; B = self.sys.B; H = self.sys.H
        Q = self.sys.Q; R = self.sys.R
        #create identity matrix for error covariance computation
        I = mat(N.identity(self.x_hat.shape[0]))

        #the current estimated values are the old values from now on
        x_old = self.x_hat #estimated system state
        P_old = self.P #error covariance of estimated system state

        #TODO: how do I integrate the size of the time step into the covariance
        #      computation?
        #Time update - does not use new measurements (but control values)
        #compute a priori states
        x_pri = A * x_old  +  B * u_k1 #system states - simulate one time step
        P_pri = A * P_old * A.T  +  Q #error covariance

        #Measurement update - incorporate new measurement information
        #compute a posteriori states
        K = P_pri * H.T * inv(H * P_pri * H.T  +  R) # gain
        x_hat = x_pri  +  K*(z_k - H*x_pri) # system states
        P = (I - K*H) * P_pri               # error covariance

        #store filter states as member attributes
        self.x_hat = x_hat
        self.P = P

        return x_hat, P

    def estimate(self, Z, U, x_hat_0=None, P_0=None):
        """
        Estimate multiple time steps. The algorithm is run until the
        measurements (Z) and control inputs (U) are exhausted.

        Dimensions of input and outputs:
        The second dimension is the step number; so U[:, 2] is the control
        input for step two.

        Parameter:
            Z:
                Measured values: 2D Array. 2nd dimension is step number.
            U:
                Control inputs to the linear system: 2D Array.
                2nd dimension is step number.
            x_hat_0:
                Start value for estimated system states.(If None: algorithm
                will guess this value).
            P_0:
                Start value for estimation error covariance matrix (If None:
                algorithm will guess this value).
        Return:
            X_hat:
                Estimated state variables of the linear system.
                2D Array. First dimension is step number.
        """
        #prepare inputs
        Z = mat(Z)
        assert Z.shape[0] == self.sys.H.shape[0], \
               'Parameter Z: 2nd dimension must be %d' % self.sys.H.shape[0]
        U = mat(U)
        assert U.shape[0] == self.sys.B.shape[1], \
               'Parameter U: 2nd dimension must be %d' % self.sys.B.shape[1]
        assert Z.shape[1] == U.shape[1], \
               'Parameters Z, U: Arrays must contain data for same number of steps.'

        #create output array
        X_hat = mat(zeros((self.sys.A.shape[0], Z.shape[1])))
        #estimate
        X_hat[:, 0], dummy = self.startEstimation(x_hat_0=x_hat_0, P_0=P_0)
        for k in range(1, Z.shape[1]):
            #causality: an input from the past U[:,k-1] caused the current
            #measured value Z[:,k]
            x_hat_k, P_k = self.estimateStep(Z[:,k], U[:,k-1])
            X_hat[:, k] = x_hat_k

        #TODO: return also P_k
        return N.asarray(X_hat)


class PidController(object):
    """Simple PID controller"""
    def __init__(self, kp, ki, kd, ts=1.):
        """
        Parameter:
            kp:
                Proportional Gain
            ki:
                Integral Gain
            kd:
                Derivative Gain
            ts:
                Time step; default = 1
        """
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.ts = float(ts)
        self.errInt = 0.  #integrated error
        self.errOld = 0.  #error value from last time step

    def computeStep(self, error):
        """
        Compute one new control value.

        - Parameter
            error:
                Current error: float
        - Return
            New control value: float
        """
        #integrate error
        self.errInt += self.ts * error
        #differentiate error
        dErr = (error - self.errOld)/self.ts
        self.errOld = error

        return self.kp * error  +  self.ki * self.errInt  +  self.kd * dErr


#------------------------------------------------------------------------------
#  Some example systems to experiment with the Kalman filter
#------------------------------------------------------------------------------

class SysConstVal(LinSys):
    """
    Very simple system where the state values remain constant.

    The control inputs are ignored but they must be given.
    u_k.shape == (1,1); or U.shape == (1,number_of_steps)

    The constant values (that the Kalman filter should guess later) are
    specified as start values (x_0) for the state vector. The methods
    startSimulation(...) and simulate(...) both take start values.
    """

    def __init__(self, n=2, R=mat('0.1, 0; 0, 0.1')):
        """
        Parameter:
            n:
                Number of constant values
            R:
                Covariance matrix of measurement noise. Must have shape (n,n)
        """
        assert R.shape[0] == n and R.shape[1] == n, \
               'R must be square and must have shape (%d,%d)' % (n,n)

        A = N.identity(n)
        B = ones((n,1))
        H = N.identity(n)
        Q = zeros((n,n))
        LinSys.__init__(self, A, B, H, Q, R)


class SysTank(LinSys):
    """
    Tank with inlet valve and outlet valve.

    u[0] is inlet, u[1] is outlet

    Sketch::
              -----
     u[0] --> ~~~~~~.
              ----- ~
                ||  ~     ||
                ||~~~~~~~~||......................
                ||~~~~~~~~||                  ^
                ||~~~~~~~~||----              | x
                ||~~~~~~~~~~~~~~ --> u[1]     v
                ||=========-----..................

    Difference Equation:
        x_k = x_{k-1} + ts * u[0] - ts * u[1] + w_k
    Noisy measurement:
        z_k = x_k + v_k
     with:
        ts : temporal step size
        w_k, v_k : noise
   """

    def __init__(self, ts=1., q=0.001, r=0.5):
        """
        Parameter:
            ts:
                Time between steps.
            q:
                Variance of process noise: float
            r:
                Variance of measurement noise: float
        """
        A = mat('1.')
        B = mat([[ts, -ts]])
        H = mat('1.')
        Q = mat(float(q))
        R = mat(float(r))
        LinSys.__init__(self, A, B, H, Q, R)


class SysSpringPendulum(LinSys):
    """
    One dimensional spring pendulum with damper.

    Sketch::
                            x, v, a
                          :--------->
               ||         :
               ||   c1    +------+
               ||/\/\/\/\/|      |
               ||   d1    |      |
               || |-----  |   m  |  vibrating mass
               ||=|   ]===|      |
               || |-----  |      |---> F_i  forces
               ||         +------+
               ||

    Differential equations:
        sum(F_i) = m*a = -c1 * x - d1 * v + F_load + F_control
        a = diff(v,t)
        v = diff(x, t)

    Difference equations:
        v_k = v_{k-1}  -  c1*ts/m * x_k  -  d1*ts/m * v_k
             +  ts/m * F_load  +  ts/m * F_control  +  noiseP1_k
        x_k = x_{k-1}  +  ts * v_k  +  noiseP2_k

        with:
            ts : time step size (delta t)
            noiseP1_k, noiseP2_k : process noise

     Measurement model:
        z1_k = x_k + noiseM2_k  -  you can only measure the position

    state vector:           x = mat([[v_k], [x_k]])
    (control) input vector: u = mat([[F_load], [F_control]])
    """

    def __init__(self, m=1., c1=0.04, d1=0.1, ts=1., q=0.001, r=0.1):
        """
        Parameter:
            m:  Mass                           : float
            c1: Spring constant                : float
            d1: Damping constant               : float
            ts: Time between steps.            : float
            q:  Variance of process noise.     : float
            r:  Variance of measurement noise. : float
        """
        #compute characteristic properties
        omega = N.sqrt(c1/m)      # circular frequency
        period = 2 * N.pi / omega
        damping = d1/N.sqrt(c1*m) # if damping>1 then: damping is strong
        dStrong = 'yes' if damping > 1 else 'no'
        print 'Period (T): %g. Circular frequency: %g.' % (period, omega)
        print 'Damping: %g; strong damping? %s' % (damping, dStrong)

        A = mat([[1.-d1*ts/m,  -c1*ts/m],
                 [ts,          1.     ]])
        B = mat([[ts/m,  ts/m],
                 [0.,    0.  ]])
        H = mat([[0., 1.]])

        Q = mat([[q,  0.   ],  # process noise consists mainly of forces
                 [0., q/100]]) # only very few noise is added to the position
        R = mat([[r]])

        LinSys.__init__(self, A, B, H, Q, R)


#------------------------------------------------------------------------------
#  Do some experiments
#------------------------------------------------------------------------------

if __name__ == '__main__':
    # Estimate two constant values from noisy data
    #--------------------------------------------------------------------------
    def experiment_ConstantValue():
        # 'simulate' the system - results in two constants + Gaussian noise
        #     Creating a constant with noise could be done much more easy:
        #     Z0 = N.random.normal(loc=2.5, scale=2., size=(100,))
        constSys = SysConstVal(n=2, R=mat('2., 0; 0, 0.02'))
        U=N.zeros((1, 100)) #control input (will be ignored)
        X, Z = constSys.simulate(x_0=mat('2.5; -1.2'), U=U)
        #estimate the value from the noisy data
        #constSys.Q = mat(N.identity(2)) * 1e-2 #pretend that there is process noise
        kFilt = KalmanFilter(constSys)
        X_hat = kFilt.estimate(Z, U)

        #create plot
        pylab.figure()

        pylab.plot(X[0], 'b-', label='val 0, true', linewidth=2)
        pylab.plot(Z[0], 'b:x', label='val 0, noisy', linewidth=1)
        pylab.plot(X_hat[0], 'b-', label='val 0, estimated', linewidth=1)

        pylab.plot(X[1], 'g-', label='val 1, true', linewidth=2)
        pylab.plot(Z[1], 'g:x', label='val 1, noisy', linewidth=1)
        pylab.plot(X_hat[1], 'g-', label='val 1, estimated', linewidth=1)

        pylab.xlabel('step')
        pylab.legend()
        pylab.savefig('kalman_filter_constantValue.png', dpi=75)
    experiment_ConstantValue() #enable/disable here

    # Simulate simple tank and estimate amount in tank
    #--------------------------------------------------------------------------
    def experiment_tank():
        # create the (control) inputs
        valveIn  =    ones(100)*0.1
        valveOut = r_[zeros(50), ones(10)*0.3, ones(40)*0.08]
        U = vstack((valveIn, valveOut))
        # simulate the system
        tank = SysTank(q=0.002, r=2)
        X, Z = tank.simulate(x_0=mat('2.'), U=U)
        # estimate tank level from noisy data
        kFilt = KalmanFilter(tank)
        X_hat = kFilt.estimate(Z, U)

        #try if smoothing is better or worse
        kernelSize = 15
        kernel = ones(kernelSize) / kernelSize
        Z_smooth = N.zeros_like(Z.ravel())
        Z_smooth[kernelSize-1:] = N.correlate(Z.ravel(), kernel, 'valid')

        #create plot
        pylab.figure()
        pylab.plot(X[0], 'b-', label='true states', linewidth=2)
        pylab.plot(Z[0], 'b:x', label='noisy measurements', linewidth=1)
        pylab.plot(X_hat[0], 'b-', label='estimated states', linewidth=1)
        pylab.plot(Z_smooth, 'g-', label='smoothed measurements', linewidth=1)
        pylab.xlabel('step')
        pylab.legend(loc='lower right')
        pylab.title('Tank')
        pylab.savefig('kalman_filter_tank.png', dpi=75)
    experiment_tank() #enable/disable here

    # Simulate spring pendulum and estimate position
    #--------------------------------------------------------------------------
    def experiment_SpringPendulum():
        #diagram used in all sub-experiments
        def plotFigure(X, X_hat, Z, U, SetPoint=None,
                       title='Spring Pendulum', legendLoc='upper left'):
            X = N.asarray(X); X_hat = N.asarray(X_hat); Z = N.asarray(Z)
            U = N.asarray(U);
            pylab.figure()
            if SetPoint is not None:
                SetPoint = N.asarray(SetPoint)
                pylab.plot(SetPoint[0],'y-',  label='set-point', linewidth=1)
            # input vector:     u = mat([[F_load], [F_control]])
            pylab.plot(U[0],     'r-',  label='F load',      linewidth=1)
            pylab.plot(U[1],     'm-',  label='F control',   linewidth=1)
            # state vector:    x = mat([[v_k], [x_k]])
            pylab.plot(X[0],     'g-',  label='v true',      linewidth=2)
            pylab.plot(X_hat[0], 'g-',  label='v estimated', linewidth=1)
            pylab.plot(X[1],     'b-',  label='x true',      linewidth=2)
            pylab.plot(Z[0],     'b:x', label='x measured',  linewidth=1)
            pylab.plot(X_hat[1], 'b-',  label='x estimated', linewidth=1)
            pylab.xlabel('step')
            pylab.legend(loc=legendLoc)
            pylab.title(title)

        ts = 0.5     #step time - shorten for better estimation results
        nSteps = 200 #duration of simulation
        #Create system and filter - same for all experiments.
        #
        # The spring pendulum is only weakly damped.
        # Noise is mainly added to the velocity; this means random *forces* act
        # on the pendulum. Very little noise is added to the position.
        # The random forces let the pendulum oscillate strongly.
        # Even though the Kalman filter can only see the position, it can
        # estimate the speed fairly well.
        pendu = SysSpringPendulum(m=1., c1=0.04, d1=0.1, ts=ts, q=0.004, r=2.)
        kFilt = KalmanFilter(pendu)

        # See how much the pendulum oscillates only because of noise.
        def onlyNoise():
            # create the inputs
            U = zeros((2, nSteps))
            # simulate the system
            X, Z = pendu.simulate(x_0=mat('0.; 0.'), U=U)

            # estimate pendulum position and speed
            X_hat = kFilt.estimate(Z, U)
            plotFigure(X, X_hat, Z, U,
                       title='Spring Pendulum - Only Noise')
            pylab.savefig('kalman_filter_springPendulum_onlyNoise.png', dpi=75)
        #onlyNoise()

        # Excite the spring pendulum with an impulse
        def impulseExcitation():
            # create the inputs
            # input vector:     u = mat([[F_load], [F_control]])
            U = zeros((2, nSteps))
            U[0, int(nSteps/2)] = 2.5/ts #impulse of load force in middle
            # simulate the system
            X, Z = pendu.simulate(x_0=mat('0.; 0.'), U=U)

            # estimate pendulum position and speed
            X_hat = kFilt.estimate(Z, U)
            plotFigure(X, X_hat, Z, U,
                       title='Spring Pendulum - Impulse Excitation')
            pylab.savefig('kalman_filter_springPendulum_impulseExcitation.png', dpi=75)
        impulseExcitation()

        # Excite the spring pendulum with an impulse; but Kalman filter can not
        # see force.
        def impulseExcitation_KalmanNotSeeForce():
            # create the inputs
            # input vector:     u = mat([[F_load], [F_control]])
            U = zeros((2, nSteps))
            U[0, int(nSteps/2)] = 2.5/ts #impulse of load force in middle
            # simulate the system
            X, Z = pendu.simulate(x_0=mat('0.; 0.'), U=U)

            #estimate pendulum position and speed, but Kalman filter can not
            #see F_load (exciting force)
            Uzero = N.zeros_like(U) #Pretend F_load is always 0
            X_hat = kFilt.estimate(Z, Uzero)
            plotFigure(X, X_hat, Z, U,
                   title='Spring Pendulum - Kalman filter can not see force')
            pylab.savefig('kalman_filter_springPendulum_KalmanNotSeeForce.png', dpi=75)
        impulseExcitation_KalmanNotSeeForce()

        # Controller for spring pendulum; use Kalman filter to get position
        def springPendulumController():
            # set-point for position controller
            SetPoint = mat(zeros(nSteps))
            SetPoint[0, int(nSteps/2):] = 8.
            #controller for position (as usual)
            controllerPos = PidController(kp=0.3, ki=0.01, kd=0.0, ts=ts)
            #controller to add damping, sees -speed, drives speed low
            controllerDamp = PidController(kp=1.0, ki=0.0, kd=0.1, ts=ts)
            #create arrays for computation results
            X = mat(zeros((2,nSteps)))
            X_hat = mat(zeros((2,nSteps)))
            Z = mat(zeros((1,nSteps)))
            U = mat(zeros((2,nSteps)))
            #start simulation and filter (iteration 0)
            x_k, z_k = pendu.startSimulation(x_0=mat('0.; 0.'))
            x_hat_k, P = kFilt.startEstimation()
            u_k = mat('0.; 0.')
            # main loop
            for k in range(0, nSteps):
                X[:, k] = x_k
                Z[:, k] = z_k
                X_hat[:, k] = x_hat_k
                U[:, k] = u_k
                #compute control values
                # state vector:    x = mat([[v_k], [x_k]])
                #pos, speed = X[1, k],     X[0, k] #controller sees true data
                pos, speed = X_hat[1, k], X_hat[0, k] #control with estimated data
                sp = SetPoint[0, k]
                f_control_k = controllerPos.computeStep(sp-pos) + \
                              controllerDamp.computeStep(-speed)
                # input vector:    u = mat([[F_load], [F_control]])
                u_k = mat([[0.], [f_control_k]])
                #-- simulate and estimate new values --
                x_k, z_k = pendu.simulateStep(u_k)
                x_hat_k, P = kFilt.estimateStep(z_k, u_k)

            plotFigure(X, X_hat, Z, U, SetPoint,
                title='Spring Pendulum - Controller for position and damping')
            pylab.savefig('kalman_filter_springPendulum_springPendulumController.png', dpi=75)
        springPendulumController() #enable/disable here

    experiment_SpringPendulum()  #enable/disable here


    pylab.show()
