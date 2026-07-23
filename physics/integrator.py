"""
RK4 Fixed-Step Integrator

Integrates the full drone state using 4th-order Runge-Kutta method.
Decoupled from GUI render rate.
"""

import numpy as np
from typing import Callable


class RK4Integrator:
    """
    Runge-Kutta 4th-order integrator for ODE: dy/dt = f(t, y)
    """
    
    def __init__(self, dt: float = 0.001):
        """
        Initialize integrator.
        
        Args:
            dt: fixed timestep (seconds)
        """
        self.dt = dt
    
    def step(
        self,
        state: np.ndarray,
        state_derivative_fn: Callable,
        *args
    ) -> np.ndarray:
        """
        Perform one RK4 integration step.
        
        y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        where:
          k1 = f(t_n, y_n)
          k2 = f(t_n + dt/2, y_n + (dt/2)*k1)
          k3 = f(t_n + dt/2, y_n + (dt/2)*k2)
          k4 = f(t_n + dt, y_n + dt*k3)
        
        Args:
            state: current state vector
            state_derivative_fn: function that computes ds/dt (state, *args) -> ds/dt
            *args: additional arguments to pass to state_derivative_fn
        
        Returns:
            Updated state vector
        """
        # RK4 stages
        k1 = state_derivative_fn(state, *args)
        k2 = state_derivative_fn(state + 0.5 * self.dt * k1, *args)
        k3 = state_derivative_fn(state + 0.5 * self.dt * k2, *args)
        k4 = state_derivative_fn(state + self.dt * k3, *args)
        
        # Update state
        state_new = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return state_new
