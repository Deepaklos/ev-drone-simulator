"""
PID Controller

Generic PID controller with integral windup protection and derivative filtering.
"""

import numpy as np
from typing import Optional


class PIDController:
    """
    Simple PID controller: u = Kp*e + Ki*∫e - Kd*de/dt
    """
    
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.001):
        """
        Initialize PID controller.
        
        Args:
            kp: proportional gain
            ki: integral gain
            kd: derivative gain
            dt: timestep (seconds)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        
        # Anti-windup limits
        self.integral_max = 1.0
        self.integral_min = -1.0
        
        # Derivative low-pass filter
        self.tau_d = 0.01  # Time constant (seconds)
    
    def update(self, error: float) -> float:
        """
        Compute PID output for given error.
        
        Args:
            error: current error (setpoint - measurement)
        
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, self.integral_min, self.integral_max)
        i_term = self.ki * self.integral
        
        # Derivative term (with low-pass filter to reduce noise)
        derivative = (error - self.prev_error) / self.dt
        self.prev_derivative = (self.tau_d * self.prev_derivative + derivative) / (1 + self.tau_d)
        d_term = self.kd * self.prev_derivative
        
        self.prev_error = error
        
        return p_term + i_term + d_term
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
