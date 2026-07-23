"""
Unit tests for PID controller.

Tests PID gains, integral windup, and derivative filtering.
"""

import numpy as np
import pytest
from control.pid import PIDController


class TestPIDBasic:
    """
    Test basic PID properties.
    """
    
    def test_pid_initialization(self):
        """Test PID is initialized correctly."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.5, dt=0.01)
        
        assert pid.kp == 1.0
        assert pid.ki == 0.1
        assert pid.kd == 0.5
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
    
    def test_pid_reset(self):
        """Test PID reset clears state."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.5, dt=0.01)
        
        # Apply errors to accumulate state
        for _ in range(10):
            pid.update(error=1.0)
        
        assert pid.integral > 0.0
        assert pid.prev_error != 0.0
        
        # Reset
        pid.reset()
        
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
        assert pid.prev_derivative == 0.0


class TestPIDProportional:
    """
    Test proportional term.
    """
    
    def test_proportional_only(self):
        """Test proportional gain without I and D."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, dt=0.01)
        
        output = pid.update(error=1.0)
        
        # Should be Kp * error
        assert abs(output - 2.0) < 1e-6
    
    def test_proportional_scales_error(self):
        """Test P term scales with error magnitude."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=0.01)
        
        out1 = pid.update(error=1.0)
        pid.reset()
        out2 = pid.update(error=2.0)
        
        # Output should double with error
        assert abs(out2 - 2.0 * out1) < 1e-6


class TestPIDIntegral:
    """
    Test integral term.
    """
    
    def test_integral_accumulation(self):
        """Test integral accumulates error over time."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, dt=0.1)
        
        output1 = pid.update(error=1.0)
        output2 = pid.update(error=1.0)
        
        # Integral should increase
        assert output2 > output1
    
    def test_integral_windup_protection(self):
        """Test integral windup limiting."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, dt=0.01)
        pid.integral_max = 1.0
        pid.integral_min = -1.0
        
        # Apply large error repeatedly
        for _ in range(200):
            pid.update(error=10.0)
        
        # Integral should be clamped
        assert pid.integral <= pid.integral_max
        assert pid.integral >= pid.integral_min


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
