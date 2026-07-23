"""
Unit tests for rotor model.

Tests motor dynamics, thrust scaling, and drag torque.
"""

import numpy as np
import pytest
from physics.rotor import Rotor


class TestRotorBasic:
    """
    Test basic rotor properties.
    """
    
    def test_rotor_initialization(self):
        """Test rotor is initialized correctly."""
        rotor = Rotor(
            kv=920,
            k_thrust=9.65e-5,
            k_drag=1.2e-6,
            i_rotor=1e-5,
            direction=1,
            r_internal=0.01
        )
        
        assert rotor.kv == 920
        assert rotor.k_thrust == 9.65e-5
        assert rotor.k_drag == 1.2e-6
        assert rotor.i_rotor == 1e-5
        assert rotor.direction == 1
        assert rotor.r_internal == 0.01
        assert rotor.rpm == 0.0  # Start at rest
        assert rotor.throttle == 0.0
    
    def test_throttle_clipping(self):
        """Test throttle is clipped to [0, 1]."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.set_throttle(1.5)
        assert rotor.throttle == 1.0
        
        rotor.set_throttle(-0.5)
        assert rotor.throttle == 0.0


class TestRotorThrust:
    """
    Test thrust calculation.
    """
    
    def test_thrust_zero_at_zero_rpm(self):
        """Test thrust is zero when rotor is not spinning."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 0.0
        thrust = rotor.get_thrust(rho=1.225)
        
        assert thrust == 0.0
    
    def test_thrust_increases_with_rpm(self):
        """Test thrust scales with RPM squared."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 5000
        t1 = rotor.get_thrust(rho=1.225)
        
        rotor.rpm = 10000
        t2 = rotor.get_thrust(rho=1.225)
        
        # Thrust should scale as RPM^2
        # So doubling RPM should quadruple thrust
        assert t2 > t1
        # Check approximately 4x relationship
        assert 3.5 < t2 / t1 < 4.5
    
    def test_thrust_scales_with_air_density(self):
        """Test thrust scales with air density."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 8000
        
        # At sea level
        t_sea_level = rotor.get_thrust(rho=1.225)
        
        # At high altitude (lower density)
        t_high_alt = rotor.get_thrust(rho=0.9)
        
        # Thrust should decrease at lower density
        assert t_high_alt < t_sea_level
        
        # Check scaling: T ∝ ρ
        ratio = t_high_alt / t_sea_level
        expected_ratio = 0.9 / 1.225
        assert abs(ratio - expected_ratio) < 0.01
    
    def test_thrust_positive(self):
        """Test thrust is always non-negative."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        for rpm in [-1000, 0, 1000, 5000, 10000]:
            rotor.rpm = rpm
            thrust = rotor.get_thrust(rho=1.225)
            assert thrust >= 0.0


class TestRotorDragTorque:
    """
    Test drag torque calculation.
    """
    
    def test_drag_torque_zero_at_zero_rpm(self):
        """Test drag torque is zero at zero RPM."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 0.0
        torque = rotor.get_drag_torque(rho=1.225)
        
        assert torque == 0.0
    
    def test_drag_torque_increases_with_rpm(self):
        """Test drag torque scales with RPM squared."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 5000
        q1 = rotor.get_drag_torque(rho=1.225)
        
        rotor.rpm = 10000
        q2 = rotor.get_drag_torque(rho=1.225)
        
        # Drag torque should also scale as RPM^2
        assert q2 > q1
        assert 3.5 < q2 / q1 < 4.5
    
    def test_drag_torque_scales_with_air_density(self):
        """Test drag torque scales with air density."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 8000
        
        q_sea_level = rotor.get_drag_torque(rho=1.225)
        q_high_alt = rotor.get_drag_torque(rho=0.9)
        
        assert q_high_alt < q_sea_level
        
        ratio = q_high_alt / q_sea_level
        expected_ratio = 0.9 / 1.225
        assert abs(ratio - expected_ratio) < 0.01


class TestRotorCurrent:
    """
    Test current draw estimation.
    """
    
    def test_current_zero_at_zero_rpm(self):
        """Test current is zero when rotor is idle."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 0.0
        current = rotor.get_current(v_battery=11.1, rho=1.225)
        
        assert current == 0.0
    
    def test_current_increases_with_rpm(self):
        """Test current increases with rotor speed."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 5000
        i1 = rotor.get_current(v_battery=11.1, rho=1.225)
        
        rotor.rpm = 10000
        i2 = rotor.get_current(v_battery=11.1, rho=1.225)
        
        # Current should increase (approximately scales with drag torque * RPM)
        assert i2 > i1
    
    def test_current_zero_voltage(self):
        """Test current handling with zero voltage."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.rpm = 8000
        current = rotor.get_current(v_battery=0.0, rho=1.225)
        
        # Should handle gracefully
        assert current == 0.0
    
    def test_current_positive(self):
        """Test current is always non-negative."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        for rpm in [0, 5000, 10000]:
            rotor.rpm = rpm
            current = rotor.get_current(v_battery=11.1, rho=1.225)
            assert current >= 0.0


class TestRotorDynamics:
    """
    Test rotor dynamics (RPM response to throttle).
    """
    
    def test_rpm_increases_with_throttle(self):
        """Test RPM increases when throttle is applied."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.set_throttle(0.5)
        rotor.update(v_battery=11.1, rho=1.225, dt=0.1)
        
        rpm_after_one_step = rotor.rpm
        assert rpm_after_one_step > 0.0
    
    def test_rpm_approaches_steady_state(self):
        """Test RPM approaches steady state with constant throttle."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor.set_throttle(0.5)
        
        # Run multiple steps
        for _ in range(1000):
            rotor.update(v_battery=11.1, rho=1.225, dt=0.001)
        
        # RPM should have reached near steady state
        assert rotor.rpm > 1000  # Should be spinning
        assert rotor.rpm < 12000  # But not at unrealistic levels
    
    def test_rpm_increases_with_voltage(self):
        """Test higher battery voltage produces higher RPM."""
        rotor1 = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        rotor2 = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        rotor1.set_throttle(0.5)
        rotor2.set_throttle(0.5)
        
        # Run rotor1 at higher voltage
        for _ in range(100):
            rotor1.update(v_battery=12.0, rho=1.225, dt=0.001)
            rotor2.update(v_battery=10.0, rho=1.225, dt=0.001)
        
        # Higher voltage should produce higher RPM
        assert rotor1.rpm > rotor2.rpm
    
    def test_rpm_decreases_when_throttle_reduced(self):
        """Test RPM decreases when throttle is reduced."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        # Spin up
        rotor.set_throttle(0.8)
        for _ in range(100):
            rotor.update(v_battery=11.1, rho=1.225, dt=0.001)
        
        rpm_high = rotor.rpm
        
        # Reduce throttle
        rotor.set_throttle(0.3)
        for _ in range(100):
            rotor.update(v_battery=11.1, rho=1.225, dt=0.001)
        
        rpm_low = rotor.rpm
        
        # RPM should decrease
        assert rpm_low < rpm_high
    
    def test_rpm_zero_at_zero_throttle(self):
        """Test RPM decays to zero at zero throttle."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        # Spin up
        rotor.set_throttle(0.7)
        for _ in range(200):
            rotor.update(v_battery=11.1, rho=1.225, dt=0.001)
        
        rpm_spinning = rotor.rpm
        assert rpm_spinning > 1000
        
        # Kill throttle
        rotor.set_throttle(0.0)
        for _ in range(500):
            rotor.update(v_battery=11.1, rho=1.225, dt=0.001)
        
        # RPM should approach zero (but may have lag)
        assert rotor.rpm < 100


class TestRotorReset:
    """
    Test rotor reset.
    """
    
    def test_rotor_reset(self):
        """Test rotor resets to idle state."""
        rotor = Rotor(kv=920, k_thrust=9.65e-5, k_drag=1.2e-6, i_rotor=1e-5)
        
        # Spin up and apply throttle
        rotor.set_throttle(0.8)
        for _ in range(100):
            rotor.update(v_battery=11.1, rho=1.225, dt=0.001)
        
        assert rotor.rpm > 1000
        assert rotor.throttle == 0.8
        
        # Reset
        rotor.reset()
        
        assert rotor.rpm == 0.0
        assert rotor.throttle == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
