"""
Unit tests for battery model.

Tests voltage sag, SOC tracking, C-rate limiting, and thermal effects.
"""

import numpy as np
import pytest
from physics.battery import Battery


class TestBatteryBasic:
    """
    Test basic battery properties and initialization.
    """
    
    def test_battery_initialization(self):
        """Test battery is initialized correctly."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012,
            c_rating=50,
            num_cells=3
        )
        
        assert battery.capacity_mah == 5000
        assert battery.capacity_ah == 5.0
        assert battery.nominal_voltage == 11.1
        assert battery.r_internal_base == 0.012
        assert battery.c_rating == 50
        assert battery.soc == 1.0  # Start fully charged
        assert battery.temperature == 25.0  # Ambient
    
    def test_battery_soc_clipping(self):
        """Test SOC is clipped to [0, 1]."""
        battery = Battery(capacity_mah=5000, nominal_voltage=11.1)
        
        assert battery.get_soc() == 1.0
        battery.soc = 1.5
        assert battery.get_soc() == 1.0  # Clipped to max
        
        battery.soc = -0.5
        assert battery.get_soc() == 0.0  # Clipped to min


class TestBatteryVoltage:
    """
    Test voltage calculation and sag.
    """
    
    def test_voltage_no_load(self):
        """Test OCV at no load."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        # At full charge, should be near nominal
        v_ocv = battery._get_ocv()
        assert v_ocv > 11.0
        assert v_ocv <= 11.1
    
    def test_voltage_drop_under_load(self):
        """Test voltage sag under current draw."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        v_no_load = battery.get_terminal_voltage(current=0.0)
        v_with_load = battery.get_terminal_voltage(current=20.0)  # 20A draw
        
        # Voltage should drop under load
        assert v_with_load < v_no_load
        
        # Voltage drop should be approximately I * R
        expected_drop = 20.0 * 0.012  # 0.24V
        actual_drop = v_no_load - v_with_load
        assert 0.2 < actual_drop < 0.3  # Allow some tolerance
    
    def test_voltage_increases_with_load_decrease(self):
        """Test that voltage recovers as load decreases."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        v1 = battery.get_terminal_voltage(current=10.0)
        v2 = battery.get_terminal_voltage(current=5.0)
        v3 = battery.get_terminal_voltage(current=0.0)
        
        # Voltage should increase with decreasing load
        assert v1 < v2 < v3
    
    def test_voltage_discharge_curve(self):
        """Test that voltage drops as battery discharges (SOC decreases)."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        v_full = battery.get_terminal_voltage(current=0.0)
        
        # Discharge to 50% SOC
        battery.soc = 0.5
        v_half = battery.get_terminal_voltage(current=0.0)
        
        # Voltage should decrease with SOC
        assert v_half < v_full
        
        # Further discharge
        battery.soc = 0.1
        v_low = battery.get_terminal_voltage(current=0.0)
        assert v_low < v_half


class TestBatterySOC:
    """
    Test state of charge tracking.
    """
    
    def test_soc_coulomb_counting(self):
        """Test SOC decreases with current draw via coulomb counting."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        initial_soc = battery.get_soc()
        
        # Draw 10A for 1 second
        # dSOC = -I / (capacity_Ah * 3600) * dt
        # dSOC = -10 / (5 * 3600) * 1 = -10/18000 ≈ -0.000556
        battery.update(current=10.0, dt=1.0)
        
        final_soc = battery.get_soc()
        assert final_soc < initial_soc
        
        # Check approximately correct
        expected_dq = 10.0 / (5.0 * 3600.0)
        assert abs((initial_soc - final_soc) - expected_dq) < 1e-6
    
    def test_soc_discharge_to_empty(self):
        """Test battery discharges to SOC=0."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        # Draw at 1C rate (5A) for 1 hour = 5 Ah
        # This should fully discharge the battery
        current = 5.0  # 1C rate
        dt = 3600.0  # 1 hour in seconds
        
        battery.update(current=current, dt=dt)
        
        # Should be fully discharged
        assert battery.get_soc() <= 0.0


class TestBatteryCRate:
    """
    Test C-rate limiting.
    """
    
    def test_max_current_at_full_charge(self):
        """Test max current is highest when battery is full."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012,
            c_rating=50
        )
        
        i_max_full = battery.get_max_current()
        
        # Max current should be capacity_Ah * C_rating
        # = 5.0 Ah * 50 C = 250 A
        assert abs(i_max_full - 250.0) < 1.0
    
    def test_max_current_reduces_with_discharge(self):
        """Test max current decreases as battery discharges."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012,
            c_rating=50
        )
        
        i_max_100 = battery.get_max_current()
        
        battery.soc = 0.5  # 50% SOC
        i_max_50 = battery.get_max_current()
        
        battery.soc = 0.1  # 10% SOC
        i_max_10 = battery.get_max_current()
        
        # Max current should scale linearly with SOC
        assert i_max_100 > i_max_50 > i_max_10
        assert abs(i_max_50 - i_max_100 * 0.5) < 1.0
        assert abs(i_max_10 - i_max_100 * 0.1) < 1.0


class TestBatteryThermal:
    """
    Test thermal model.
    """
    
    def test_temperature_increases_under_load(self):
        """Test battery temperature rises when drawing current."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        initial_temp = battery.temperature
        
        # Draw high current for 10 seconds
        battery.update(current=50.0, dt=10.0)
        
        final_temp = battery.temperature
        
        # Temperature should increase
        assert final_temp > initial_temp
    
    def test_temperature_affects_resistance(self):
        """Test that temperature increases internal resistance."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        r_cold = battery._get_internal_resistance()
        
        # Heat the battery
        battery.temperature = 50.0
        r_hot = battery._get_internal_resistance()
        
        # Resistance should increase with temperature
        assert r_hot > r_cold
    
    def test_temperature_affects_voltage(self):
        """Test that higher temperature reduces terminal voltage."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        v_cold = battery.get_terminal_voltage(current=20.0)
        
        # Heat battery
        battery.temperature = 50.0
        v_hot = battery.get_terminal_voltage(current=20.0)
        
        # Voltage should be lower when hot (higher resistance)
        assert v_hot < v_cold


class TestBatteryReset:
    """
    Test battery reset.
    """
    
    def test_battery_reset_to_full(self):
        """Test battery resets to full charge."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        # Discharge battery
        battery.soc = 0.3
        battery.temperature = 50.0
        battery.v_sag = 0.5
        
        # Reset
        battery.reset()
        
        # Check reset
        assert battery.soc == 1.0
        assert battery.temperature == 25.0
        assert battery.v_sag == 0.0


class TestBatteryIntegration:
    """
    Integration tests for realistic discharge scenarios.
    """
    
    def test_realistic_flight(self):
        """Simulate a realistic flight profile."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012,
            c_rating=50
        )
        
        # Phase 1: Hover at 50% throttle (~25A draw) for 10 minutes
        hover_current = 25.0
        hover_time = 10 * 60  # 10 minutes in seconds
        
        battery.update(current=hover_current, dt=hover_time)
        
        # Battery should have discharged
        assert battery.get_soc() < 1.0
        
        # But not completely (5000mAh at 25A for 10 min = 4.17 Ah used)
        assert battery.get_soc() > 0.17  # 5 - 4.17 = 0.83 Ah left, ~16.6% SOC
    
    def test_high_current_draw(self):
        """Test battery behavior under high current draw."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012,
            c_rating=50
        )
        
        # Max current at 1C = 5A, at 50C = 250A
        max_current = battery.get_max_current()
        assert max_current == 250.0
        
        # Try to request more than max
        # (In real system, this would be limited by mixer)
        requested_current = 300.0
        # Current should be limited to max
        current_limited = min(requested_current, max_current)
        assert current_limited == 250.0
    
    def test_voltage_sag_recovery(self):
        """Test voltage sag dynamics (transient response)."""
        battery = Battery(
            capacity_mah=5000,
            nominal_voltage=11.1,
            internal_resistance=0.012
        )
        
        # Get baseline voltage
        v_baseline = battery.get_terminal_voltage(current=0.0)
        
        # Apply sudden high current
        battery.update(current=100.0, dt=0.1)
        v_sag = battery.get_terminal_voltage(current=100.0)
        
        # Voltage should sag
        assert v_sag < v_baseline
        
        # Voltage sag should be non-zero (transient response)
        assert battery.v_sag > 0.0
        
        # Let voltage recover
        battery.update(current=100.0, dt=1.0)  # Long time
        v_recovered = battery.get_terminal_voltage(current=100.0)
        
        # Voltage sag should decrease (but steady-state drop remains)
        # v_recovered should be higher than initial v_sag
        # but lower than baseline (due to steady-state IR drop)
        assert v_sag < v_recovered < v_baseline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
