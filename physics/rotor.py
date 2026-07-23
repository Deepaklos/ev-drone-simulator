"""
Rotor Model - Motor Electrical Dynamics + Blade Element Momentum Theory (BEMT)

Models the rotor as:
  1. Motor: PWM → motor torque → RPM dynamics
  2. Thrust: RPM → thrust via BEMT (thrust ∝ ρ * RPM²)
  3. Drag: RPM → drag torque (back-torque on frame)

Equations:
  Motor electrical: τ_motor = K_t * (PWM * V - K_e * RPM)
  RPM dynamics: dRPM/dt = (τ_motor - τ_drag) / I_rotor
  Thrust: T = K_thrust * (ρ / ρ_0) * RPM²
  Drag torque: Q = K_drag * (ρ / ρ_0) * RPM²
  Current draw: I = Q * RPM / V
"""

import numpy as np
from typing import Tuple


class Rotor:
    """
    Represents a single rotor (motor + propeller).
    
    Parameters:
      - K_v: motor velocity constant (RPM/V)
      - R_internal: motor internal resistance (Ω)
      - I_max: max motor current (A) — limits thrust under low voltage
      - K_thrust: thrust coefficient (N / (ρ * RPM²))
      - K_drag: drag torque coefficient (N*m / (ρ * RPM²))
      - I_rotor: rotor rotational inertia (kg*m²)
      - direction: +1 for CCW, -1 for CW (determines yaw moment)
    """
    
    def __init__(
        self,
        kv: float,
        k_thrust: float,
        k_drag: float,
        i_rotor: float,
        direction: int = 1,
        r_internal: float = 0.01
    ):
        """
        Initialize a rotor.
        
        Args:
            kv: motor velocity constant (RPM/V)
            k_thrust: thrust coefficient [N/(kg/m³ * RPM²)]
            k_drag: drag torque coefficient [N*m/(kg/m³ * RPM²)]
            i_rotor: rotor inertia (kg*m²)
            direction: rotor spin direction (+1 or -1)
            r_internal: motor internal resistance (Ω)
        """
        self.kv = kv  # RPM/V
        self.k_e = 60.0 / (2 * np.pi * kv)  # Back-EMF constant (V*s/rad)
        self.k_t = 8.27 / kv  # Torque constant (N*m/A) — derived from motor equations
        self.r_internal = r_internal  # Ohms
        
        self.k_thrust = k_thrust
        self.k_drag = k_drag
        self.i_rotor = i_rotor
        self.direction = direction  # +1 or -1
        
        self.rpm = 0.0  # Current RPM
        self.throttle = 0.0  # Commanded throttle (0-1)
        
        # Motor time constant for first-order lag
        self.tau_motor = 0.02  # seconds (50 Hz bandwidth)
    
    def set_throttle(self, throttle: float):
        """
        Set motor throttle command.
        
        Args:
            throttle: throttle (0-1)
        """
        self.throttle = np.clip(throttle, 0.0, 1.0)
    
    def get_rpm(self) -> float:
        """Get current rotor RPM."""
        return self.rpm
    
    def get_thrust(self, rho: float) -> float:
        """
        Compute thrust from current RPM.
        
        T = K_thrust * (ρ / ρ_0) * RPM²
        
        where ρ_0 = 1.225 kg/m³ (sea level)
        
        Args:
            rho: air density (kg/m³)
        
        Returns:
            Thrust (N)
        """
        rho_0 = 1.225
        rpm_rad_s = self.rpm * 2 * np.pi / 60.0
        thrust = self.k_thrust * (rho / rho_0) * (rpm_rad_s ** 2)
        return max(0.0, thrust)  # Thrust cannot be negative
    
    def get_drag_torque(self, rho: float) -> float:
        """
        Compute drag torque (back-torque on frame due to propeller drag).
        
        Q = K_drag * (ρ / ρ_0) * RPM²
        
        This is the reaction torque that causes yaw moment.
        
        Args:
            rho: air density (kg/m³)
        
        Returns:
            Drag torque (N*m)
        """
        rho_0 = 1.225
        rpm_rad_s = self.rpm * 2 * np.pi / 60.0
        torque = self.k_drag * (rho / rho_0) * (rpm_rad_s ** 2)
        return torque
    
    def get_current(self, v_battery: float, rho: float) -> float:
        """
        Estimate current draw from motor.
        
        I ≈ (P_out) / V_battery, where P_out = Q * ω
        
        Args:
            v_battery: battery voltage (V)
            rho: air density (kg/m³)
        
        Returns:
            Current draw (A)
        """
        if v_battery < 0.1:
            return 0.0
        
        rpm_rad_s = self.rpm * 2 * np.pi / 60.0
        torque = self.get_drag_torque(rho)
        power_output = torque * rpm_rad_s
        current = power_output / v_battery
        return max(0.0, current)
    
    def update(self, v_battery: float, rho: float, dt: float):
        """
        Update rotor RPM based on throttle and battery voltage.
        
        Motor dynamics (first-order lag):
          τ_motor = K_t * (throttle * V_battery - K_e * RPM)
          dRPM/dt = (τ_motor - τ_drag) / I_rotor
        
        Args:
            v_battery: battery terminal voltage (V)
            rho: air density (kg/m³)
            dt: timestep (seconds)
        """
        # Clamp battery voltage
        v_battery = max(0.0, v_battery)
        
        # Motor back-EMF (proportional to RPM)
        rpm_rad_s = self.rpm * 2 * np.pi / 60.0
        back_emf = self.k_e * rpm_rad_s
        
        # Motor voltage: V_motor = throttle * V_battery
        v_motor = self.throttle * v_battery
        
        # Limit current to prevent motor burnout
        i_max_motor = 50.0  # 50 A max (configurable)
        
        # Motor current (limited)
        if v_battery > 0.1:
            current_ideal = (v_motor - back_emf) / self.r_internal
            current = np.clip(current_ideal, 0.0, i_max_motor)
        else:
            current = 0.0
        
        # Motor torque
        tau_motor = self.k_t * current
        
        # Drag torque (air resistance on propeller)
        tau_drag = self.get_drag_torque(rho)
        
        # RPM dynamics: dRPM/dt = (τ_motor - τ_drag) / I_rotor (rad/s²)
        # Convert to RPM/s: dRPM/dt = dω/dt * 60 / (2π)
        if self.i_rotor > 1e-9:
            d_rpm_rad_s = (tau_motor - tau_drag) / self.i_rotor
            d_rpm = d_rpm_rad_s * 60.0 / (2 * np.pi)
        else:
            d_rpm = 0.0
        
        # Update RPM with first-order lag to smoothen response
        # dRPM_smooth/dt = (RPM_target - RPM) / tau
        rpm_target = max(0, self.rpm + d_rpm * dt)
        self.rpm += (rpm_target - self.rpm) / self.tau_motor * dt
        self.rpm = max(0.0, self.rpm)
    
    def reset(self):
        """Reset rotor to standby state."""
        self.rpm = 0.0
        self.throttle = 0.0
