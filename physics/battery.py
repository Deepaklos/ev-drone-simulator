"""
Realistic Battery Model - Electrochemical Discharge

Models:
  1. Voltage OCV (open-circuit voltage) as function of SOC (S-shaped LiPo curve)
  2. Internal resistance (constant + temperature-dependent)
  3. Terminal voltage: V = V_ocv - I*R_internal - V_sag (voltage sag under transient load)
  4. SOC tracking via coulomb counting: dSOC/dt = -I / Q_nominal
  5. C-rate limiting: max current ∝ capacity and C-rating
  6. Thermal model: I²R heating → temperature rise → R increase
"""

import numpy as np
from typing import Dict, List


class Battery:
    """
    Represents a LiPo/Li-ion battery.
    
    Parameters:
      - capacity: battery capacity (mAh)
      - nominal_voltage: nominal voltage (V)
      - internal_resistance: base internal resistance (Ω)
      - c_rating: max discharge C-rate (e.g., 50C)
      - num_cells: number of cells in series (for LiPo: 1S=3.7V, 3S=11.1V, etc.)
    """
    
    def __init__(
        self,
        capacity_mah: float,
        nominal_voltage: float,
        internal_resistance: float,
        c_rating: float = 50.0,
        num_cells: int = 3
    ):
        """
        Initialize battery.
        
        Args:
            capacity_mah: capacity in mAh
            nominal_voltage: nominal voltage (V)
            internal_resistance: base internal resistance (Ω)
            c_rating: C-rate (A per mAh)
            num_cells: number of cells in series
        """
        self.capacity_mah = capacity_mah  # mAh
        self.capacity_ah = capacity_mah / 1000.0  # Ah
        self.nominal_voltage = nominal_voltage  # V
        self.r_internal_base = internal_resistance  # Ω (at 25°C)
        self.c_rating = c_rating  # C
        self.num_cells = num_cells
        
        # Thermal model
        self.temperature = 25.0  # °C (ambient)
        self.mass = 0.150  # kg (typical 3S 5000mAh LiPo)
        self.c_thermal = 1500.0  # J/(kg*K) — specific heat capacity
        self.h_convection = 5.0  # W/(m²*K) — convection heat transfer
        self.area_thermal = 0.01  # m² — surface area
        
        # State
        self.soc = 1.0  # State of charge (0-1), start fully charged
        self.energy_remaining = self.capacity_ah * nominal_voltage  # Wh
        
        # Voltage sag dynamics (transient response)
        self.v_sag = 0.0
        self.tau_sag = 0.1  # Time constant for sag (seconds)
        
        # Generate discharge curve (LiPo-like S-curve)
        self._generate_discharge_curve()
    
    def _generate_discharge_curve(self):
        """
        Generate realistic LiPo discharge curve: V_ocv(SOC)
        
        LiPo cells have an S-shaped discharge curve:
        - 100% → 90% SOC: ~4.0-4.2V per cell (flat)
        - 90% → 20% SOC: ~3.8-3.0V per cell (slope)
        - 20% → 0% SOC: ~3.0-2.8V per cell (steep drop)
        """
        # SOC points (0 to 1)
        soc_points = np.array([0.0, 0.2, 0.5, 0.8, 0.9, 1.0])
        
        # Voltage per cell for LiPo (multiply by num_cells later)
        v_per_cell = np.array([2.8, 3.0, 3.3, 3.7, 3.95, 4.2])
        
        # Scale to nominal voltage per cell
        v_nominal_per_cell = self.nominal_voltage / self.num_cells
        self.discharge_curve_soc = soc_points
        self.discharge_curve_v = v_per_cell / 4.2 * v_nominal_per_cell  # Normalize
    
    def _get_ocv(self) -> float:
        """
        Get open-circuit voltage as function of SOC (S-curve interpolation).
        
        Returns:
            V_ocv (V)
        """
        v_ocv = np.interp(self.soc, self.discharge_curve_soc, self.discharge_curve_v)
        return v_ocv * self.num_cells
    
    def _get_internal_resistance(self) -> float:
        """
        Get temperature-dependent internal resistance.
        
        R(T) = R_base * (1 + α*(T - T_ref))
        where α ≈ 0.005 /°C for LiPo
        
        Returns:
            R_internal (Ω)
        """
        alpha_temp = 0.005  # /°C
        t_ref = 25.0  # °C
        r_temp = self.r_internal_base * (1 + alpha_temp * (self.temperature - t_ref))
        return r_temp
    
    def get_terminal_voltage(self, current: float) -> float:
        """
        Get battery terminal voltage under load.
        
        V_terminal = V_ocv(SOC) - I * R_internal(T) - V_sag
        
        Args:
            current: discharge current (A)
        
        Returns:
            Terminal voltage (V)
        """
        v_ocv = self._get_ocv()
        r_internal = self._get_internal_resistance()
        v_drop = current * r_internal
        v_terminal = v_ocv - v_drop - self.v_sag
        return max(0.0, v_terminal)
    
    def get_max_current(self) -> float:
        """
        Get maximum allowed discharge current based on C-rating and SOC.
        
        I_max = (SOC * capacity_Ah) * C_rating
        
        Returns:
            I_max (A)
        """
        # Reduce max current as battery depletes
        i_max = self.soc * self.capacity_ah * self.c_rating
        return i_max
    
    def get_soc(self) -> float:
        """Get state of charge (0-1)."""
        return np.clip(self.soc, 0.0, 1.0)
    
    def get_energy_remaining(self) -> float:
        """Get remaining energy (Wh)."""
        return self.soc * self.capacity_ah * self.nominal_voltage
    
    def update(self, current: float, dt: float):
        """
        Update battery state (SOC, temperature, voltage sag).
        
        Args:
            current: discharge current (A)
            dt: timestep (seconds)
        """
        # Coulomb counting: dSOC/dt = -I / (capacity_Ah * 3600)
        # (3600 converts Ah to As)
        if self.capacity_ah > 0:
            dsoc_dt = -current / (self.capacity_ah * 3600.0)
            self.soc += dsoc_dt * dt
            self.soc = np.clip(self.soc, 0.0, 1.0)
        
        # Thermal model: dT/dt = (P_loss - P_convection) / (m * c_p)
        p_loss = current**2 * self._get_internal_resistance()  # Joule heating
        t_ambient = 25.0  # °C
        p_convection = self.h_convection * self.area_thermal * (self.temperature - t_ambient)
        
        if self.mass * self.c_thermal > 0:
            dt_dt = (p_loss - p_convection) / (self.mass * self.c_thermal)
            self.temperature += dt_dt * dt
            self.temperature = np.clip(self.temperature, -20, 80)  # Limit to reasonable range
        
        # Voltage sag dynamics (first-order lag of voltage drop)
        r_internal = self._get_internal_resistance()
        v_sag_target = current * r_internal * 0.1  # Transient sag ≈ 10% of steady-state drop
        dv_sag_dt = (v_sag_target - self.v_sag) / self.tau_sag
        self.v_sag += dv_sag_dt * dt
    
    def reset(self):
        """Reset battery to full charge."""
        self.soc = 1.0
        self.temperature = 25.0
        self.v_sag = 0.0
