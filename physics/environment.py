"""
Environment Model

Models:
  1. Air density as function of altitude and temperature (barometric formula)
  2. Wind field (constant + gust noise)
  3. Gravity
"""

import numpy as np


class Environment:
    """
    Represents the simulation environment.
    """
    
    def __init__(
        self,
        gravity: float = 9.81,
        rho_sea_level: float = 1.225,
        temperature_ref: float = 288.15,
        wind_speed: float = 0.0,
        wind_direction: float = 0.0
    ):
        """
        Initialize environment.
        
        Args:
            gravity: gravitational acceleration (m/s²)
            rho_sea_level: air density at sea level (kg/m³)
            temperature_ref: reference temperature (K, 288.15 = 15°C)
            wind_speed: constant wind speed (m/s)
            wind_direction: wind direction (degrees, 0 = +X axis)
        """
        self.g = gravity
        self.rho_sea_level = rho_sea_level
        self.temperature_ref = temperature_ref
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        
        # Barometric formula constants
        self.M = 0.029  # Molar mass of air (kg/mol)
        self.R = 8.314  # Gas constant (J/(mol*K))
        self.L = 0.0065  # Adiabatic lapse rate (K/m)
        
        # Gust model parameters
        self.gust_amplitude = 0.5  # m/s
        self.gust_frequency = 1.0  # Hz
        self.time = 0.0
    
    def get_air_density(self, altitude: float, temperature: float = None) -> float:
        """
        Compute air density using barometric formula.
        
        ρ(h, T) = ρ_0 * (T / T_0)^4.256 * exp(-1.577 * M * g * h / (R * T_0))
        
        Simplified version using temperature lapse rate:
        T(h) = T_0 - L*h
        ρ(h) = ρ_0 * (1 - L*h / T_0)^(g*M / (R*L))
        
        Args:
            altitude: altitude above sea level (m)
            temperature: temperature at altitude (K); if None, uses lapse rate
        
        Returns:
            Air density (kg/m³)
        """
        if temperature is None:
            # Use adiabatic lapse rate
            temperature = self.temperature_ref - self.L * altitude
        
        # Clamp to reasonable range
        temperature = np.clip(temperature, 200, 320)  # K
        altitude = np.clip(altitude, 0, 8000)  # m (up to ~26k ft)
        
        # Barometric formula
        exponent = -self.M * self.g * altitude / (self.R * self.temperature_ref)
        rho = self.rho_sea_level * (self.temperature_ref / temperature) * np.exp(exponent)
        return rho
    
    def get_wind(self, altitude: float = 0.0) -> np.ndarray:
        """
        Get wind velocity vector in world frame.
        
        Wind = constant_wind + gust
        
        Args:
            altitude: altitude (m) — wind may increase with altitude
        
        Returns:
            Wind velocity [vx, vy, vz] (m/s) in world frame
        """
        # Constant wind component
        wind_rad = np.deg2rad(self.wind_direction)
        wind_x = self.wind_speed * np.cos(wind_rad)
        wind_y = self.wind_speed * np.sin(wind_rad)
        
        # Gust component (sinusoidal for now)
        gust_x = self.gust_amplitude * np.sin(2 * np.pi * self.gust_frequency * self.time)
        gust_y = self.gust_amplitude * np.cos(2 * np.pi * self.gust_frequency * self.time)
        
        return np.array([wind_x + gust_x, wind_y + gust_y, 0.0])
    
    def update(self, dt: float):
        """
        Update environment state (time for gust model).
        
        Args:
            dt: timestep (seconds)
        """
        self.time += dt
