"""
Drone - Aggregates all physics components

Combines:
  - Rigid body (6-DOF dynamics)
  - Rotors (4+ motors with BEMT thrust model)
  - Battery (electrochemical discharge, C-rate limiting)
  - Environment (air density, wind, gravity)
  - Integrator (RK4 physics stepping)
"""

import numpy as np
from typing import List, Dict
from physics.rigid_body import RigidBody
from physics.rotor import Rotor
from physics.battery import Battery
from physics.environment import Environment
from physics.integrator import RK4Integrator


class Drone:
    """
    Complete drone model: body + rotors + battery + environment.
    """
    
    def __init__(
        self,
        mass: float,
        inertia: np.ndarray,
        rotor_configs: List[Dict],
        battery: Battery,
        environment: Environment,
        dt: float = 0.001
    ):
        """
        Initialize drone.
        
        Args:
            mass: drone mass excluding battery (kg)
            inertia: [Ixx, Iyy, Izz] diagonal moments of inertia (kg*m²)
            rotor_configs: list of rotor configs, each with:
              {"position": [x, y, z], "direction": ±1, "kv": ..., "k_thrust": ..., "k_drag": ...}
            battery: Battery instance
            environment: Environment instance
            dt: physics timestep (seconds)
        """
        # Rigid body
        total_mass = mass + battery.mass  # Include battery mass
        self.body = RigidBody(total_mass, inertia)
        
        # Rotors
        self.rotors: List[Rotor] = []
        self.rotor_positions: List[np.ndarray] = []
        for cfg in rotor_configs:
            rotor = Rotor(
                kv=cfg.get("kv", 920),
                k_thrust=cfg.get("k_thrust", 9.65e-5),
                k_drag=cfg.get("k_drag", 1.2e-6),
                i_rotor=cfg.get("i_rotor", 1e-5),
                direction=cfg.get("direction", 1),
                r_internal=cfg.get("r_internal", 0.01)
            )
            self.rotors.append(rotor)
            self.rotor_positions.append(np.array(cfg["position"]))
        
        self.battery = battery
        self.environment = environment
        self.integrator = RK4Integrator(dt)
        self.dt = dt
        
        # Total current draw (sum of all rotors)
        self.total_current = 0.0
    
    def set_motor_throttles(self, throttles: np.ndarray):
        """
        Set throttle for each motor.
        
        Args:
            throttles: array of throttles [0-1] for each rotor
        """
        for i, throttle in enumerate(throttles):
            self.rotors[i].set_throttle(throttle)
    
    def compute_forces_moments(self) -> tuple:
        """
        Compute total force and moment from rotor thrusts and battery state.
        
        Returns:
            (force_world, moment_body) where:
              - force_world: [Fx, Fy, Fz] in world frame (N)
              - moment_body: [τx, τy, τz] in body frame (N*m)
        """
        rho = self.environment.get_air_density(self.body.get_position()[2])
        
        # Compute thrust and torque from each rotor
        thrusts = []
        for rotor in self.rotors:
            t = rotor.get_thrust(rho)
            thrusts.append(t)
        
        # Total thrust in body frame: [0, 0, sum(T_i)]
        total_thrust_body = np.array([0, 0, sum(thrusts)])
        
        # Convert to world frame using rotation matrix
        R = self.body.get_rotation_matrix()
        force_world = R @ total_thrust_body
        
        # Add wind disturbance
        wind = self.environment.get_wind(self.body.get_position()[2])
        v_world = self.body.get_velocity()
        v_rel = v_world - wind  # Velocity relative to air
        
        # Aerodynamic drag (quadratic): F_drag = -0.5 * ρ * v² * C_d * A
        c_d = 0.1  # Drag coefficient
        a_ref = 0.1  # Reference area (m²)
        if np.linalg.norm(v_rel) > 0.1:
            f_drag = -0.5 * rho * np.linalg.norm(v_rel) * v_rel * c_d * a_ref
        else:
            f_drag = np.zeros(3)
        
        force_world += f_drag
        
        # Moments from differential thrust and rotor drag
        moment_body = np.zeros(3)
        
        for i, (rotor, pos) in enumerate(zip(self.rotors, self.rotor_positions)):
            # Roll/pitch from thrust imbalance
            thrust_i = thrusts[i]
            # Moment arm: r × F = pos × [0, 0, thrust]
            moment_rp = np.cross(pos, np.array([0, 0, thrust_i]))
            moment_body[0] += moment_rp[0]  # Roll
            moment_body[1] += moment_rp[1]  # Pitch
            
            # Yaw from rotor drag torque
            drag_torque = rotor.get_drag_torque(rho)
            moment_body[2] += drag_torque * rotor.direction
        
        # Gyroscopic effects (simplified: ignore for now)
        # Full gyroscopic: I_rotor * omega_rotor × omega_body
        
        return force_world, moment_body
    
    def step(self):
        """
        Perform one physics integration step.
        
        1. Update rotor RPMs based on throttle and battery voltage
        2. Compute forces/moments from rotors
        3. Integrate rigid-body dynamics (RK4)
        4. Update battery state from current draw
        5. Update environment
        """
        # Get current battery voltage
        v_battery = self.battery.get_terminal_voltage(self.total_current)
        rho = self.environment.get_air_density(self.body.get_position()[2])
        
        # Update rotor RPMs
        for rotor in self.rotors:
            rotor.update(v_battery, rho, self.dt)
        
        # Compute current draw
        self.total_current = sum(rotor.get_current(v_battery, rho) for rotor in self.rotors)
        
        # Limit current by C-rate
        i_max = self.battery.get_max_current()
        if self.total_current > i_max:
            # Proportionally reduce all rotor thrusts
            throttle_scale = i_max / self.total_current
            for rotor in self.rotors:
                rotor.set_throttle(rotor.throttle * throttle_scale)
            self.total_current = i_max
        
        # Compute forces and moments
        def state_derivative(state, force_w, moment_b):
            self.body.state = state
            return self.body.get_state_derivative(force_w, moment_b)
        
        force_world, moment_body = self.compute_forces_moments()
        
        # Integrate with RK4
        self.body.state = self.integrator.step(
            self.body.state,
            state_derivative,
            force_world,
            moment_body
        )
        
        # Update battery
        self.battery.update(self.total_current, self.dt)
        
        # Update environment
        self.environment.update(self.dt)
    
    def reset(self):
        """Reset drone to initial state."""
        self.body.set_position(np.array([0, 0, 0]))
        self.body.set_velocity(np.array([0, 0, 0]))
        self.body.set_quaternion(np.array([1, 0, 0, 0]))
        self.body.set_angular_velocity(np.array([0, 0, 0]))
        
        for rotor in self.rotors:
            rotor.reset()
        
        self.battery.reset()
        self.total_current = 0.0
