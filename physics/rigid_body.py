"""
6-DOF Rigid Body Dynamics

Models a rigid body with 6 degrees of freedom:
  - Position: (x, y, z) in world frame
  - Velocity: (vx, vy, vz) in world frame
  - Orientation: quaternion (w, x, y, z) representing rotation from body to world
  - Angular velocity: (p, q, r) in body frame

State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]

Physics equations:
  - Force balance: F_total = m * a_world
  - Torque balance: τ_total = I * α_body (in body frame)
  - Quaternion kinematics: dq/dt = 0.5 * Ω * q
"""

import numpy as np
from typing import Tuple


class Quaternion:
    """
    Quaternion representation: q = (w, x, y, z) where w is scalar part.
    """
    
    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """
        Normalize a quaternion to unit length.
        
        Args:
            q: quaternion [w, x, y, z]
        
        Returns:
            Normalized quaternion
        """
        return q / np.linalg.norm(q)
    
    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """
        Return conjugate of quaternion: q* = (w, -x, -y, -z)
        
        Args:
            q: quaternion [w, x, y, z]
        
        Returns:
            Conjugate quaternion
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions: q_result = q1 * q2
        
        Uses the formula:
        (w1, v1) * (w2, v2) = (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
        
        Args:
            q1: quaternion [w, x, y, z]
            q2: quaternion [w, x, y, z]
        
        Returns:
            Result quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix (body to world).
        
        Args:
            q: quaternion [w, x, y, z]
        
        Returns:
            3x3 rotation matrix R such that v_world = R @ v_body
        """
        q = Quaternion.normalize(q)
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R
    
    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric matrix from 3D vector for cross product.
        
        Args:
            v: 3D vector [x, y, z]
        
        Returns:
            3x3 skew matrix such that [v]_x @ u = v × u
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])


class RigidBody:
    """
    6-DOF rigid body with mass and inertia.
    
    State: [position (3), velocity (3), quaternion (4), angular_velocity (3)]
    """
    
    def __init__(self, mass: float, inertia: np.ndarray, initial_position: np.ndarray = None):
        """
        Initialize rigid body.
        
        Args:
            mass: body mass (kg)
            inertia: diagonal principal moments of inertia [Ixx, Iyy, Izz] (kg*m²)
            initial_position: initial position in world frame [x, y, z] (m)
        """
        self.mass = mass
        self.I = np.diag(inertia)  # Principal inertia matrix
        self.I_inv = np.linalg.inv(self.I)
        
        # State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
        self.state = np.zeros(13)
        
        if initial_position is not None:
            self.state[0:3] = initial_position
        else:
            self.state[0:3] = [0, 0, 0]
        
        # Initialize at rest with identity quaternion (no rotation)
        self.state[6] = 1.0  # qw = 1
        
        self.g = 9.81  # m/s²
    
    def get_position(self) -> np.ndarray:
        """Get position in world frame."""
        return self.state[0:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get velocity in world frame."""
        return self.state[3:6].copy()
    
    def get_quaternion(self) -> np.ndarray:
        """Get orientation quaternion [w, x, y, z]."""
        return self.state[6:10].copy()
    
    def get_angular_velocity(self) -> np.ndarray:
        """Get angular velocity in body frame [p, q, r]."""
        return self.state[10:13].copy()
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from body to world frame."""
        q = self.get_quaternion()
        return Quaternion.to_rotation_matrix(q)
    
    def set_position(self, pos: np.ndarray):
        """Set position in world frame."""
        self.state[0:3] = pos
    
    def set_velocity(self, vel: np.ndarray):
        """Set velocity in world frame."""
        self.state[3:6] = vel
    
    def set_quaternion(self, q: np.ndarray):
        """Set orientation quaternion."""
        self.state[6:10] = Quaternion.normalize(q)
    
    def set_angular_velocity(self, omega: np.ndarray):
        """Set angular velocity in body frame."""
        self.state[10:13] = omega
    
    def apply_force(self, force_world: np.ndarray) -> np.ndarray:
        """
        Compute acceleration from applied force.
        
        F_total = m * a_world
        a_world = F_total / m
        
        Args:
            force_world: force in world frame [Fx, Fy, Fz] (N)
        
        Returns:
            Acceleration in world frame [ax, ay, az] (m/s²)
        """
        # Add gravity (acts downward in world frame)
        F_total = force_world + np.array([0, 0, -self.mass * self.g])
        return F_total / self.mass
    
    def apply_moment(self, moment_body: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Compute angular acceleration from applied moment (torque).
        
        τ_total = I * α + ω × (I * ω)  (Euler's equation in body frame)
        α = I^-1 * (τ_total - ω × (I * ω))
        
        Args:
            moment_body: torque in body frame [τx, τy, τz] (N*m)
            omega: angular velocity in body frame [p, q, r] (rad/s)
        
        Returns:
            Angular acceleration in body frame [αp, αq, αr] (rad/s²)
        """
        # Gyroscopic term: ω × (I * ω)
        I_omega = self.I @ omega
        gyro_term = np.cross(omega, I_omega)
        
        # Angular acceleration: α = I^-1 * (τ - gyro_term)
        alpha = self.I_inv @ (moment_body - gyro_term)
        return alpha
    
    def get_state_derivative(self, force_world: np.ndarray, moment_body: np.ndarray) -> np.ndarray:
        """
        Compute state derivative: ds/dt = [dx/dt, dv/dt, dq/dt, dω/dt]
        
        Args:
            force_world: total force in world frame [Fx, Fy, Fz] (N)
            moment_body: total torque in body frame [τx, τy, τz] (N*m)
        
        Returns:
            State derivative vector (13 elements)
        """
        state_dot = np.zeros(13)
        
        # Position derivative: dx/dt = v_world
        state_dot[0:3] = self.get_velocity()
        
        # Velocity derivative: dv/dt = a_world
        state_dot[3:6] = self.apply_force(force_world)
        
        # Quaternion derivative: dq/dt = 0.5 * Ω * q
        # where Ω is the skew matrix of angular velocity
        omega = self.get_angular_velocity()
        q = self.get_quaternion()
        
        # Quaternion kinematics (Euler's method for quaternion)
        # dq/dt = 0.5 * [−p*qx − q*qy − r*qz,
        #                 p*qw + r*qy − q*qz,
        #                 q*qw − r*qx + p*qz,
        #                 r*qw + q*qx − p*qy]
        p, q_b, r = omega
        qw, qx, qy, qz = q
        
        state_dot[6] = 0.5 * (-p*qx - q_b*qy - r*qz)
        state_dot[7] = 0.5 * (p*qw + r*qy - q_b*qz)
        state_dot[8] = 0.5 * (q_b*qw - r*qx + p*qz)
        state_dot[9] = 0.5 * (r*qw + q_b*qx - p*qy)
        
        # Angular velocity derivative: dω/dt = α_body
        state_dot[10:13] = self.apply_moment(moment_body, omega)
        
        return state_dot
    
    def update_state(self, state_dot: np.ndarray, dt: float):
        """
        Update state using Euler integration (called by integrator).
        
        Args:
            state_dot: state derivative
            dt: timestep (seconds)
        """
        self.state += state_dot * dt
        
        # Renormalize quaternion to maintain unit length
        q = self.get_quaternion()
        self.set_quaternion(q)
