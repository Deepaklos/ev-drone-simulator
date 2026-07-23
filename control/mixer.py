"""
Motor Mixer - Converts desired moments to individual motor throttles

For quadcopter X configuration:
  Motor 0 (front-right):  +roll, +pitch, +yaw
  Motor 1 (front-left):   -roll, +pitch, -yaw
  Motor 2 (rear-left):    -roll, -pitch, +yaw
  Motor 3 (rear-right):   +roll, -pitch, -yaw

Mixing matrix: throttles = M^-1 @ [thrust, roll_moment, pitch_moment, yaw_moment]
"""

import numpy as np
from typing import List


class Mixer:
    """
    Motor mixer for multirotor drones.
    """
    
    def __init__(self, drone_type: str = "quad_x", num_rotors: int = 4):
        """
        Initialize mixer for a given drone type.
        
        Args:
            drone_type: "quad_x", "quad_plus", "hex", "octo", etc.
            num_rotors: number of rotors
        """
        self.drone_type = drone_type
        self.num_rotors = num_rotors
        self.mixer_matrix = self._get_mixer_matrix()
        self.mixer_matrix_inv = np.linalg.pinv(self.mixer_matrix)
    
    def _get_mixer_matrix(self) -> np.ndarray:
        """
        Get mixer matrix for the drone configuration.
        
        Mixer matrix M maps motor commands to net forces/moments:
        [thrust, roll, pitch, yaw]^T = M @ [throttle0, throttle1, ...]
        
        Returns:
            Mixer matrix (4 x num_rotors)
        """
        if self.drone_type == "quad_x":
            # Quadcopter X configuration
            # Each rotor contributes thrust and moments
            M = np.array([
                [1, 1, 1, 1],      # Collective thrust
                [1, -1, -1, 1],    # Roll  (+ for M0,M3, - for M1,M2)
                [1, 1, -1, -1],    # Pitch (+ for M0,M1, - for M2,M3)
                [-1, 1, -1, 1]     # Yaw   (alternating for reaction torque)
            ])
            return M / 4.0  # Normalize
        
        elif self.drone_type == "quad_plus":
            # Quadcopter Plus configuration
            M = np.array([
                [1, 1, 1, 1],
                [0, 1, 0, -1],     # Roll
                [1, 0, -1, 0],     # Pitch
                [-1, 1, -1, 1]     # Yaw
            ])
            return M / 4.0
        
        elif self.drone_type == "hex":
            # Hexacopter configuration (6 rotors)
            angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            M = np.zeros((4, 6))
            M[0, :] = 1  # Thrust
            M[1, :] = np.cos(angles)  # Roll
            M[2, :] = np.sin(angles)  # Pitch
            M[3, :] = [-1 if i % 2 == 0 else 1 for i in range(6)]  # Yaw
            return M / 6.0
        
        else:
            # Default: assume symmetric quadcopter
            M = np.eye(4)
            return M
    
    def mix(self, thrust: float, roll_moment: float, pitch_moment: float, yaw_moment: float) -> np.ndarray:
        """
        Mix desired forces/moments to motor throttles.
        
        Args:
            thrust: desired total thrust (0-1)
            roll_moment: desired roll moment (-1 to 1)
            pitch_moment: desired pitch moment (-1 to 1)
            yaw_moment: desired yaw moment (-1 to 1)
        
        Returns:
            Array of motor throttles [0-1]
        """
        desired = np.array([thrust, roll_moment, pitch_moment, yaw_moment])
        throttles = self.mixer_matrix_inv @ desired
        
        # Normalize if any throttle exceeds 1.0
        throttles = np.clip(throttles, 0.0, 1.0)
        return throttles
