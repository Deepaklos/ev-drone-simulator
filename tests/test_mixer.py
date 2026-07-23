"""
Unit tests for motor mixer.

Tests motor mixing for different drone configurations.
"""

import numpy as np
import pytest
from control.mixer import Mixer


class TestMixerBasic:
    """
    Test basic mixer properties.
    """
    
    def test_mixer_initialization_quad_x(self):
        """Test mixer initialization for quad-X."""
        mixer = Mixer(drone_type="quad_x", num_rotors=4)
        
        assert mixer.drone_type == "quad_x"
        assert mixer.num_rotors == 4
        assert mixer.mixer_matrix.shape == (4, 4)
    
    def test_mixer_initialization_quad_plus(self):
        """Test mixer initialization for quad-plus."""
        mixer = Mixer(drone_type="quad_plus", num_rotors=4)
        
        assert mixer.drone_type == "quad_plus"
        assert mixer.mixer_matrix.shape == (4, 4)
    
    def test_mixer_initialization_hex(self):
        """Test mixer initialization for hexacopter."""
        mixer = Mixer(drone_type="hex", num_rotors=6)
        
        assert mixer.drone_type == "hex"
        assert mixer.mixer_matrix.shape == (4, 6)


class TestMixerQuadX:
    """
    Test quad-X mixer.
    """
    
    def test_hover_mix(self):
        """Test mixing for hover (equal thrust on all motors)."""
        mixer = Mixer(drone_type="quad_x", num_rotors=4)
        
        throttles = mixer.mix(
            thrust=0.5,
            roll_moment=0.0,
            pitch_moment=0.0,
            yaw_moment=0.0
        )
        
        # All throttles should be equal
        assert np.allclose(throttles, [0.5, 0.5, 0.5, 0.5], atol=0.01)
    
    def test_positive_roll(self):
        """Test mixing for positive roll (right side up)."""
        mixer = Mixer(drone_type="quad_x", num_rotors=4)
        
        throttles = mixer.mix(
            thrust=0.5,
            roll_moment=0.2,
            pitch_moment=0.0,
            yaw_moment=0.0
        )
        
        # For positive roll: right motors should go up, left down
        # Quad-X: M0 (FR) and M3 (RR) increase, M1 (FL) and M2 (RL) decrease
        assert throttles[0] > throttles[1]
        assert throttles[3] > throttles[2]
    
    def test_throttle_saturation(self):
        """Test mixer clamps throttles to [0, 1]."""
        mixer = Mixer(drone_type="quad_x", num_rotors=4)
        
        throttles = mixer.mix(
            thrust=0.9,
            roll_moment=1.0,  # Large request
            pitch_moment=0.0,
            yaw_moment=0.0
        )
        
        # All throttles should be in [0, 1]
        assert np.all(throttles >= 0.0)
        assert np.all(throttles <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
