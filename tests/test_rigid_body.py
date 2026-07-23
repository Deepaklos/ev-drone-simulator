"""
Unit tests for rigid body dynamics.

Tests 6-DOF dynamics, quaternion operations, and force/moment application.
"""

import numpy as np
import pytest
from physics.rigid_body import RigidBody, Quaternion


class TestQuaternion:
    """
    Test quaternion operations.
    """
    
    def test_quaternion_normalize(self):
        """Test quaternion normalization."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_norm = Quaternion.normalize(q)
        
        # Norm should be 1
        norm = np.linalg.norm(q_norm)
        assert abs(norm - 1.0) < 1e-6
    
    def test_quaternion_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_conj = Quaternion.conjugate(q)
        
        expected = np.array([1.0, -2.0, -3.0, -4.0])
        assert np.allclose(q_conj, expected)
    
    def test_identity_rotation(self):
        """Test identity quaternion (no rotation)."""
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        R = Quaternion.to_rotation_matrix(q_identity)
        
        # Should be identity matrix
        expected = np.eye(3)
        assert np.allclose(R, expected, atol=1e-6)
    
    def test_rotation_matrix_orthogonal(self):
        """Test rotation matrix is orthogonal (R*R^T = I)."""
        q = np.array([0.707, 0.707, 0.0, 0.0])  # 90° rotation around X
        R = Quaternion.to_rotation_matrix(q)
        
        # Check orthogonality: R @ R^T = I
        should_be_identity = R @ R.T
        assert np.allclose(should_be_identity, np.eye(3), atol=1e-6)


class TestRigidBodyBasic:
    """
    Test basic rigid body properties.
    """
    
    def test_rigid_body_initialization(self):
        """Test rigid body is initialized correctly."""
        mass = 1.0
        inertia = np.array([0.018, 0.018, 0.032])
        
        rb = RigidBody(mass=mass, inertia=inertia)
        
        assert rb.mass == mass
        assert rb.get_position()[2] == 0.0  # On ground
        assert np.allclose(rb.get_velocity(), [0, 0, 0])
        assert np.allclose(rb.get_quaternion(), [1, 0, 0, 0])  # Identity
        assert np.allclose(rb.get_angular_velocity(), [0, 0, 0])
    
    def test_state_setters_and_getters(self):
        """Test state getter/setter methods."""
        rb = RigidBody(mass=1.0, inertia=np.array([0.1, 0.1, 0.1]))
        
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.1, 0.2, 0.3])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.01, 0.02, 0.03])
        
        rb.set_position(pos)
        rb.set_velocity(vel)
        rb.set_quaternion(q)
        rb.set_angular_velocity(omega)
        
        assert np.allclose(rb.get_position(), pos)
        assert np.allclose(rb.get_velocity(), vel)
        assert np.allclose(rb.get_quaternion(), [1, 0, 0, 0])
        assert np.allclose(rb.get_angular_velocity(), omega)


class TestRigidBodyForces:
    """
    Test force application and acceleration.
    """
    
    def test_gravity_acceleration(self):
        """Test gravity produces downward acceleration."""
        rb = RigidBody(mass=1.0, inertia=np.array([0.1, 0.1, 0.1]))
        
        # Apply zero external force
        force = np.array([0.0, 0.0, 0.0])
        accel = rb.apply_force(force)
        
        # Should have downward acceleration due to gravity
        expected = np.array([0.0, 0.0, -9.81])
        assert np.allclose(accel, expected, atol=1e-6)
    
    def test_upward_force_counteracts_gravity(self):
        """Test upward force reduces downward acceleration."""
        rb = RigidBody(mass=1.0, inertia=np.array([0.1, 0.1, 0.1]))
        
        # Apply upward force equal to weight
        force = np.array([0.0, 0.0, 9.81])  # 1kg * 9.81 m/s^2
        accel = rb.apply_force(force)
        
        # Should have zero acceleration (hover)
        assert np.allclose(accel, [0, 0, 0], atol=1e-6)


class TestRigidBodyIntegration:
    """
    Integration tests for rigid body dynamics.
    """
    
    def test_free_fall(self):
        """Test free fall motion."""
        rb = RigidBody(mass=1.0, inertia=np.array([0.1, 0.1, 0.1]))
        rb.set_position(np.array([0.0, 0.0, 100.0]))  # 100m high
        
        force = np.array([0.0, 0.0, 0.0])
        moment = np.array([0.0, 0.0, 0.0])
        
        # Simulate for 1 second
        for _ in range(1000):
            state_dot = rb.get_state_derivative(force, moment)
            rb.update_state(state_dot, dt=0.001)
        
        pos = rb.get_position()
        vel = rb.get_velocity()
        
        # Should have fallen and gained downward velocity
        assert pos[2] < 100.0
        assert vel[2] < 0.0  # Moving downward
        
        # Check approximately correct (free fall: v = gt)
        expected_vel = 9.81 * 1.0
        assert 9.0 < abs(vel[2]) < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
