# EV Drone Flight Simulator

A **parametric, real-time, configurable electric drone flight simulator** built in pure Python. Design your drone frame, choose batteries and motors, load payloads, and simulate performance—all in interactive software.

## Key Features

### 🎯 **Parametric Design**
- Define drone geometry: # rotors, arm lengths, placement (quad-X, hex, octo, custom)
- Specify mass, inertia, center of gravity
- Load payloads with position offset
- **All configs saved as JSON** — design library for different drones

### 🔋 **Realistic Battery Model**
- Electrochemical discharge curve (voltage sag, not linear)
- C-rate limiting based on cell specs
- Internal resistance + thermal heating effects
- Real-time SOC (state of charge) via coulomb counting
- Battery voltage directly throttles available thrust — watch performance degrade as battery depletes

### ⚙️ **Physics-Based Rotor Model**
- Motor electrical dynamics (PWM → motor torque → RPM lag)
- Blade Element Momentum Theory (BEMT) thrust calculation
- RPM² scaling with air density (altitude + temperature effects)
- Realistic drag torque and gyroscopic effects

### 🎮 **6-DOF Rigid Body Dynamics**
- Position, velocity, orientation (quaternion), angular velocity
- RK4 fixed-step integration (physics decoupled from render)
- Aerodynamic drag, gravity, rotor thrust/moments
- Modular, readable Python code — inspect and modify physics directly

### 📊 **Real-Time Simulation & Visualization**
- Live 3D drone visualization (PyQt6 + pyqtgraph OpenGL)
- Telemetry dashboard: battery V/I/SOC, motor RPM, attitude, altitude, speed
- Drone design panel — modify parameters mid-simulation
- Start/pause/reset controls

### 🎛️ **Cascaded PID Control**
- Position → velocity → attitude → rate → motor mixing
- Configurable gains for different drone types
- Manual keyboard input (WASD + arrows)
- Auto-hover capability

---

## Why Build This?

Existing simulators (Gazebo, AirSim, jMAVSim) have known weaknesses:
- **Unrealistic battery models** — linear drain, no voltage sag
- **Oversimplified rotor models** — constant thrust coefficients
- **Hard-to-modify physics** — buried in game engines
- **Heavy dependencies** — slow iteration

This simulator fixes all four by providing **transparent, parametric, physics-first design**.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Deepaklos/ev-drone-simulator.git
cd ev-drone-simulator

# Install dependencies
pip install -r requirements.txt

# Run the simulator
python main.py
```

---

## Project Structure

```
ev_drone_sim/
├── main.py                     # GUI entry point
├── config.yaml                 # Global physics parameters
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── physics/
│   ├── __init__.py
│   ├── rigid_body.py           # 6-DOF nonlinear dynamics
│   ├── rotor.py                # Motor electrical model + BEMT thrust/torque
│   ├── battery.py              # Realistic battery: voltage sag, C-rate, thermal, SOC
│   ├── environment.py          # Air density, wind field, gravity
│   ├── integrator.py           # RK4 fixed-step integrator
│   └── drone.py                # Aggregates all physics for a drone
│
├── control/
│   ├── __init__.py
│   ├── mixer.py                # Motor mixing (quad-X, hex, octo, custom)
│   ├── pid.py                  # Cascaded PID controller
│   ├── attitude_controller.py  # Attitude control loops
│   └── position_controller.py  # Position/velocity control loops
│
├── models/
│   ├── __init__.py
│   ├── drone_config.py         # Drone specification class
│   ├── battery_spec.py         # Battery specification class
│   ├── motor_spec.py           # Motor specification class
│   └── simulation_state.py     # Simulation state object
│
├── gui/
│   ├── __init__.py
│   ├── main_window.py          # PyQt6 main window
│   ├── viewport_3d.py          # 3D drone visualization
│   ├── telemetry_panel.py      # Real-time plots and telemetry
│   ├── design_panel.py         # Drone design configuration UI
│   ├── control_panel.py        # Simulation controls
│   └── widgets/
│       ├── param_editor.py     # Parameter adjustment widgets
│       ├── plot_widget.py      # Plotting utility
│       └── mesh_3d.py          # Drone mesh rendering
│
├── utils/
│   ├── __init__.py
│   ├── constants.py            # Physical constants
│   ├── config_loader.py        # YAML config loading
│   ├── geometry.py             # Rotor placement helpers
│   └── logger.py               # Logging utilities
│
├── assets/
│   ├── default_drone.yaml      # Default quad drone config
│   ├── hex_drone.yaml          # Hex drone template
│   └── battery_library.yaml    # Common battery specs
│
└── tests/
    ├── __init__.py
    ├── test_battery.py         # Battery model verification
    ├── test_rotor.py           # Rotor model verification
    ├── test_rigid_body.py      # 6-DOF dynamics verification
    ├── test_mixer.py           # Motor mixing verification
    └── test_pid.py             # PID control verification
```

---

## Physics Models

### 1. **Rigid Body (6-DOF)**

State: position, velocity, orientation (quaternion), angular velocity

**Forces:**
- Gravity: `F_g = [0, 0, -m*g]`
- Rotor thrust: sum of 4+ thrusts along body Z-axis
- Aerodynamic drag: `F_drag = -0.5 * ρ * v * |v| * C_d * A`

**Moments:**
- Differential thrust (roll/pitch)
- Reaction torque (yaw)
- Gyroscopic effects from spinning rotors

**Integration:** RK4, 1ms timestep (decoupled from render)

---

### 2. **Rotor Model**

**Motor Dynamics:**
```
τ_motor = K_t * (PWM * V_battery - K_e * RPM)
dRPM/dt = (τ_motor - τ_drag) / I_rotor
```

**Thrust (BEMT-based):**
```
T = K_thrust * (ρ / ρ_sea_level) * RPM²
```

**Drag Torque:**
```
Q = K_drag * (ρ / ρ_sea_level) * RPM²
```

---

### 3. **Battery Model (Electrochemical)**

**State:**
- SOC via coulomb counting: `dSOC/dt = -I_total / Q_nominal`
- Temperature: `dT/dt = (I²R - h*A*(T - T_amb)) / (m*c_p)`

**Terminal Voltage:**
```
V = V_ocv(SOC) - I * R_internal(T) - V_sag_lag
```

**C-Rate Limiting:**
```
I_max = SOC * Q_nominal * C_rating
I = min(I_requested, I_max)
```

---

### 4. **Environment**

**Air Density (barometric):**
```
ρ(h, T) = ρ_sea * (T_ref / T) * exp(-M*g*h / (R*T_ref))
```

**Wind Field:**
```
v_wind = v_constant + v_gust(t)
```

---

## Control Architecture

```
User Input
  ↓ (position setpoint)
Position Controller
  ↓ (velocity setpoint)
Velocity Controller
  ↓ (attitude setpoint)
Attitude Controller
  ↓ (angular rate setpoint)
Rate Controller
  ↓ (motor commands)
Mixer
  ↓ (PWM per motor)
Rotors
```

---

## Configuration Example

```yaml
# assets/default_drone.yaml
drone:
  name: "DJI Phantom 4 Clone"
  type: "quadcopter"
  mass_empty: 1.3  # kg
  inertia: [0.018, 0.018, 0.032]  # kg*m²
  rotors:
    - id: 0
      position: [0.215, 0.215, 0]  # meters
      direction: 1  # +1 or -1 for CW/CCW
    - id: 1
      position: [-0.215, 0.215, 0]
      direction: -1
    - id: 2
      position: [-0.215, -0.215, 0]
      direction: 1
    - id: 3
      position: [0.215, -0.215, 0]
      direction: -1

battery:
  name: "5S LiPo 5000mAh"
  voltage_nominal: 18.5  # V
  capacity_mah: 5000
  internal_resistance: 0.012  # Ohms
  c_rating: 50  # C
  discharge_curve: "lipo_5s.csv"

motor:
  kv: 920  # RPM/V
  max_rpm: 10000
  k_thrust: 9.65e-5  # T / (ρ * RPM²)
  k_drag: 1.2e-6  # N*m / (ρ * RPM²)
  i_rotor: 0.00001  # kg*m²

control:
  position_pid: [1.0, 0.1, 0.5]  # [Kp, Ki, Kd]
  velocity_pid: [1.5, 0.2, 0.3]
  attitude_pid: [4.5, 0.05, 0.2]
  rate_pid: [0.15, 0.05, 0.004]
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Specific test
pytest tests/test_battery.py -v

# With coverage
pytest tests/ --cov=physics --cov=control
```

---

## Future Enhancements

- [ ] MAVLink bridge (PX4/ArduPilot SITL)
- [ ] Multi-body dynamics
- [ ] Blade flutter modes
- [ ] Camera gimbal
- [ ] Waypoint missions
- [ ] Hardware-in-the-loop via UDP
- [ ] ROS integration

---

## License

MIT
