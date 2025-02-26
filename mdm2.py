import numpy as np
import matplotlib.pyplot as plt

# System parameters
m = 1.0
L = 1.0
g = 9.81
dt = 0.01
total_time = 20.0  # Simulation duration
n_steps = int(total_time / dt)
time = np.linspace(0, total_time, n_steps+1)

# Initial conditions
theta0 = np.pi/3  # Initial angle
p0 = 0.0           # Initial momentum

# Hamiltonian function
def hamiltonian(theta, p):
    return (p**2)/(2*m*L**2) + m*g*L*(1 - np.cos(theta))

# Derivatives for equations of motion
def derivatives(theta, p):
    dtheta_dt = p / (m * L**2)
    dp_dt = -m * g * L * np.sin(theta)
    return dtheta_dt, dp_dt

# Traditional Euler method
def euler_step(theta, p, dt):
    dtheta, dp = derivatives(theta, p)
    return theta + dt*dtheta, p + dt*dp

# Symplectic Integrator (semi-implicit Euler)
def symplectic_step(theta, p, dt):
    # Update momentum first using current position
    dp = -m * g * L * np.sin(theta) * dt
    p_new = p + dp
    # Update position using new momentum
    dtheta = p_new / (m * L**2) * dt
    theta_new = theta + dtheta
    return theta_new, p_new

# Initialize arrays
theta_euler, p_euler = np.zeros(n_steps+1), np.zeros(n_steps+1)
theta_sympl, p_sympl = np.zeros(n_steps+1), np.zeros(n_steps+1)
energy_euler, energy_sympl = np.zeros(n_steps+1), np.zeros(n_steps+1)

# Set initial conditions
theta_euler[0] = theta_sympl[0] = theta0
p_euler[0] = p_sympl[0] = p0
energy_euler[0] = energy_sympl[0] = hamiltonian(theta0, p0)

# Integration loop
for i in range(n_steps):
    # Euler method
    theta_euler[i+1], p_euler[i+1] = euler_step(theta_euler[i], p_euler[i], dt)
    energy_euler[i+1] = hamiltonian(theta_euler[i+1], p_euler[i+1])
    
    # Symplectic method
    theta_sympl[i+1], p_sympl[i+1] = symplectic_step(theta_sympl[i], p_sympl[i], dt)
    energy_sympl[i+1] = hamiltonian(theta_sympl[i+1], p_sympl[i+1])

# Plot results
plt.figure(figsize=(15, 10))

# Energy comparison
plt.subplot(2, 2, 1)
plt.plot(time, energy_euler, label='Euler Method')
plt.plot(time, energy_sympl, label='Symplectic Method')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (J)')
plt.title('Energy Conservation')
plt.legend()

# Phase space trajectory
plt.subplot(2, 2, 2)
plt.plot(theta_euler, p_euler, label='Euler Method')
plt.plot(theta_sympl, p_sympl, label='Symplectic Method')
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$p$ (kg·m²/s)')
plt.title('Phase Space Trajectory')
plt.legend()

# Angle evolution
plt.subplot(2, 2, 3)
plt.plot(time, theta_euler, label='Euler Method')
plt.plot(time, theta_sympl, label='Symplectic Method')
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta$ (rad)')
plt.title('Angular Displacement')
plt.legend()

# Energy difference
plt.subplot(2, 2, 4)
plt.plot(time, energy_euler - energy_euler[0], label='Euler Method')
plt.plot(time, energy_sympl - energy_sympl[0], label='Symplectic Method')
plt.xlabel('Time (s)')
plt.ylabel('Energy Drift (J)')
plt.title('Energy Deviation from Initial')
plt.legend()

plt.tight_layout()
plt.show()