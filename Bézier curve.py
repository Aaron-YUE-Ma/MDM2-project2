import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the pendulum
length = 1.0    # Length of the pendulum (m)
g = 9.81        # Gravitational acceleration (m/s^2)

def lagrangian(theta, omega):
    """Compute the Lagrangian of the pendulum."""
    kinetic = 0.5 * (length ** 2) * omega ** 2
    potential = g * length * (1 - np.cos(theta))
    return kinetic - potential

def euler_lagrange_step(theta, omega, dt):
    """Traditional discrete Lagrangian method."""
    alpha = -(g / length) * np.sin(theta)
    omega_new = omega + alpha * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new

def bezier_lagrange_step(theta, omega, dt):
    """Bézier curve-based variational integrator."""
    alpha = -(g / length) * np.sin(theta)
    
    # Define control points for the Bézier curve
    theta_mid = theta + 0.5 * omega * dt  # Midpoint estimate
    omega_mid = omega + 0.5 * alpha * dt  # Midpoint velocity estimate
    
    # Bezier cubic update (approximated)
    theta_new = (1 - dt) ** 3 * theta + 3 * (1 - dt) ** 2 * dt * theta_mid + 3 * (1 - dt) * dt ** 2 * theta_mid + dt ** 3 * (theta + omega * dt)
    omega_new = omega + alpha * dt
    return theta_new, omega_new

# Simulation parameters
dt = 0.01  # Time step
time = np.arange(0, 10, dt)

# Initial conditions
theta_0 = np.pi / 4  # 45 degrees
omega_0 = 0.0  # Initial angular velocity

# Arrays to store results
theta_euler, omega_euler = [theta_0], [omega_0]
theta_bezier, omega_bezier = [theta_0], [omega_0]
energy_euler, energy_bezier = [], []

# Run the simulation
for t in time:
    # Euler-Lagrange method
    theta_new, omega_new = euler_lagrange_step(theta_euler[-1], omega_euler[-1], dt)
    theta_euler.append(theta_new)
    omega_euler.append(omega_new)
    energy_euler.append(lagrangian(theta_new, omega_new))
    
    # Bézier method
    theta_new, omega_new = bezier_lagrange_step(theta_bezier[-1], omega_bezier[-1], dt)
    theta_bezier.append(theta_new)
    omega_bezier.append(omega_new)
    energy_bezier.append(lagrangian(theta_new, omega_new))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time, energy_euler, label='Euler-Lagrange Energy', linestyle='dashed')
plt.plot(time, energy_bezier, label='Bézier Lagrange Energy', linestyle='solid')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.legend()
plt.title('Energy Conservation Comparison')
plt.show()
