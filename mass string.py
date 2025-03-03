import numpy as np
import matplotlib.pyplot as plt

# System parameters
m = 1.0          # Mass
k = 1.0          # Spring constant
dt = 0.01        # Time step
total_time = 20.0 # Total simulation time
steps = int(total_time / dt)
x0 = 1.0         # Initial displacement
v0 = 0.0         # Initial velocity

# Initialize Euler integrator
x_euler, v_euler = x0, v0
H_euler = [0.5*m*v0**2 + 0.5*k*x0**2]
L_euler = [0.5*m*v0**2 - 0.5*k*x0**2]

# Euler integration loop
for _ in range(steps):
    F = -k * x_euler
    a = F / m
    x_new = x_euler + v_euler * dt
    v_new = v_euler + a * dt
    
    # Compute Hamiltonian and Lagrangian
    K = 0.5 * m * v_new**2
    V = 0.5 * k * x_new**2
    H_euler.append(K + V)
    L_euler.append(K - V)
    
    x_euler, v_euler = x_new, v_new

# Initialize Symplectic integrator
x_sym, v_sym = x0, v0
H_sym = [0.5*m*v0**2 + 0.5*k*x0**2]
L_sym = [0.5*m*v0**2 - 0.5*k*x0**2]

# Symplectic Euler integration loop
for _ in range(steps):
    F = -k * x_sym
    a = F / m
    v_new = v_sym + a * dt      # Update velocity first
    x_new = x_sym + v_new * dt  # Then update position
    
    # Compute Hamiltonian and Lagrangian
    K = 0.5 * m * v_new**2
    V = 0.5 * k * x_new**2
    H_sym.append(K + V)
    L_sym.append(K - V)
    
    x_sym, v_sym = x_new, v_new

# Generate time array
time = np.linspace(0, total_time, steps + 1)

# Plot Hamiltonian comparison
plt.figure(figsize=(10, 5))
plt.plot(time, H_euler, label='Explicit Euler', alpha=0.8)
plt.plot(time, H_sym, label='Symplectic Euler', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.title('Hamiltonian Conservation Comparison')
plt.legend()
plt.grid(True)
plt.savefig('hamiltonian_comparison.png')
plt.show()

# Plot Lagrangian comparison
plt.figure(figsize=(10, 5))
plt.plot(time, L_euler, label='Explicit Euler', alpha=0.8)
plt.plot(time, L_sym, label='Symplectic Euler', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Lagrangian (K - V)')
plt.title('Lagrangian Behavior Comparison')
plt.legend()
plt.grid(True)
plt.savefig('lagrangian_comparison.png')
plt.show()