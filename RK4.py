import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, CubicSpline

plt.style.use('dark_background')

# System parameters
m = 1.0
L = 1.0
g = 9.81
dt = 0.001
total_time = 2.5  # Simulation duration
n_steps = int(total_time / dt)
time = np.linspace(0, total_time, n_steps+1)

# Initial conditions
theta0 = np.pi/3  # Initial angle
p0 = 0.0           # Initial momentum

# Hamiltonian function
def hamiltonian(theta, p):
    return (p**2)/(2*m*L**2) - m*g*L*np.cos(theta)

# Derivatives for equations of motion
def derivatives(theta, p):
    dtheta_dt = p / (m * L**2)
    dp_dt = -m * g * L * np.sin(theta)
    return dtheta_dt, dp_dt

def rk4_step(theta, p, dt):
    k1_theta, k1_p = derivatives(theta, p)
    k2_theta, k2_p = derivatives(theta + dt/2*k1_theta, p + dt/2*k1_p)
    k3_theta, k3_p = derivatives(theta + dt/2*k2_theta, p + dt/2*k2_p)
    k4_theta, k4_p = derivatives(theta + dt*k3_theta, p + dt*k3_p)
    
    theta_new = theta + dt/6*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    p_new = p + dt/6*(k1_p + 2*k2_p + 2*k3_p + k4_p)
    return theta_new, p_new

# Traditional Euler method
def euler_step(theta, p, dt):
    dtheta, dp = derivatives(theta, p)
    return theta + dt*dtheta, p + dt*dp

# Symplectic Integrator (semi-implicit Euler)
def symplectic_step(theta, p, dt):
    p_new = p - m * g * L * np.sin(theta) * dt
    theta_new = theta + p_new / (m * L**2) * dt
    return theta_new, p_new

# Lagrangian update

def update_x(pk, xk):
    return xk - ((0.04905 * np.sin(xk)) - pk)/(100 + 0.024525 * np.cos(xk))

def update_p(pk, xk, xk_k):
    return 100 * (xk_k - xk) - (0.04905 * np.sin((xk + xk_k)/2))

# Initialize arrays
theta_euler, p_euler = np.zeros(n_steps+1), np.zeros(n_steps+1)
theta_sympl, p_sympl = np.zeros(n_steps+1), np.zeros(n_steps+1)
theta_rk4, p_rk4 = np.zeros(n_steps+1), np.zeros(n_steps+1) 
energy_euler = np.zeros(n_steps+1)
energy_sympl = np.zeros(n_steps+1)
energy_rk4 = np.zeros(n_steps+1)

# Lagrangian method initialization
xk0 = theta0
pk0 = p0
xk = xk0
pk = pk0
lagrangian_xs = [xk0]
lagrangian_ps = [pk0]
lagrangian_energy = []

# Set initial conditions
theta_euler[0] = theta_sympl[0] = theta_rk4[0] = theta0
p_euler[0] = p_sympl[0] = p_rk4[0] = p0
energy_euler[0] = energy_sympl[0] = energy_rk4[0] = hamiltonian(theta0, p0)

# Integration loop
for i in range(n_steps):
    # Euler method
    theta_euler[i+1], p_euler[i+1] = euler_step(theta_euler[i], p_euler[i], dt)
    energy_euler[i+1] = hamiltonian(theta_euler[i+1], p_euler[i+1])
    
    # Symplectic method
    theta_sympl[i+1], p_sympl[i+1] = symplectic_step(theta_sympl[i], p_sympl[i], dt)
    energy_sympl[i+1] = hamiltonian(theta_sympl[i+1], p_sympl[i+1])
    
    # RK4
    theta_rk4[i+1], p_rk4[i+1] = rk4_step(theta_rk4[i], p_rk4[i], dt)
    energy_rk4[i+1] = hamiltonian(theta_rk4[i+1], p_rk4[i+1])
    
    # Lagrangian update
    xk_new = update_x(pk, xk)
    pk_new = update_p(pk, xk, xk_new)
    lagrangian_xs.append(xk_new)
    lagrangian_ps.append(pk_new)
    xk, pk = xk_new, pk_new
    T = 0.5 * pk**2
    U = -m * g * L * np.cos(xk)
    lagrangian_energy.append(T + U)

# Fit a cubic spline to Lagrangian energy
ts_fine = np.linspace(0, total_time, 500)
spline_lagrangian_energy = CubicSpline(np.linspace(0, total_time, len(lagrangian_energy)), lagrangian_energy, bc_type='clamped')
lagrangian_energy_fine = spline_lagrangian_energy(ts_fine)

# Plot Energy Conservation
plt.figure(figsize=(7, 5))


plt.plot(time, energy_rk4,  label='RK4 Method', color='#FA8174') 

plt.xlabel('Time (s)')
plt.ylabel('Total Energy (J)')
plt.title('Energy Conservation')
plt.legend()
plt.show()

# Plot Phase Space Trajectory
plt.figure(figsize=(7, 5))

plt.plot(theta_rk4, p_rk4, label='RK4 Method', color='#FA8174') 

plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$p$ (kg·m²/s)')
plt.title('Phase Space Trajectory')
plt.legend()
plt.show()
