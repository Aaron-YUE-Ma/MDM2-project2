import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
def energy(xs, ps):
    for i in range(len(xs)):
        T = 1/2 * 1 * ps[i]**2
        U = - 1 * 9.81 * 1 * np.cos(xs[i])
        H = T + U
        HS.append(H)
    return HS

def compute_new_v(old_v, h, g, L, old_q):
    return old_v - h * g / L * np.sin(old_q)


def compute_new_q(old_q, h, new_v):
    return old_q + h * new_v


h = 0.01
g = 9.81
L = 1
q0 = np.pi / 3
v0 = 0
t = 0

ts = []
qs = [q0]
vs = [v0]
HS = []

old_v = v0
old_q = q0

for i in range(215):
    new_v = compute_new_v(old_v, h, g, L, old_q)  # Use function name correctly
    vs.append(new_v)

    new_q = compute_new_q(old_q, h, new_v)  # Use function name correctly
    qs.append(new_q)

    old_v = new_v
    old_q = new_q

# Define the x-axis points

HS = energy(qs, vs)
# Plot the results



plt.style.use('dark_background')
plt.scatter(qs, vs, marker='x', s = 1)
plt.xticks(fontsize=12)  # Change x-axis tick size
plt.yticks(fontsize=12)  # Change y-axis tick size
plt.xlabel('Position (Rad)', fontsize=15)
plt.ylabel('Velocity (Rad/s)', fontsize=15)
plt.tight_layout()

plt.savefig('positon, velocity.png')

plt.show()

for j in range(len(HS)):
    ts.append(t)
    t+=0.01

plt.style.use('dark_background')
plt.scatter(ts, HS, marker='x', s = 1, c='#81B1D2')
plt.xticks(np.arange(0, 2.5, 0.5), fontsize=15)  # Change x-axis tick size
plt.yticks(fontsize=15)  # Change y-axis tick size
plt.xlabel('Time (seconds)', fontsize=15)
plt.ylabel('Energy (Joules)', fontsize=15)
plt.tight_layout()


plt.savefig('Symplectic.png')

plt.show()

sorted_indices = np.argsort(ts)
ts_sorted = np.array(ts)[sorted_indices]
HS_sorted = np.array(HS)[sorted_indices]

# Fit a cubic spline
spline = CubicSpline(ts_sorted, HS_sorted, bc_type='clamped')

# Generate smooth curve points
ts_fine = np.linspace(min(ts_sorted), max(ts_sorted), 500)
HS_fine = spline(ts_fine)

plt.plot(ts_fine, HS_fine)
plt.xlabel('Time (seconds)', fontsize=15)
plt.ylabel('Energy (Joules)', fontsize=15)
plt.tight_layout()
plt.show()