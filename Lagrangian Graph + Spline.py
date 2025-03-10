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
def update_x(pk, xk):
    xk_k = xk - ((0.04905 * np.sin(xk)) - pk)/(100 + 0.024525 * (np.cos(xk)))
    return xk_k


def update_p(pk, xk, xk_k):
    pk_k = 100 * (xk_k - xk) - (0.04905 * (np.sin((xk + xk_k)/2)))
    return pk_k


# Fit curve to Energy-Time data

xk0 = np.pi/3
pk0 = 0


xk = xk0
pk = pk0
t = 0

ts = []
xs = [xk0]
ps = [pk0]
HS = []

for i in range(215):
    xk_new = update_x(pk, xk)
    pk_new = update_p(pk, xk, xk_new)
    xs.append(xk_new)
    ps.append(pk_new)
    xk = xk_new
    pk = pk_new

HS = energy(xs, ps)

plt.style.use('dark_background')
plt.scatter(xs, ps, marker='x', s = 1)
plt.xticks(fontsize=12)  # Change x-axis tick size
plt.yticks(fontsize=12)  # Change y-axis tick size
plt.xlabel('Position (Rad)', fontsize=15)
plt.ylabel('Velocity (Rad/s)', fontsize=15)
plt.tight_layout()

plt.savefig('positon, velocity_L.png')

plt.show()

for j in range(len(HS)):
    ts.append(t)
    t+=0.01



plt.style.use('dark_background')
plt.scatter(ts, HS, marker='x', s = 1, c='#81B1D2')
#plt.xticks(np.arange(0, 2.5, 0.5), fontsize=12)  # Change x-axis tick size
#plt.yticks(fontsize=12)  # Change y-axis tick size
plt.xlabel('Time (seconds)', fontsize=15)
plt.ylabel('Energy (Joules)', fontsize=15)
plt.tight_layout()

plt.savefig('Lagrangian.png')

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