from sympy import symbols, nsolve, sin, pi # solve
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from scipy.interpolate import CubicSpline

plt.style.use('dark_background')


# Initial conditions 
theta_0 = pi/3
theta_dot_0 = 0
t = 2.5 # total time
h = 0.01

M = 1
L = 1
g = 9.81


# Functions
def step(p_k, theta_old, h):
    '''
    step function
    '''
    M = 1 # mass
    L = 1 # length of rod
    g = 9.81 # gravity 

    # defines the first equations
    theta_new = symbols('theta_new')
    func = (((M * L**2)/h) * (theta_new - theta_old)) + ((M * g * L * h)/2 * sin((theta_old + theta_new) / 2)) - p_k
    
    # solves the first equation for theta_(k+1)
    initial_guess = round(theta_old, 2)  
    th_n = nsolve(func, theta_new, initial_guess) # solves the equation 'func' using sympy

    # solves the second equation for p_(k+1)
    p_new = (((M * L**2)/h) * (th_n - theta_old)) - ((M * g * L * h)/2 * sin((theta_old + th_n) / 2))

    return th_n, p_new


def solve(theta_0, P_0, t, h):
    '''
    initial conditions: theta_0 = pi/3, P_0 = 0
    t: total time
    h: step length
    '''
    theta_n = []
    theta_n.append(theta_0)
    P_n = []
    P_n.append(P_0)

    for k in range(math.trunc(t/h)):
        theta_k, P_k = step(P_n[k], theta_n[k], h)

        theta_n.append(theta_k)
        P_n.append(P_k)

    return theta_n, P_n


def spline_fitting_grad(t, c, d, plot=False):
    T = np.array([
        [t[0]**3, t[0]**2, t[0], 1],
        [t[1]**3, t[1]**2, t[1], 1],
        [(3 * t[0]**2), (2 * t[0]), 1, 0],
        [(3 * t[1]**2), (2 * t[1]), 1, 0]
    ])
    C = np.concatenate((c, d), axis=None)

    Ti = np.linalg.inv(T)
    y = np.matmul(Ti, C)

    if plot==True:
        fig = plt.subplot()
        fig.scatter(t, c, label='data')
        xt = np.arange(max(t)+1)
        yt = (y[0] * xt**3) + (y[1] * xt**2) + (y[2] * xt) + y[3]
        fig.plot(xt, yt, label='spline fit')

        plt.show()
    return y


def grad(start, end, t, c):
    d = []
    d.append(start)
    for n in range(len(t)-2):
        change = (c[n+2] - c[n]) / (t[n+2] - t[n])
        d.append(change)
    d.append(end)
    return d


def cr_spline(t, c, d):
    e = grad(-8.49258200155620, 6.14330506465390, t, d)

    ys = []
    zs = []

    fit_x = []
    fit_y = []
    fit_z = []

    for n in range(len(t)-1):
        tt = [t[n], t[n+1]]
        ct = [c[n], c[n+1]]
        dt = [d[n], d[n+1]]
        et = [e[n], e[n+1]]

        y = spline_fitting_grad(tt, ct, dt)
        ys.append(y)
        z = spline_fitting_grad(tt, dt, et)
        zs.append(z)

        if (n) == len(t)-2:
            xt = np.arange(tt[0], tt[1]+0.001, 0.001)
        else:
            xt = np.arange(tt[0], tt[1], 0.001)
        yt = (y[0] * xt**3) + (y[1] * xt**2) + (y[2] * xt) + y[3]
        zt = (z[0] * xt**3) + (z[1] * xt**2) + (z[2] * xt) + z[3]
        fit_x.append(xt)
        fit_y.append(yt)
        fit_z.append(zt)

    fit_x = np.concatenate((fit_x), axis=None)
    fit_y = np.concatenate((fit_y), axis=None)
    fit_z = np.concatenate((fit_z), axis=None)

    return fit_x, fit_y, fit_z


def total_energy(theta, theta_dot, m=M, L=L, g=g):
    energy = []
    for n in range(len(theta)):
        T = 1/2 * m * L**2 * theta_dot[n]**2
        U = -m * g * L * math.cos(theta[n])
        energy.append(T + U)
    return energy


# Data collection
time = [(n*0.01) for n in range(251)]
theta, P = solve(theta_0, theta_dot_0, t, h)

# time_fit, theta_fit, P_fit = cr_spline(time, theta, P)  # [0:215]
theta_fit = CubicSpline(time, theta) # , bc_type='clamped'
P_fit = CubicSpline(time, P)

time_f = np.arange(0, t, 0.001)
theta_f = theta_fit(time_f)
P_f = P_fit(time_f)

energy = total_energy(theta, P)
energy_fit = total_energy(theta_f, P_f)


# Graphs
fig1, ax = plt.subplots(figsize=(6, 6))
'''
Phase plot ~ Lagrange
'''

ax.scatter(theta, P, marker='.', color='#81B1D2')
ax.set_xlabel('Position (Rad))', fontsize=14)
ax.set_ylabel('Momentum (Rad/s))', fontsize=14)

fig1.savefig('Phase plot ~ Lagrange.png')
plt.show()


fig2, ax = plt.subplots(figsize=(8, 6))
'''
Energy plot ~ Lagrange
'''

ax.scatter(time, energy, marker='.', color='#FA8174')
ax.set_xlabel('Time (Seconds)', fontsize=14)
ax.set_ylabel('Energy (Joules)', fontsize=14)

fig2.savefig('Energy plot ~ Lagrange.png')
plt.show()


fig3, ax = plt.subplots(figsize=(8, 6))
'''
Fitted energy plot ~ Lagrange
'''

ax.plot(time_f, energy_fit, color='#FA8174')
ax.set_xlabel('Time (Seconds)', fontsize=14)
ax.set_ylabel('Energy (Joules)', fontsize=14)

fig3.savefig('Fitted energy plot ~ Lagrange.png')
plt.show()











#
#
#
'''
x = [(L * math.sin(theta_i)) for theta_i in theta]
y = [(-L * math.cos(theta_i)) for theta_i in theta]
x0, y0 = x[0], y[0]


fig3, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].set_aspect('equal')

line, = axs[0].plot([0, x0], [0, y0], lw=3, c='white')
bob_radius = 0.08  # Radius of the bob
circle = axs[0].add_patch(plt.Circle((x0, y0), bob_radius, fc='white', zorder=3))

axs[0].set_xlim([-L-0.5, L+0.5])
axs[0].set_ylim([-L-0.5, L])

theta_line, = axs[1].plot(time, theta, c='#81B1D2')
axs[1].set_xlim([0, max(time)])
axs[1].set_ylim([-np.pi, np.pi]) # [-np.pi, np.pi]
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Theta (rad)')
axs[1].set_title('Theta vs. Time')

def animate(i):
    """Update the animation at frame i."""
    line.set_data([0, x[i]], [0, y[i]])
    circle.set_center((x[i], y[i]))

    theta_line.set_data(t[:i], theta[:i])

nsteps = len(x)
nframes = nsteps
dt = t[1] - t[0]
interval = dt * 1000
'''
'''ani = animation.FuncAnimation(fig3, animate, frames=nframes, repeat=True, interval=interval)

plt.tight_layout()
plt.show()

ani = animation.FuncAnimation(fig3, animate, frames=nframes, repeat=True, interval=interval, blit=False)

ani.save('pendulum_animation.gif', writer='pillow', fps=30)

plt.close(fig3)'''

# test code
'''
print('')

print(theta[0])
print(theta[1])
print(theta[2])
print(theta[3])
print(theta[4])

print('')

print(pk[0])
print(pk[1])
print(pk[2])
print(pk[3])
print(pk[4])
'''

