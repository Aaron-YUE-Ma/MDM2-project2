from sympy import symbols, nsolve, sin, pi # solve
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

plt.style.use('dark_background')

M = 1
L = 1
g = 9.81

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


theta, P = solve(pi/3, 0, 10, 0.01)
t = [(n*0.01) for n in range(1001)]


fig1, ax = plt.subplots()

ax.plot(theta, P, color='#81B1D2')
ax.set(title='Phase Space', xlabel='Position', ylabel='Momentum')

plt.show()


fig2, axs = plt.subplots(2)

axs[0].plot(t, theta, color='#81B1D2')
axs[0].set(title='', xlabel='Time (s)', ylabel='Angle (Rad)')
axs[1].plot(t, P, color='#FA8174')
axs[1].set(title='', xlabel='Time (s)', ylabel='Momentum ()')

plt.show()


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

theta_line, = axs[1].plot(t, theta, c='#81B1D2')
axs[1].set_xlim([0, max(t)])
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

ani = animation.FuncAnimation(fig3, animate, frames=nframes, repeat=True, interval=interval)

plt.tight_layout()
plt.show()

ani = animation.FuncAnimation(fig3, animate, frames=nframes, repeat=True, interval=interval, blit=False)

ani.save('pendulum_animation.gif', writer='pillow', fps=30)

plt.close(fig3)

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

