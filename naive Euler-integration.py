# imported packages
import numpy as np 
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

'''
d20/dt2 + g/L sin0 = 0
'''

theta0 = 1
phi = 0 # 3.14
gravity = 9.81
length = 1
omega = (gravity / length)**(1/2)

f = lambda theta, t: theta * math.cos(omega * t + phi)
h = 0.25
t = np.arange(0, 20 + h, h)
time = np.arange(0, 20, 0.01)

theta = np.zeros(len(t))
theta[0] = theta0

for i in range(0, len(t)-1):
    theta[i+1] = theta[i] + h * f(t[i], theta[i])

x = [(-length * np.sin(theta_i)) for theta_i in theta]
y = [(-length * np.cos(theta_i)) for theta_i in theta]

z = [(theta0 * math.cos(omega * t + phi)) for t in time]

plt.plot(time, z, color='blue')
plt.plot(t, theta, color='red')
plt.show()

x0, y0 = x[0], y[0]

fig = plt.figure()
ax = fig.add_subplot(aspect='equal')

line, = ax.plot([0, x0], [0, y0], lw=3, c='black')
bob_radius = 0.08
circle = ax.add_patch(plt.Circle((x0,y0), bob_radius,
                      fc='black', zorder=3))

ax.set_xlim([-length-0.5, length+0.5])
ax.set_ylim([-length-0.5, length])

def animate(i):
    """Update the animation at frame i."""
    line.set_data([0, x[i]], [0, y[i]])
    circle.set_center((x[i], y[i]))

nsteps = len(x)
nframes = nsteps
dt = t[1]-t[0]
interval = dt * 1000
ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True,
                              interval=interval)

# Show the animation
plt.show()

# Explicitly close the figure
plt.close(fig)

# References
# https://www.acs.psu.edu/drussell/Demos/Pendulum/Pendula.html
# https://www.youtube.com/watch?v=xuxCk-VrF8c 'make the animations'
# https://en.wikipedia.org/wiki/Euler_method
# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter22.03-The-Euler-Method.html