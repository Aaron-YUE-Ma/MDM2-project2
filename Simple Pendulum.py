import numpy as np 
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def theta(t, theta0, phi, omega):
    '''
    The simple harmonic solution is
    theta(t) = theta0 * cos(omega * t + phi)
    with omega = (gravity / length)^(1/2)
    '''
    return (theta0 * math.cos(omega * t + phi))

theta0 = 1
phi = 3.14
gravity = 9.81
length = 1
omega = (gravity / length)**(1/2)

time = np.arange(0, 10, 0.01)
z = [theta(t,theta0, phi, omega) for t in time]

# Correcting x and y computation:
x = [(-length * np.sin(z_i)) for z_i in z]
y = [(-length * np.cos(z_i)) for z_i in z]

plt.plot(time, z)
plt.show()

x0, y0 = x[0], y[0]

fig = plt.figure()
ax = fig.add_subplot(aspect='equal')

line, = ax.plot([0, x0], [0, y0], lw=3, c='black')
bob_radius = 0.08
circle = ax.add_patch(plt.Circle((x0,y0), bob_radius,
                      fc='black', zorder=3))

ax.set_xlim([-max(x)-0.5, max(x)+0.5])
ax.set_ylim([min(y)-0.5,0.5])

def animate(i):
    """Update the animation at frame i."""
    line.set_data([0, x[i]], [0, y[i]])
    circle.set_center((x[i], y[i]))

nsteps = len(x)
nframes = nsteps
dt = time[1]-time[0]
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