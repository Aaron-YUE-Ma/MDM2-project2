import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('dark_background')


# Initial conditions 
q = []
q.append(math.pi/3)
q3 = []
q3.append(math.pi/3)
v = []
v.append(0)
v3 = []
v3.append(0)
h = 0.01
g = 9.81
L = 1
M = 1


# Functions
def step1(q_old, v_old, h, g, L):
    q_new = q_old + h * v_old
    v_new = v_old - h * (g/L) * math.sin(q_old)
    return q_new, v_new
    

def step2(q_old, v_old, h, g, L):
    v_new = v_old - h * (9.81/L) * math.sin(q_old)
    q_new = q_old + h * v_new
    return q_new, v_new


def total_energy(theta, theta_dot, m=M, L=L, g=g):
    energy = []
    for n in range(len(theta)):
        T = 1/2 * m * L**2 * theta_dot[n]**2
        U = -m * g * L * math.cos(theta[n])
        energy.append(T + U)
    return energy


# Data collection
for n in range(250):
    q_n, v_n = step1(q[n], v[n], h, g, L)
    q.append(q_n)
    v.append(v_n)


start = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
t = [(n*0.01) for n in range(251)]

for n in range(250):
    q_n, v_n = step2(q3[n], v3[n], h, q, L)
    q3.append(q_n)
    v3.append(v_n)

energy = total_energy(q3, v3)
energy1 = total_energy(q, v)



# Graphs
fig1, ax = plt.subplots(figsize=(6, 6))
'''
Phase plot ~ Explicit
'''

ax.scatter(q, v, marker='.', color='#81B1D2')
ax.set_xlabel('Position (Rad))', fontsize=14)
ax.set_ylabel('Momentum (Rad/s))', fontsize=14)

fig1.savefig('Phase plot ~ Explicit.png')
plt.show()


fig2, ax = plt.subplots(figsize=(8, 6))
'''
Total energy plot ~ Explicit
'''

ax.scatter(t, energy1, marker='.', color='#FA8174')
ax.set_xlabel('Time (Seconds)', fontsize=14)
ax.set_ylabel('Energy (Joules)', fontsize=14)

fig2.savefig('Energy plot ~ Explicit.png')
plt.show()


fig3, ax = plt.subplots(figsize=(6, 6))
'''
Phase plot ~ Symplectic
'''

ax.scatter(q3, v3, marker='.', color='#81B1D2')
ax.set_xlabel('Position (Rad))', fontsize=14)
ax.set_ylabel('Momentum (Rad/s))', fontsize=14)

fig3.savefig('Phase plot ~ Symplectic.png')
plt.show()


fig4, ax = plt.subplots(figsize=(8, 6))
'''
Total energy plot (Spline fitted) ~ Symplectic
'''

ax.scatter(t, energy, marker='.', color='#FA8174')
ax.set_xlabel('Time (Seconds)', fontsize=14)
ax.set_ylabel('Energy (Joules)', fontsize=14)

fig4.savefig('Energy plot ~ Symplectic.png')
plt.show()

