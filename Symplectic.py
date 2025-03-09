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
for n in range(1000):
    q_n, v_n = step1(q[n], v[n], h, g, L)
    q.append(q_n)
    v.append(v_n)


start = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
t = [(n*0.01) for n in range(1001)]

for n in range(1000):
    q_n, v_n = step2(q3[n], v3[n], h, q, L)
    q3.append(q_n)
    v3.append(v_n)

energy = total_energy(q3, v3)


# Graphs
fig1, ax = plt.subplots(1, 2, figsize=(15 ,6))
'''

'''

ax[0].plot(q, v, color='#81B1D2')
ax[0].set(title='', xlabel='Position', ylabel='Momentum')
ax[1].plot(q3, v3, color='#FA8174')
ax[1].set(title='', xlabel='Position', ylabel='Momentum')

plt.show()


fig2, ax = plt.subplots(2, figsize=(12 ,6))
'''

'''

ax[0].plot(t, q3, color='#81B1D2')
ax[0].set(title='', xlabel='', ylabel='Position')
ax[1].plot(t, v3, color='#FA8174')
ax[1].set(title='', xlabel='Time', ylabel='Momentum')

plt.show()


fig3, ax = plt.subplots(figsize=(12 ,6))
'''

'''

ax.plot(t, energy, color='#81B1D2')
ax.set(title='', xlabel='Time', ylabel='Energy')

plt.show()