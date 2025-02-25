from sympy import symbols, nsolve, sin, pi # solve
import matplotlib.pyplot as plt
import math

plt.style.use('dark_background')


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

