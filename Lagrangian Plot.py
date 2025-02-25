from sympy import symbols, nsolve, sin, pi # solve
import matplotlib.pyplot as plt

theta_new = symbols('theta_new')

M = 1
L = 1
h = 0.01
theta_old = pi / 3
g = 9.81
p_k = 0

func = (((M * L**2)/h) * (theta_new - theta_old)) + ((M * g * L * h)/2 * sin((theta_old + theta_new) / 2)) - p_k

initial_guess = 1.05  # You can adjust this based on the expected range of solutions
sol = nsolve(func, theta_new, initial_guess)

# Print the solution
# print(sol)

def step(p_k, theta_old):
    M = 1
    L = 1
    h = 0.01
    g = 9.81

    theta_new = symbols('theta_new')
    func = (((M * L**2)/h) * (theta_new - theta_old)) + ((M * g * L * h)/2 * sin((theta_old + theta_new) / 2)) - p_k
    
    initial_guess = round(theta_old, 2)  
    th_n = nsolve(func, theta_new, initial_guess)

    p_new = (((M * L**2)/h) * (th_n - theta_old)) - ((M * g * L * h)/2 * sin((theta_old + th_n) / 2))

    return th_n, p_new

pk = []
theta = []
pk.append(0)
theta.append(pi/3)

for n in range(1000):
    th, p = step(pk[n], theta[n])
    pk.append(p)
    theta.append(th)

t = [(n*0.01) for n in range(1001)]

plt.plot(theta, pk)

plt.show()

fig, ax = plt.subplots(2)

ax[0].plot(t, theta)
ax[1].plot(t, pk)

plt.show()

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