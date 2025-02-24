import numpy as np
import matplotlib.pyplot as plt

# Spline fitting using points
def spline_fitting_points(t, c):
    T = np.array([
        [t[0]**3, t[0]**2, t[0], 1],
        [t[1]**3, t[1]**2, t[1], 1],
        [t[2]**3, t[2]**2, t[2], 1],
        [t[3]**3, t[3]**2, t[3], 1]
    ])
    Ti = np.linalg.inv(T)
    y = np.matmul(Ti, c)

    fig = plt.subplot()
    fig.scatter(t, c, label='data')
    xt = np.arange(max(t)+1)
    yt = (y[0] * xt**3) + (y[1] * xt**2) + (y[2] * xt) + y[3]
    fig.plot(xt, yt, label='spline fit')

    plt.show()
    return y

# test code
'''
t = np.array([0, 2, 7, 10])
C = np.array([1, 1.5, 3, 1])
y = spline_fitting_points(t, C)
print(y)
'''

# Spline fitting using gradients
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

# test code
'''
t = [0, 10]
c = [1, 1]
d = [0, -2]
y = spline_fitting_grad(t, c, d, plot=True)
print(y)
'''

# Catmull-Rom Spline
def grad(start, end, t, c):
    d = []
    d.append(start)
    for n in range(len(t)-2):
        change = (c[n+2] - c[n]) / (t[n+2] - t[n])
        d.append(change)
    d.append(end)
    return d

def cr_spline(t, c, d):
    '''
    the gradient (d) only wants two inputs like [x, y] for the start and end
    '''
    d = grad(d[0], d[1], t, c)
    ys = []
    fit_x = []
    fit_y = []
    for n in range(len(t)-1):
        tt = [t[n], t[n+1]]
        ct = [c[n], c[n+1]]
        dt = [d[n], d[n+1]]
        y = spline_fitting_grad(tt, ct, dt)
        ys.append(y)

        if (n) == len(t)-2:
            xt = np.arange(tt[0], tt[1]+0.1, 0.1)
        else:
            xt = np.arange(tt[0], tt[1], 0.1)
        yt = (y[0] * xt**3) + (y[1] * xt**2) + (y[2] * xt) + y[3]
        fit_x.append(xt)
        fit_y.append(yt)

    fit_x = np.concatenate((fit_x), axis=None)
    fit_y = np.concatenate((fit_y), axis=None)

    fig = plt.subplot()
    fig.scatter(t, c, label='data')
    xt = np.arange(max(t)+1)
    yt = (y[0] * xt**3) + (y[1] * xt**2) + (y[2] * xt) + y[3]
    fig.plot(fit_x, fit_y, label='spline fit')

    plt.show()
    return ys

# test code

t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
c = [4, 4, 2, 2, 3.5, 1.5, 5, 2, 5, 4.5, 4]
d = [0, -1]

y = cr_spline(t, c, d)
# print(y)

# https://www.youtube.com/watch?v=YMl25iCCRew
# https://www.youtube.com/watch?v=DLsqkWV6Cag
