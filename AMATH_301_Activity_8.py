import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

dt = 0.1
dx = 0.01
L = 3
x_span = np.arange(0, L + dx, dx)
t_span = np.arange(0, 4 + dt, dt)
len_t = len(t_span)
len_x = len(x_span)
k = 2
mu = k * dt / (2 * (dx ** 2))

dx = 0.01
L = 3
x_span = np.arange(0, L + dx, dx)
u_0 = np.cos(x_span) + x_span
u = np.cos(x_span) + x_span
A1 = u_0
print(A1)
main_diag = (1 + 2 * mu) * np.ones(len_x - 2)
second_diag = -mu * np.ones(len_x - 3)
A = np.diag(main_diag) + np.diag(second_diag, 1) + np.diag(second_diag, -1)
main_diag = (1 - 2 * mu) * np.ones(len_x - 2)
second_diag = mu * np.ones(len_x - 3)
B = np.diag(main_diag) + np.diag(second_diag, 1) + np.diag(second_diag, -1)
u_sol = np.zeros((len_x, len_t))

A2 = A
A3 = B
A4 = 0

u_sol[:, 0] = u
for i in range (1, len(t_span)):
    b = B @ u[1 : -1]
    if (i == 1):
        b_0 = b
        A4 = b_0
    u[1 : -1] = np.linalg.solve(A, b)
    u_sol[:, i] = u

plt.show()
A5 = u_sol[:, -1]
steady_state = lambda x: ((np.cos(3) + 2) / 3) * x + 1
A6 = steady_state(x_span)
A7 = np.zeros(len_t)
for i in range(len_t):
    A7[i] = np.linalg.norm(u_sol[:, i] - A6)
print(A1)
print(A5)
plt.plot(x_span, A1, label = "Initial")
plt.plot(x_span, A5, label = "Final")
plt.plot(x_span, A6, label = "Steady State")
plt.show()