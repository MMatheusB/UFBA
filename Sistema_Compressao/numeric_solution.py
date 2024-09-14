import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

A1 = 2.6e-3
Lc = 2
kv = 0.38
P1 = 8.5
P_out = 5
C = 479
initial_conditions = [7.948, 27/4]
intervalo = [np.linspace(0,60,100), np.linspace(60,120,100),np.linspace(120,180,100),np.linspace(180,240,100),np.linspace(240,300,100)]
x = ca.MX.sym('x', 2)
alpha = ca.MX.sym('alpha', 1)
x0_values = []
x1_values = []

for i in range(0,5):
    rhs = ca.vertcat((A1/Lc)*((1.5 * P1) - x[1]), (C**2)/2 * (x[0] - alpha * kv * np.sqrt(x[1] - P_out)))

    ode = {}
    ode['x'] = x
    ode['ode'] = rhs
    ode['p'] = alpha

    F = ca.integrator('F','cvodes', ode, 0, intervalo[i])
    
    if i ==0 or i ==2 or i==4:
        alpha0 = 0.5
    elif i == 1:
        alpha0 = 0.46
    elif i ==3:
        alpha0 = 0.54
    
    sol = F(x0 = [27/4, 7.948], p = alpha0)

    xf_values = np.array(sol["xf"])

    aux1, aux2 = xf_values

    x0_values.append(aux1)
    x1_values.append(aux2)
    initial_conditions = [x0_values[-1], x1_values[-1]]

plt.figure()
for i in range(0,5):
    plt.plot(intervalo[i], np.squeeze(x0_values[i]), label='x0(t)')
plt.grid(True)
plt.show()

plt.figure()
for i in range(0,5):
    plt.plot(intervalo[i], np.squeeze(x1_values[i]), label='x0(t)')
plt.grid(True)
plt.show()