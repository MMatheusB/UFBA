from scipy.optimize import fsolve
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
alpha0 = np.random.uniform(0.2, 0.8)
A1 = (2.6)*(10**-3)
Lc = 2
kv = 0.38
P1 = 8.5
P_out = 5
C = 479

def fun(variables, A1, Lc, kv, P1, P_out, C) :
    (x,y) = variables
    eqn_1 = (A1/Lc)* ((1.5 * P1) - y)
    eqn_2 = (C**2)/2 * (x - alpha0 * kv * np.sqrt(y - P_out))
    return [eqn_1, eqn_2]


result = fsolve(fun, (0, 10), args = (A1, Lc, kv, P1, P_out, C)) 

a = result[0]
b = result[1]
interval = [np.linspace(0,400,400), np.linspace(400,800,400),np.linspace(800,1200,400),np.linspace(1200,1600,400),np.linspace(1600,2000,400)]
x = ca.MX.sym('x', 2)
alpha = ca.MX.sym('alpha', 1)
x0_values = []
x1_values = []
alpha_values = [np.full(400, alpha0)]


for i in range(0,5):  
    if i ==0:
        alpha1 = alpha0
    else:
        alpha1 = np.random.uniform(0.2, 0.8)
        alpha_values.append(np.full(400, alpha1))
    
    rhs = ca.vertcat((A1/Lc)*((1.5 * P1) - x[1]), (C**2)/2 * (x[0] - alpha * kv * np.sqrt(x[1] - P_out)))
    ode = {'x' : x, 'ode' : rhs, 'p' : alpha }

    F = ca.integrator('F','idas', ode, interval[i][0], interval[i])
    
    sol = F(x0 = [a, b], p = alpha1)

    xf_values = np.array(sol["xf"])

    aux1, aux2 = xf_values
    x0_values.append(aux1)
    x1_values.append(aux2)
    a = aux1[-1]
    b = aux2[-1]


plt.figure("mass flow rate x time")
for i in range(0,5):
    plt.plot(interval[i], np.squeeze(x0_values[i]), label='x0(t)')
plt.grid(True)
plt.xlabel('Time')  
plt.ylabel('Value')

plt.figure("Plenum pressure x time")
for i in range(0,5):
    plt.plot(interval[i], np.squeeze(x1_values[i]), label='x0(t)')
plt.grid(True)
plt.xlabel('Time')  
plt.ylabel('Value')

plt.figure("alpha x time")
for i in range(0,5):
    plt.plot(interval[i], np.squeeze(alpha_values[i]), linestyle=':')
plt.grid(True)
plt.xlabel('time') 
plt.ylabel('Value')


plt.show()
