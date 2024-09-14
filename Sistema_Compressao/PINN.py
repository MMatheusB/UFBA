import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


A1 = 2.6 * 1e-3
Lc = 2
kv = 0.38
P1 = 8.5
P_out = 5
C = 200

def Valve_opening(t):
    alpha = int
    if t < 60 or 120 <= t < 180 or 240 <= t <= 300:
        alpha = 0.5
    elif 60 <= t < 120:
        alpha = 0.46
    elif 180 <= t < 240:
        alpha = 0.54
    return alpha


def ode_system(x,y):
    m, P = y[:, 0:1], y[:, 1:]
    dm_dt = dde.grad.jacobian(y, x, i=0)
    dP_dt = dde.grad.jacobian(y, x, i = 1)
    return [dm_dt - (A1/Lc)*((1.5 * P1) - P), dP_dt - (C**2)/2 * (m - Valve_opening(x) * kv * np.sqrt(P - P_out))]


def boundary(_, on_initial):
    return on_initial


geom = dde.geometry.TimeDomain(0, 300)
ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component = 0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component = 1)
data = dde.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, num_test = 100)

layer_size = [1] + [53] * 4 + [2]
net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")

