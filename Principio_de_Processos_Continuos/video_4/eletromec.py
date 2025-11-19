import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parâmetros do sistema
R = 2.0    # resistência [ohm]
L = 0.5    # indutância [H]
K = 1.0    # constante eletromecânica
k = 50.0   # constante da mola [N/m]
m = 2.0    # massa [kg]
b = 1.0    # coeficiente de amortecimento [N·s/m]

# Tensão de entrada
def V(t):
    return 10 * np.sin(2 * np.pi * 1 * t)  # senoidal de 1 Hz

# Sistema de EDOs
# estado = [i, di/dt, x, dx/dt]
def sistema(t, y):
    i, di, x, dx = y
    
    # Equações:
    # 1) V = R*i + L*di/dt + K*dx
    # 2) K*i = k*x + m*d²x/dt² + b*dx
    
    # derivadas:
    d2i_dt2 = (V(t) - R*i - K*dx) / L
    d2x_dt2 = (K*i - k*x - b*dx) / m

    return [di, d2i_dt2, dx, d2x_dt2]

# Condições iniciais
y0 = [0, 0, 0, 0]  # i(0), di/dt(0), x(0), dx/dt(0)

# Intervalo de simulação
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Resolver EDO
sol = solve_ivp(sistema, t_span, y0, t_eval=t_eval)

# Plotar resultados
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(sol.t, sol.y[0], label='i(t)')
plt.xlabel('Tempo [s]')
plt.ylabel('Corrente [A]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(sol.t, sol.y[2], label='x(t)', color='r')
plt.xlabel('Tempo [s]')
plt.ylabel('Deslocamento [m]')
plt.legend()

plt.tight_layout()
plt.show()
