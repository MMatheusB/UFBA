import casadi as ca
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from libs.viscosity import *
from libs.eos_database import *
from libs.gc_eos_soave import *
from libs.composicaogas import *
from libs.duto import *
from scipy.integrate import solve_ivp
from numpy import sum
from libs.duto_casadi import *
from libs.compressor_class import *
from libs.compression import *
from libs.simulation_duto import *
from libs.model import *

list_names = ["CH4", "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12", 
                  "nC6H14", "nC7H16", "nC8H18", "nC9H20", "nC10H22", "nC11H24", 
                   "nC12H26", "nC14H30", "N2", "H2O", "CO2", "C15+"]

nwe = [0.9834, 0.0061, 0.0015, 0.0003, 0.0003, 0.00055, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0008, 0.0]

dict_composition= {list_names[i]: nwe[i] for i in range(len(nwe))}

mixture = Mixture(list_of_species, dict_composition)

volumn_desviation = [0] * 19

Vpp = 2.0 
A1 = 2.6e-3
gas = gc_eos_class(mixture, 300, 4000, None, 1, 0, Aij, volumn_desviation, 'gas')
visc = viscosity(mixture, volumn_desviation)
D = 0.42
# Criar instância do duto com os parâmetros necessários  # ajuste conforme seu import real

# Número de nós (comprimento de self.l)

comp = CompressorClass()
compressor = compression(gas, comp, visc)
meu_duto = duto_casadi(gas, visc, compressor=compressor, Lc=200000.0, D=0.42)

from scipy.optimize import fsolve
import numpy as np

# --- Entradas conhecidas ---
u0 = 700
m_dot = 36
# --- Função residual (só para as variáveis livres) ---
def algebraic_residual(z_free, u, compressor):
    """
    Calcula o residual das equações algébricas do compressor
    com T2 e V2 fixos.
    """
    z_full = [z_free[0], z_free[1],
              z_free[2], z_free[3],
              z_free[4], z_free[5],
              z_free[6], z_free[7],
              z_free[8]]
    
    params = [u, m_dot/4, 4000, 300]  
    res = compressor.character_dae(z_full, params)
    return np.array(res).flatten()

Phi, eta, Mach, Gimp, G2, Gdif, PHI, G2s, k = compression.character(
    compressor, m=m_dot/4, N=u0, Gi_1=gas
)

z_guess = [Gimp.T, Gimp.V, Gdif.T, Gdif.V, G2s.T, G2s.V, G2.T, G2.V, gas.V]


# --- Resolve as equações algébricas ---
z_free_sol = fsolve(algebraic_residual, z_guess, args=(u0, compressor))

# Reconstrói o vetor completo com T2, V2 fixos
z_sol = [z_free_sol[0], z_free_sol[1],
         z_free_sol[2], z_free_sol[3],
         z_free_sol[4], z_free_sol[5],
         z_free_sol[6], z_free_sol[7],
         z_free_sol[8]]

print("Solução das variáveis algébricas:")
print(z_sol)

gas_p = gas.copy_change_conditions(z_sol[6], None, z_sol[7], 'gas')

n_nodes = len(meu_duto.l)
A = np.pi * (D**2) / 4  
MM = gas_p.mixture.MM_m  
v_kg = z_sol[7] / MM
rho = 1 / v_kg
w_teste = m_dot / (A * rho)
y0 = np.array([z_sol[6], z_sol[7], w_teste])

x_avaliar = meu_duto.l

# --- Criação do integrador CVODES ---
x = SX.sym('x')
y = SX.sym('y', 3)  # [T, V, w]
dy = meu_duto.estacionario(x, y)
f = Function('f', [x, y], [dy])

dae = {'x': y, 'p': x, 'ode': f(x, y)}

# --- Integração incremental ao longo do duto ---
y_current = y0.reshape(-1, 1)
Y_sol = [y_current.flatten()]

for i in range(1, len(x_avaliar)):
    dx = x_avaliar[i] - x_avaliar[i - 1]
    F_seg = integrator('F_seg', 'cvodes', dae,
                       {'tf': dx})
    res = F_seg(x0=y_current, p=x_avaliar[i - 1])
    y_current = res['xf']
    Y_sol.append(y_current.full().flatten())

Y_sol = np.array(Y_sol)
T_sol_estc = Y_sol[:, 0]
V_sol_estc = Y_sol[:, 1]
w_sol_estc = Y_sol[:, 2]

# --- Cálculo da pressão ---
P_sol_estc = []
for i in range(len(T_sol_estc)):
    gas2 = gas.copy_change_conditions(T_sol_estc[i], None, V_sol_estc[i], 'gas')
    P_current = gas2.P.item()
    P_sol_estc.append(P_current)

# print(f"{'x [m]':>8} | {'T [K]':>8} | {'V [m³/kg]':>10} | {'w [m/s]':>10} | {'P [Pa]':>10}")
# print("-"*60)
# for xi, Ti, Vi, wi, Pi in zip(x_avaliar, T_sol_estc, V_sol_estc, w_sol_estc, P_sol_estc):
#     print(f"{xi:8.2f} | {Ti:8.2f} | {Vi:10.6f} | {wi:10.6f} | {Pi:10.2f}")
print(T_sol_estc)
print(V_sol_estc)
print(w_sol_estc)

meu_sistema = duto_casadi(gas, visc, compressor=compressor, Lc=200000.0, D=0.42)
# --- Condições iniciais ---
T0 = T_sol_estc
V0 = V_sol_estc
w0 = w_sol_estc
gas_temp = gas.copy_change_conditions(T_sol_estc[-1], None, V_sol_estc[-1], 'gas')
y0 = np.empty(n_nodes * 3)
y0[0::3] = T0
y0[1::3] = V0
y0[2::3] = w0
print(y0)
z0 = z_sol + [rho * A * w_sol_estc[0], gas_temp.P]
y0 = np.array(y0, dtype=float)
z0 = np.array(z0, dtype=float)
print(z0)
u0 = np.array([700.0, 4000, 300.0, 2.1506])
dt = 300
# --- Simulação temporal ---
n_steps = 1500 # número de passos
t_sim = np.linspace(0, n_steps * dt, n_steps)
y_sol = np.zeros((n_steps, len(y0)))

sim = SimuladorDuto(meu_sistema)
resultados = sim.run(y0, z0, u0, 1, 1, [u0[0] - u0[0]*0.98], dt)
sim.plotar()