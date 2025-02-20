# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:08:19 2024

@author: Rodrigo Meira
"""

from eos_database import *
from compressor_class import *
from compression import *
from gc_eos_soave import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from viscosity import *

class valve:

    def __init__(self, kv=0.38, Cv=1):
        self.kv = kv
        self.Cv = Cv

    def evaluate_flow(self, alpha, delta_P):
        return alpha * self.kv * sqrt(delta_P)

class plenum:

    def __init__(self, gas, compression, valve, Vpp, Lc, A1):
        self.gas = gas
        self.Vpp = Vpp
        self.A1 = A1
        self.Lc = Lc
        self.compression = compression
        self.valve = valve

    def evaluate_dae(self, t, x, z, u):
        P1 = u[0]
        T1 = u[1]
        N = u[2]
        alpha = u[3]
        P_out = u[4]

        dot_m = x[0]
        Tp = x[1]
        Vp = x[2]

        Pp = z[0]
        P2 = z[1]
        Timp = z[2]
        Vimp = z[3]
        Tdif = z[4]
        Vdif = z[5]
        T2s = z[6]
        V2s = z[7]
        T2 = z[8]
        V2 = z[9]
        V1 = z[10]

        gas_2 = self.gas.copy_change_conditions(T2, None, V2, 'gas')
        gas_2s = self.gas.copy_change_conditions(T2s, None, V2s, 'gas')
        gas_p = self.gas.copy_change_conditions(Tp, None, Vp, 'gas')

        [g.h_gas() for g in [gas_2, gas_2s, gas_p]]
        [g.ci_real() for g in [gas_2, gas_2s, gas_p]]
        gas_p.evaluate_der_eos_P()

        ddot_m = self.A1 / self.Lc * (P2 - Pp)
        cte = self.Vpp * self.gas.mixture.MM_m
        dot_m_valve = self.valve.evaluate_flow(alpha, Pp - P_out)
        dot_Vp = -Vp**2 / cte * (dot_m - dot_m_valve)
        dot_Tp = (dot_m * (gas_2.h - gas_p.h)*0 + Tp * gas_p.dPdT * Vp * (dot_m - dot_m_valve)) * Vp / (cte * gas_p.Cvt)

        a1 = Pp - gas_p.P
        a2 = P2 - gas_2.P
        a3, a4, a5, a6, a7, a8, a9, a10, a11 = self.compression.character_dae(
            [Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1],
            [N, dot_m, P1, T1])

        alg = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]

        return [ddot_m, dot_Tp, dot_Vp], alg
def test_evaluate_dae():
    # Criando o objeto plenum
    list_names = ["CH4", "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12", 
                  "nC6H14", "nC7H16", "nC8H18", "nC9H20", "nC10H22", "nC11H24", 
                  "nC12H26", "nC14H30", "N2", "H2O", "CO2", "C15+"]

    nwe_3 = [0.9834, 0.0061, 0.0015, 0.0003, 0.0003, 0.00055, 0.0004, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0008, 0.0]

    dict_composition_3 = {list_names[i]: nwe_3[i] for i in range(len(nwe_3))}
    mixture_nwe_3 = Mixture(list_of_species, dict_composition_3)

    volumn_desviation = [0] * 19
    gas = gc_eos_class(mixture_nwe_3, 300, 4500, None, 1, 0, Aij, volumn_desviation, 'gas')  # T inicial em Kelvin

    comp = CompressorClass()
    visc = viscosity(mixture_nwe_3, volumn_desviation)
    compressor = compression(gas, comp, visc)
    vlv = valve(kv=0.38)
    Vpp = 2.0 
    Lc = 2.0 
    A1 = 2.6e-3
    plenum_sys = plenum(gas, compressor, vlv, Vpp, Lc, A1)

    dotm_0 = vlv.evaluate_flow(0.6,2500)

    # Definição de valores numéricos de teste
    x_test = [dotm_0, 320, gas.V / 2]  # Variáveis diferenciais
    z_test = [7500, 7500] + [320.0, gas.V / 2] * 4 + [gas.V]  # Variáveis algébricas
    u_test = [4500, 300, 750, 0.6, 5000]  # Parâmetros de entrada

    # Avaliação da função
    ode_values, alg_values = plenum_sys.evaluate_dae(None, x_test, z_test, u_test)

    # Imprimindo os resultados
    print("Valores das ODEs:")
    for i, val in enumerate(ode_values):
        print(f"ODE {i}: {val}")

    print("\nValores das equações algébricas:")
    for i, val in enumerate(alg_values):
        print(f"Algebraica {i}: {val}")
test_evaluate_dae()

def run_simulation():
    list_names = ["CH4", "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12", 
                  "nC6H14", "nC7H16", "nC8H18", "nC9H20", "nC10H22", "nC11H24", 
                  "nC12H26", "nC14H30", "N2", "H2O", "CO2", "C15+"]

    nwe_3 = [0.9834, 0.0061, 0.0015, 0.0003, 0.0003, 0.00055, 0.0004, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0008, 0.0]

    dict_composition_3 = {list_names[i]: nwe_3[i] for i in range(len(nwe_3))}
    mixture_nwe_3 = Mixture(list_of_species, dict_composition_3)

    volumn_desviation = [0] * 19
    gas = gc_eos_class(mixture_nwe_3, 300, 4500, None, 1, 0, Aij, volumn_desviation, 'gas')  # T inicial em Kelvin

    comp = CompressorClass()
    visc = viscosity(mixture_nwe_3, volumn_desviation)
    compressor = compression(gas, comp, visc)

    vlv = valve(kv=0.38)
    Vpp = 2.0 
    Lc = 2.0 
    A1 = 2.6e-3
    plenum_sys = plenum(gas, compressor, vlv, Vpp, Lc, A1)

    dotm_0 = vlv.evaluate_flow(0.6,2500)

    # Chutes iniciais (x0: variáveis diferenciais, z0: variáveis algébricas)
    x0 = [dotm_0, 320, gas.V / 2]
    z0 = [7500, 7500] + [320.0, gas.V / 2] * 4 + [gas.V]
    u0 = [4500, 300, 750, 0.6, 5000]

    # Definição dos símbolos para o DAE
    x_sym = SX.sym('x', 3)
    z_sym = SX.sym('z', 11)
    u_sym = SX.sym('u', 5)

    ode_sym, alg_sym = plenum_sys.evaluate_dae(None, x_sym, z_sym, u_sym)

    total_fun = ca.Function('f',[x_sym,z_sym,u_sym],ode_sym+alg_sym)

    Fun = lambda y : [f.full().flatten().item() for f in total_fun(y[0:3],y[3:],u0)]

    y0 = fsolve(Fun,x0+z0)

    print(y0)

    dae = {
        'x': x_sym,
        'z': z_sym,
        'p': u_sym,
        'ode': vertcat(*ode_sym),
        'alg': vertcat(*alg_sym)
    }

    t0 = 1
    tf = 0.1
    # Chamada atualizada do integrador com t0 e tf como argumentos
    integrator_solver = integrator('F', 'idas', dae, 0, 10, {
    'abstol': 1e-3,
    'reltol': 1e-3,
    'max_num_steps': 10000,     # Estatísticas da simulação
})

    t_sim = 10
    n_steps = int(t_sim / tf)
    time = np.zeros(n_steps + 1)
    dot_m_hist = np.zeros(n_steps + 1)
    Tp_hist = np.zeros(n_steps + 1)
    Pp_hist = np.zeros(n_steps + 1)

    x0_current, z0_current = x0.copy(), z0.copy()
    for i in range(n_steps + 1):
        if i > 0:
            res = integrator_solver(x0=x0_current, z0=z0_current, p=u0)
            x0_current = res['xf'].full().flatten()
            z0_current = res['zf'].full().flatten()
        time[i] = i * tf
        dot_m_hist[i] = x0_current[0]
        Tp_hist[i] = x0_current[1]
        Pp_hist[i] = z0_current[0]

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, dot_m_hist, 'b-')
    plt.ylabel('Fluxo de Massa [kg/s]')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, Tp_hist, 'r-')
    plt.ylabel('Temperatura [K]')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, Pp_hist / 1e5, 'g-')
    plt.ylabel('Pressão [bar]')
    plt.xlabel('Tempo [s]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("grafico.png", dpi=300)


if __name__ == "__main__":
    run_simulation()