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
import matplotlib.pyplot as plt
from viscosity import *

class valve:

    def __init__(self, kv=1, Cv=1):
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
        dot_Tp = (dot_m * (gas_2.h - gas_p.h) +
                 Tp * gas_p.dPdT * Vp * (dot_m - dot_m_valve)) * Vp / (cte * gas_p.Cvt)

        a1 = Pp - gas_p.P
        a2 = P2 - gas_2.P
        a3, a4, a5, a6, a7, a8, a9, a10, a11 = self.compression.character_dae(
            [Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1],
            [N, dot_m, P1, T1])

        alg = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]

        return [ddot_m, dot_Tp, dot_Vp], alg

def run_simulation():
    list_names = ["CH4", "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12", 
                  "nC6H14", "nC7H16", "nC8H18", "nC9H20", "nC10H22", "nC11H24", 
                  "nC12H26", "nC14H30", "N2", "H2O", "CO2", "C15+"]

    nwe_3 = [0.9834, 0.0061, 0.0015, 0.0003, 0.0003, 0.00055, 0.0004, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0008, 0.0]

    dict_composition_3 = {list_names[i]: nwe_3[i] for i in range(len(nwe_3))}
    mixture_nwe_3 = Mixture(list_of_species, dict_composition_3)

    volumn_desviation = [0] * 19
    gas = gc_eos_class(mixture_nwe_3, 358.15, 1.2e4, None, 1, 0, Aij, volumn_desviation, 'gas')  # T inicial = 85°C

    comp = CompressorClass()
    visc = viscosity(mixture_nwe_3, nwe_3)
    compressor = compression(gas, comp, visc)

    vlv = valve(kv=0.01)
    Vpp, Lc, A1 = 0.1, 0.5, 0.01
    plenum_sys = plenum(gas, compressor, vlv, Vpp, Lc, A1)

    x0 = [0.1, 300.0, 0.01]  # Fluxo de massa menor, temperatura inicial mais baixa
    z0 = [2.0e5, 2.0e5] + [300.0] * 9  # Pressão inicial = 2.0 bar
    u0 = [2.0e5, 300.0, 5000.0, 0.3, 1.5e5]  # Pressão de saída = 1.5 bar

    x_sym = SX.sym('x', 3)
    z_sym = SX.sym('z', 11)
    u_sym = SX.sym('u', 5)

    ode_sym, alg_sym = plenum_sys.evaluate_dae(None, x_sym, z_sym, u_sym)

    dae = {'x': x_sym, 'z': z_sym, 'p': u_sym, 'ode': vertcat(*ode_sym), 'alg': vertcat(*alg_sym)}

    opts = {
        'tf': 0.1,
        'abstol': 1e-6,
        'reltol': 1e-6,
        'max_num_steps': 10000
    }
    integrator_solver = integrator('F', 'idas', dae, opts)

    t_sim, n_steps = 10, int(10 / opts['tf'])
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
        time[i] = i * opts['tf']
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
    plt.show()

if __name__ == "__main__":
    run_simulation()