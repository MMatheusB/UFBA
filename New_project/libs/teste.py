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
        # P1, T1, N, alpha, P_out  = u
        P1 = u[0]
        T1  = u[1]
        N  = u[2]
        alpha  = u[3]
        P_out  = u[4]

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

        ddot_m = self.A1/self.Lc*(P2 - Pp)
        cte = self.Vpp * self.gas.mixture.MM_m
        dot_m_valve = self.valve.evaluate_flow(alpha, Pp - P_out)
        dot_Vp = -Vp**2/cte * (dot_m - dot_m_valve)
        dot_Tp = (dot_m*(gas_2.h - gas_p.h) +
                 Tp * gas_p.dPdT * Vp * (dot_m - dot_m_valve)) * Vp / (cte * gas_p.Cvt)

        a1 = Pp - gas_p.P
        a2 = P2 - gas_2.P
        a3, a4, a5, a6, a7, a8, a9, a10, a11 = self.compression.character_dae(
            [Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1],
            [N, dot_m, P1, T1])

        alg = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]

        return [ddot_m, dot_Tp, dot_Vp], alg

# Configuração da simulação
def run_simulation():
    # Parâmetros iniciais (exemplo)

    list_names  = ["CH4",	  "C2H6",	  "C3H8",	  "iC4H10",  "nC4H10",  "iC5H12",  "nC5H12",  "nC6H14",  "nC7H16",  "nC8H18",
               "nC9H20",  "nC10H22", "nC11H24", "nC12H26", "nC14H30", "N2", "H2O", "CO2", "C15+"]

    nwe_3 = [0.238095238095238,0.0262608309264608,0.0261719617862697,0.00648744723394801,
         0.0183070428793601,0.0112419462341702,0.0105754276827372,0.0193290379915574,
         0.0210708965703968,0.0190209179995643,0.0173567422429736,0.01597710063164,
         0.0148135692205084,0.0138181989380497,0.0122023902400072] + \
        [0.000844257, 0.002962305, 0.446300822] + [0.066207509]

    dict_composition_3 = {list_names[i]: nwe_3[i] for i in range(len(nwe_3))}


    mixture_nwe_3 = Mixture(list_of_species, dict_composition_3)

    volumn_desviation = [0]*19

    gas = gc_eos_class(mixture_nwe_3, 85+273.15, 1.2e4, None, 2, -1, Aij, volumn_desviation, 'gas')  # Supondo que a classe Gas está definida no eos_database

    compressor = CompressorClass()

      # Supondo existência da classe Compressor
    vlv = valve(kv=0.01)
    Vpp = 0.1    # Volume do plenum [m³]
    Lc = 0.5     # Comprimento característico [m]
    A1 = 0.01    # Área [m²]

    # Cria o sistema
    plenum_sys = plenum(gas, compressor, vlv, Vpp, Lc, A1)

    # Define variáveis simbólicas
    x = SX.sym('x', 3)      # [dot_m, Tp, Vp]
    z = SX.sym('z', 11)     # Variáveis algébricas
    u = SX.sym('u', 5)      # [P1, T1, N, alpha, P_out]

    # Obtém as equações DAE
    ode, alg = plenum_sys.evaluate_dae(None, x, z, u)

    # Cria função DAE
    dae = {'x': x, 'z': z, 'p': u, 'ode': vertcat(*ode), 'alg': vertcat(*alg)}

    # Configura integrador
    opts = {'tf': 0.1}  # Passo de integração de 0.1 segundos
    integrator = integrator('F', 'idas', dae, opts)

    # Condições iniciais
    x0 = [0.5, 300.0, 0.05]          # [dot_m, Tp, Vp]
    z0 = [101325.0, 101325.0] + [300.0]*9  # Valores iniciais para z
    u0 = [101325.0, 300.0, 10000.0, 0.5, 101325.0]  # Parâmetros fixos

    # Tempo de simulação
    t_sim = 10  # segundos
    n_steps = int(t_sim/opts['tf'])

    # Arrays para armazenamento
    time = np.zeros(n_steps+1)
    dot_m_hist = np.zeros(n_steps+1)
    Tp_hist = np.zeros(n_steps+1)
    Pp_hist = np.zeros(n_steps+1)

    # Loop de simulação
    for i in range(n_steps+1):
        if i > 0:
            res = integrator(x0=x0, z0=z0, p=u0)
            x0 = res['xf'].full().flatten()
            z0 = res['zf'].full().flatten()

        # Armazena resultados
        time[i] = i * opts['tf']
        dot_m_hist[i] = x0[0]
        Tp_hist[i] = x0[1]
        Pp_hist[i] = z0[0]

    # Plotagem dos resultados
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, dot_m_hist, 'b-', linewidth=2)
    plt.ylabel('Fluxo de Massa [kg/s]')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, Tp_hist, 'r-', linewidth=2)
    plt.ylabel('Temperatura [K]')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, Pp_hist/1e5, 'g-', linewidth=2)
    plt.ylabel('Pressão [bar]')
    plt.xlabel('Tempo [s]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()