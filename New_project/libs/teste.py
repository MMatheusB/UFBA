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
        P1, T1, N, alpha, P_out  = u 
        dot_m, Tp, Vp = x
        Pp, P2, Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1 = z[0:11]
        
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
    gas = gc_eos_class()  # Supondo que a classe Gas está definida no eos_database
    comp = compression()  # Supondo existência da classe Compressor
    vlv = valve(kv=0.01)
    Vpp = 0.1    # Volume do plenum [m³]
    Lc = 0.5     # Comprimento característico [m]
    A1 = 0.01    # Área [m²]
    
    # Cria o sistema
    plenum_sys = plenum(gas, comp, vlv, Vpp, Lc, A1)
    
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