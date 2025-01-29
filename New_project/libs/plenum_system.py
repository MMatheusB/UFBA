# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:08:19 2024

@author: Rodrigo Meira
"""

from eos_database import *
from casadi import *
from numpy import exp, log, array, roots, zeros, linalg
from math import isnan

class valve:
    
    def __init__(self,kv = 1, Cv = 1):
        self.kv = kv
        self.Cv = Cv
    
    def evaluate_flow(self,alpha,delta_P):
        
        return alpha*self.kv*delta_P**0.5
    

class plenum:
    
    def __init__(self,gas,compression,valve,Vpp,Lc,A1):
        
        self.gas = gas
        self.Vpp = Vpp
        self.A1 = A1
        self.Lc = Lc
        self.compression = compression
        self.valve = valve
    
    def evaluate_dae(self,t,x,z,u):
        
        P1, T1, N, alpha, P_out  = u 
        dot_m, Tp, Vp = x
        Pp, P2, Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1 = z[0:2]
        
        gas_2 = self.gas.copy_change_conditions(T2,None,V2,'gas')
        gas_2s = self.gas.copy_change_conditions(T2s,None,V2s,'gas')
        gas_p = self.gas.copy_change_conditions(Tp,None,Vp,'gas')
        
        [gas.h_gas() for gas in [gas_2, gas_2s, gas_p]]
        
        [gas.ci_real() for gas in [gas_2, gas_2s, gas_p]]
        
        gas_p.evaluate_der_eos_P()
        
        ddot_m = self.A1/self.Lc*(P2-Pp)
        
        cte = self.Vpp*self.gas.mixture.MM_m 
        
        dot_m_valve = self.valve.evaluate_flow(alpha,Pp-P_out)
        
        dot_Vp = -Vp**2/cte*(dot_m - dot_m_valve)
        
        dot_Tp = (dot_m*(gas_2.h-gas_p.h) + \
                  Tp*gas_p.dPdT*Vp*(dot_m - dot_m_valve))*Vp/cte/gas_p.Cvt
        
        a1, a2 = [Pp - gas_p.P, P2 - gas_2.P]
            
        a3, a4, a5, a6, a7, a8, a9, a10, a11 = self.compressor.character_dae(self,
                                                                             [Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1],
                                                                             [N, dot_m, P1, T1])
        
        alg = [a1, a2, a3, a4, a5, a6,a7, a8, a9, a10, a11]
        
        
        return [ddot_m, dot_Tp, dot_Vp] + alg
        

    def simulate_plenum():
        # Variáveis de estado
        x = MX.sym("x", 3)  # [dot_m, Tp, Vp]
        
        # Variáveis algébricas
        z = MX.sym("z", 11)  # [Pp, P2, Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1]
        
        # Variáveis de controle
        u = MX.sym("u", 5)  # [P1, T1, N, alpha, P_out]
        

        gas = ...  
        compression = ... 
        valve_system = valve(kv=1, Cv=1)
        
        # Criar o plenum
        my_plenum = plenum(gas, compression, valve_system, Vpp=1, Lc=1, A1=1)
        
        # Avaliar as equações DAE
        dae_rhs = my_plenum.evaluate_dae(0, x, z, u)
        
        f_x = vertcat(*dae_rhs[:3]) 
        g_z = vertcat(*dae_rhs[3:])  
        
        dae = {'x': x, 'z': z, 'p': u, 'ode': f_x, 'alg': g_z}
        
        # Configurar o integrador
        opts = {'tf': 1.0}  
        integrator = integrator('integrator', 'idas', dae, opts)
        
        # Condições iniciais
        x0 = np.array([0, 300, 1])  
        z0 = np.zeros(11)  
        u0 = np.array([1, 300, 3000, 0.5, 1])  
        
        # Simular
        results = []
        for t in np.linspace(0, 10, 100):  
            res = integrator(x0=x0, z0=z0, p=u0)
            x0 = res['xf']  
            z0 = res['zf']  
            results.append((t, x0, z0))
        
        return results

# Executar a simulação
    simulation_results = simulate_plenum()

# Processar e visualizar os resultados
    for time, state, algebraic in simulation_results:
        print(f"Time: {time}, States: {state}, Algebraic: {algebraic}")

        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    