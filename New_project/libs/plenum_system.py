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
        dot_Tp = (dot_m * (gas_2.h - gas_p.h) + Tp * gas_p.dPdT * Vp * (dot_m - dot_m_valve)) * Vp / (cte * gas_p.Cvt)

        a1 = Pp - gas_p.P
        a2 = P2 - gas_2.P
        a3, a4, a5, a6, a7, a8, a9, a10, a11 = self.compression.character_dae(
            [Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1],
            [N, dot_m, P1, T1])

        alg = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]

        return [ddot_m, dot_Tp, dot_Vp], alg
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    