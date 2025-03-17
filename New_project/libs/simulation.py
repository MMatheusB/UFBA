# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:08:19 2024

@author: Rodrigo Meira
"""

from libs.eos_database import *
from libs.compressor_class import *
from libs.compression import *
from libs.gc_eos_soave import *
from libs.viscosity import *
from libs.plenum_system import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, plenum_sys, compressor, x0, z0, u0, nAlphas, alphas, nData, dt):
        self.plenum = plenum_sys
        self.compressor = compressor
        self.u0 = u0
        self.nAlphas = nAlphas
        self.alphas = alphas
        self.nData = nData
        self.dt = dt
        self.x0 = x0
        self.z0 = z0
    
    def system_residuals(self, y):
        x = y[:3]
        z = y[3:]
    
        ode_sym, alg_sym = self.plenum.evaluate_dae(None, x, z, self.u0)
    
        res_ode = np.array([ode_sym[i].item() for i in range(3)])
    
        res_alg = np.array([alg_sym[i] for i in range(11)])

        res = np.concatenate((res_ode, res_alg))
        return res
    
    def compute_steady_state(self):
        y0 = np.array(self.x0 + self.z0)
    
        sol = fsolve(self.system_residuals, y0, args = ())
    
        x_ss = sol[:3]
        z_ss = sol[3:]
        return x_ss, z_ss
    
    def run(self):
        
        x_ss, z_ss = self.compute_steady_state()
        
        x_sym = SX.sym('x', 3)
        z_sym = SX.sym('z', 11)
        u_sym = SX.sym('u', 5)
        
        # Avaliação do DAE
        ode_sym, alg_sym = self.plenum.evaluate_dae(None, x_sym, z_sym, u_sym)
        dae = {
            'x': x_sym,
            'z': z_sym,
            'p': u_sym,
            'ode': vertcat(*ode_sym),
            'alg': vertcat(*alg_sym)
        }
        
        integrator_solver = integrator('F', 'idas', dae, {'tf': self.dt})
        
        # Listas para armazenar os resultados
        time_steps = []
        x_values = []
        z_values = []
        alpha_values = []
        
        time = 0
        
        for i in range(self.nAlphas):
            self.u0[3] = self.alphas[i]  # Atualiza a abertura da válvula
            alpha_value = self.alphas[i] + np.random.normal(0, 0, self.nData)
            
            for j in range(self.nData):
                res = integrator_solver(x0=x_ss, z0=z_ss, p=self.u0)
                
                # Atualiza x0 e z0 para o próximo passo
                x_ss = np.array(res["xf"])
                z_ss = np.array(res["zf"])
                
                # Armazena os valores
                time_steps.append(time)
                x_values.append(x_ss.copy())
                z_values.append(z_ss.copy())
                alpha_values.append(self.u0[3])
                
                time += self.dt
        x_values = np.array(x_values)
        z_values = np.array(z_values)
        alpha_values = np.array(alpha_values)
        
        return x_values, z_values, time_steps, alpha_values