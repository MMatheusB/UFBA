# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:13:27 2024

@author: rodri
"""
import numpy as np

class CompressorClass:
    def __init__(self, N=0):
        self.Dh1 = 0.032
        self.Dt1 = 0.074
        self.Dt2 = 0.128
        self.Dh2 = 0.128
        self.li = 0.053
        self.ld = 0.053
        self.B1b = 0.61
        self.sigma = 0.9
        self.D = 0.02
        self.E = 0.00160
        self.Lc = 1.253
        self.Vpp = 0.21
        self.N = N

        # Initialization calculations
        self.D1 = np.sqrt((self.Dt1 ** 2 + self.Dh1 ** 2) / 2)
        self.D2 = np.sqrt((self.Dt2 ** 2 + self.Dh2 ** 2) / 2)
        self.b = self.D / 2
        self.A1 = np.pi * self.D1 ** 2 / 4
        self.a2b = np.arctan(self.D1 * np.tan(self.B1b) / self.sigma / self.D2)
        self.U1 = self.N * np.pi * self.D1
        self.U2 = self.N * np.pi * self.D2
        self.Ah_ideal = self.sigma * self.U2 ** 2
        self.Ca2 = self.sigma * self.U2
        self.Vc = self.A1 * self.Lc

    def change_parameters(self):
        # Change parameters as needed
        self.Dh1 = 0.032
        self.Dt1 = 0.074
        self.Dt2 = 0.128
        self.Dh2 = 0.128
        self.li = 0.053
        self.ld = 0.053
        self.B1b = 0.61
        self.sigma = 0.9
        self.D = 0.02
        self.E = 0.00160
        self.Lc = 1.253
        self.Vpp = 0.21

    def update_speed(self, N):
        self.N = N
        self.U1 = self.N * np.pi * self.D1
        self.U2 = self.N * np.pi * self.D2
        self.Ah_ideal = self.sigma * self.U2 ** 2
        self.Ca2 = self.sigma * self.U2

    def compute_ws(self, N):
        U = N * np.pi * self.D2
        W = self.sigma * U ** 2
        return W

    def losses_incimp(self, m, fluid):
        roo1 = fluid.mixture.MM_m / fluid.V
        Ahii = 0.5 * (self.U1 - 1/np.tan(self.B1b) * m / roo1 / self.A1) ** 2
        return Ahii
    
    def losses_incdif(self,m,fluid):
           
        rooimp = fluid.mixture.MM_m/fluid.V
        Ahid = 0.5*(self.sigma*self.U2 - 1/np.tan(self.a2b)*m/rooimp/self.A1)**2
        
        return Ahid
    
    def friction(self, fluid, vis_model, l, m, factor):
        roo = fluid.mixture.MM_m / fluid.V
        vis = vis_model.evaluate_viscosity(fluid.T, fluid.P)
        visc = vis / roo
        Re = self.U2 * self.b / visc
        f = (-1.8 * np.log10(6.9 / Re + ((self.E / self.D) / 3.7) ** 1.1)) ** -2
        Ahf = f * l / (2 * self.D * (roo * self.A1 * np.sin(self.B1b)) ** 2) * m ** 2 * factor
        return Ahf

    def compute_efficiency(self, Ah_loss):
        eta = (self.Ah_ideal - Ah_loss) / self.Ah_ideal
        return eta
    
    def perdas_fricimp(self,fluid,vis_model,m):
            
        return self.friction(fluid,vis_model,self.li,m,1)
        
    def perdas_fricdif(self,fluid,vis_model,m):
        
        return self.friction(fluid,vis_model,self.ld,m,1)
        