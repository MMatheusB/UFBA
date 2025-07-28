from libs.gc_eos_soave import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from libs.viscosity import *
from libs.eos_database import *
from libs.composicaogas import *


class duto:
    def _init_(self, gas, Lc, D, T_solo):
        self.gas = gas
        self.Lc = Lc
        self.D = D
        self.e_D = 1.5 * 1e-6 #m
        self.k_solo = 0.89 # w / (m*K)
        self.T_solo = 15 # C
        self.z_solo = 2 # m
        self.l = [0, 0.15*self.Lc, 0.30*self.Lc, 0.40*self.Lc, 0.50*self.Lc, 0.60*self.Lc, 0.70*self.Lc, 0.85*self.Lc, self.Lc]

    def fator_friccao(self, Re): 
        return (1/ (-4 * np.log((self.e_D / 3.7) - (5.02/ Re) * (np.log((self.e_D/ 3.7) - (5.02/Re) * (np.log((self.e_D / 3.7) + (13/Re))))))))**2
    
    def q_solo(self, Rho, T, U): 
        return (1/Rho) * (4*self.U/ self. D) * (T - self.T_solo)

    def coef_cov_fluid(self, kappa, mu, Re, gas):
        P_r = (gas.Cpt*mu)/kappa
        h_t = (kappa/self.D)*(1/8)* ((1.82*np.log(10)*((Re - 1.64)**(-2))*(Re - 1000)*P_r)/(1.07 + 12.7*(1.82*np.log(10)*((Re - 1.64)**(-1))*(P_r**(2/3) - 1))))
        return h_t


    def evaluate_dae(self, t, x, z, u):
        """
        Aqui eu devo calcular as variaveis diferenciais, sendo elas,
        Temperatura, Volume especifico, velocidade,
        a pressao vai ser calculada via PVT.
        lembretes:
               T, P são perturbações do sistema <= [  |  | | | |  |  ] => vazão volumetrica é perturbação do sistema no final do duto, com ela calcula-se o omega direto
        """
        T, V, w = x
        gas2 = self.gas.copy_change_conditions(T, None, V, 'gas') # calculando o P via PVT.
        rho = 1 / V
        
        Cv = self.gas.cvT #cv_real ou seria o ci_real? perguntar.
        
        mu = viscosity.evaluate_viscosity(T, gas2.P.item()) #viscosidate? para calcular o numero de reynolds.

        Re = rho * w * (self.D / mu) #numero de reynolds, verificar se a equacao ta certa.

        f = self.fator_friccao(Re) #fator de friccao.

        kappa = coef_con_ter(gas2)

        h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)

        U = 1/((1/h_t) + (self.D/2*self.k_solo)*(np.arccosh(2*self.z_solo/self.D)))

        q = self.q_solo(rho, T, U) #calor.

        dT_dx = 0
        dV_dx = 0
        dw_dt = 0

        for i in range(len(self.l)):
            prod_x = 1
            for k in range(len(self.l)):
                if k != i:
                    prod_x *= (self.l - self.l[k])/(self.l[i] - self.l[k]) 
            dT_dx += T*(prod_x)
            dV_dx += V*(prod_x)
            dw_dt += w *(prod_x)

        dT_dt = (f * w**2 * abs(w) / (2 * self.D * Cv)) + q / Cv #Temperatura e coisas.
        dV_dt = 0 
        dw_dt = f * w * abs(w) / (2 * self.D) # Velocidade, utilizar para calcular a vazao no futuro.

        return dT_dt, dV_dt, dw_dt
