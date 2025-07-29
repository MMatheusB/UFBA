from libs.gc_eos_soave import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from libs.viscosity import *
from libs.eos_database import *
from libs.composicaogas import *


class duto:
    def __init__(self, gas, Lc, D):
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

    def derivada_lagrange(x, f, i):
        n = len(x)
        df_dx = 0.0

        for j in range(n):
            if j == i:
                continue
            prod = 1.0 / (x[i] - x[j])
            for k in range(n):
                if k != i and k != j:
                    prod *= (x[i] - x[k]) / (x[j] - x[k])
            df_dx += f[j] * prod

        return df_dx

    def evaluate_dae(self, t, y):
        """
        Aqui eu devo calcular as variaveis diferenciais, sendo elas,
        Temperatura, Volume especifico, velocidade,
        a pressao vai ser calculada via PVT.
        lembretes:
               T, P são perturbações do sistema <= [  |  | | | |  |  ] => vazão volumetrica é perturbação do sistema no final do duto, com ela calcula-se o omega direto
        """
        T = y[0::3]
        V = y[1::3]
        w = y[2::3]
        dTdt = np.zeros_like(T)
        dVdt = np.zeros_like(V)
        dwdt = np.zeros_like(w)
        for i in range(len(self.l)):
            dT_dx = self.derivada_lagrange(self.l, T_array, i)
            dV_dx = self.derivada_lagrange(self.l, V_array, i)
            dw_dx = self.derivada_lagrange(self.l, w_array, i)
            gas2 = self.gas.copy_change_conditions(T, None, V, 'gas') # calculando o P via PVT.
            rho = 1 / V
        
            Cv = self.gas.cvT #cv_real ou seria o ci_real? perguntar.
        
            mu = viscosity.evaluate_viscosity(T, gas2.P.item()) #viscosidate? para calcular o numero de reynolds.

            Re = rho * w * (self.D / mu) #numero de reynolds, verificar se a equacao ta certa.

            f = self.fator_friccao(Re) #fator de fricção.

            kappa = coef_con_ter(gas2)

            h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)

            U = 1/((1/h_t) + (self.D/2*self.k_solo)*(np.arccosh(2*self.z_solo/self.D)))

            q = self.q_solo(rho, T, U) #calor.

            v_kg = V/gas2.mixture.MM_m


            dT_dt = (-w*dT_dx -T*((v_kg*gas2.dPdT)/Cv)) + (f * w**2 * abs(w) / (2 * self.D * Cv)) + q / Cv #mudar o V*
            dV_dt = (-w * dV_dx) + dw_dx*V
            dw_dt = ((-v_kg*gas2.dPdT*dT_dx) + (-v_kg*gas2.dPdV*dT_dx) + (-(w**2))) + (f * w * abs(w) / (2 * self.D)) # Velocidade, utilizar para calcular a vazão no futuro.

        return dT_dt, dV_dt, dw_dt
