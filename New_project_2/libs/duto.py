from libs.gc_eos_soave import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from libs.viscosity import *
from libs.eos_database import *
from libs.composicaogas import *


class duto:
    def __init__(self, gas, visc, Lc, D):
        self.visc = visc
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
        return (1/Rho) * (4*U/ self. D) * (T - self.T_solo)

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
        
        # => [condições de contorno] <=
        #T_in, P_in => entrada =>  [ | | | |]
        
        T[0] = 300.0
        P_init = 4500
        gas_init = self.gas.copy_change_conditions(T[0], P_init, None, 'gas')
        V[0] = gas_init.V.item()     # Volume específico de entrada    
        
        #saída [ | | | | |] => saída => Q_out  
        Q_out = 0.05  # [m³/s] <- substitua por seu valor real
        A = np.pi * (self.D**2) / 4  # Área da seção transversal do duto
        w[-1] = Q_out / A

        dTdt = np.zeros_like(T)
        dVdt = np.zeros_like(V)
        dwdt = np.zeros_like(w)
        
        for i in range(len(self.l)):
            gas2 = self.gas.copy_change_conditions(T[i], None, V[i], 'gas')
            
            rho = 1 / V[i]
            gas2.ci_real()
            Cv = gas2.Cvt #cv_real ou seria o ci_real? perguntar.
            
            mu = self.visc.evaluate_viscosity(T[i], gas2.P.item())
            Re = rho * w * (self.D / mu) #numero de reynolds, verificar se a equacao ta certa.

            f = self.fator_friccao(Re) #fator de fricção.

            kappa = coef_con_ter(gas2)

            h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)

            U = 1/((1/h_t) + (self.D/2*self.k_solo)*(np.arccosh(2*self.z_solo/self.D)))

            q = self.q_solo(rho, T[i], U) #calor.

            v_kg = V[i]/gas2.mixture.MM_m

            dT_dx = self.derivada_lagrange(self.l, T, i)
            dV_dx = self.derivada_lagrange(self.l, V, i)
            dw_dx = self.derivada_lagrange(self.l, w, i)

            dTdt[i] = (-w[i]*dT_dx -T[i]*((v_kg*gas2.dPdT)/Cv)) + (f * w[i]**2 * abs(w[i]) / (2 * self.D * Cv)) + q / Cv #mudar o V*
            dVdt[i] = (-w[i] * dV_dx) + dw_dx*V[i]
            dwdt[i] = ((-v_kg*gas2.dPdT*dT_dx) + (-v_kg*gas2.dPdV*dT_dx) + (-(w[i]**2))) + (f * w[i] * abs(w[i]) / (2 * self.D)) # Velocidade, utilizar para calcular a vazão no futuro.
        
        dydt = np.empty_like(y)
        dydt[0::3] = dTdt
        dydt[1::3] = dVdt
        dydt[2::3] = dwdt
        
        return dydt
