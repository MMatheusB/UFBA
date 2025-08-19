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
        return float((1/ (-16 * np.log(self.e_D / 3.7 - 5.02/ Re * np.log(self.e_D/ 3.7 - 5.02/Re * np.log(self.e_D / 3.7 + 13/Re)))))**2)
    
    def q_solo(self, Rho, T, U): 
        return float((1/Rho) * (4*U/ self. D) * (T - self.T_solo))

    def coef_cov_fluid(self, kappa, mu, Re, gas):
        P_r = (gas.Cpt*mu)/kappa
        h_t = (kappa/self.D)*(1/8)* ((1.82*np.log(10)*((Re - 1.64)**(-2))*(Re - 1000)*P_r)/(1.07 + 12.7*(1.82*np.log(10)*((Re - 1.64)**(-1))*(P_r**(2/3) - 1))))
        return float(h_t)

    def derivada_centrada(self, x, f, i):
        if i == 0:
        # Derivada pra frente
            return (f[1] - f[0]) / (x[1] - x[0])
        elif i == len(x) - 1:
        # Derivada pra trás
            return (f[-1] - f[-2]) / (x[-1] - x[-2])
        else:
        # Derivada central
            return (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])


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
            gas2 = self.gas.copy_change_conditions(T[i], None, V[i], 'gas')
            v_kg = V[i]/gas2.mixture.MM_m
            rho = 1 / v_kg
            gas2.ci_real()
            Cv = gas2.Cvt.item() #cv_real ou seria o ci_real? perguntar.
            
            mu = self.visc.evaluate_viscosity(T[i], gas2.P.item())
            Re = rho * w[i] * (self.D / mu) #numero de reynolds, verificar se a equacao ta certa.

            f = self.fator_friccao(Re) #fator de fricção.

            kappa = coef_con_ter(gas2)

            h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)

            U = 1/((1/h_t) + (self.D/2*self.k_solo)*(np.arccosh(2*self.z_solo/self.D)))

            q = self.q_solo(rho, T[i], U) #calor.
            dPdT = float(gas2.dPdT)
            dPdV = float(gas2.dPdV)

            dT_dx = self.derivada_centrada(self.l, T, i)
            dV_dx = self.derivada_centrada(self.l, V, i)
            dw_dx = self.derivada_centrada(self.l, w, i)
            
            matrix_dx = [[dT_dx],
                         [dV_dx],
                         [dw_dx]]
            matrix_a = np.array([
            [-w[i], 0.0, -T[i] * (v_kg * dPdT / Cv)],
            [0.0, -w[i], V[i]],
            [-v_kg * dPdT, -v_kg * dPdV, -w[i]]
            ], dtype=float)
    
            matrix_b = np.array([
            [f * w[i]**2 * abs(w[i]) / (2 * self.D * Cv) + q / Cv],
            [0.0],
            [f * w[i] * abs(w[i]) / (2 * self.D)]
            ], dtype=float)

            result = (matrix_a @ matrix_dx) + matrix_b

            dTdt[i] = result[0] 
            dVdt[i] = result[1]
            dwdt[i] = result[2]
        
        dydt = np.empty_like(y)
        dydt[0::3] = dTdt
        dydt[1::3] = dVdt
        dydt[2::3] = dwdt
        
        return dydt
    def estacionario(self, x, y):
        T, V, w = map(float, y)  # garante que T, V, w são floats puros
    
        gas2 = self.gas.copy_change_conditions(T, None, V, 'gas')
        v_kg = float(V / gas2.mixture.MM_m)
        gas2.ci_real()
    
        Cv = float(gas2.Cvt)  
        mu = float(self.visc.evaluate_viscosity(T, float(gas2.P)))
        rho = 1.0 / v_kg
        Re = rho * w * (self.D / mu)
        f = self.fator_friccao(Re)
        kappa = float(coef_con_ter(gas2))
    
        h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)
        U = 1.0 / ((1.0 / h_t) + (self.D / (2 * self.k_solo)) * (np.arccosh(2 * self.z_solo / self.D)))
        q = self.q_solo(rho, T, U)
    
        dPdT = float(gas2.dPdT)
        dPdV = float(gas2.dPdV)
    
        matrix_a = np.array([
        [-w, 0.0, -T * (v_kg * dPdT / Cv)],
        [0.0, -w, V],
        [-v_kg * dPdT, -v_kg * dPdV, -w]
        ], dtype=float)
    
        matrix_b = np.array([
        [f * w**2 * abs(w) / (2 * self.D * Cv) + q / Cv],
        [0.0],
        [f * w * abs(w) / (2 * self.D)]
        ], dtype=float)
    
        result = np.linalg.inv(matrix_a) @ matrix_b
        dTdx = float(result[0].item())
        dVdx = float(result[1].item())
        dwdx = float(result[2].item())

        return dTdx, dVdx, dwdx