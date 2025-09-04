from libs.gc_eos_soave import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from libs.viscosity import *
from libs.eos_database import *
from libs.composicaogas import *
from builtins import sum


class duto:
    def __init__(self, gas, visc, Lc, D):
        self.visc = visc
        self.gas = gas
        self.Lc = Lc
        self.D = D
        self.e_D = 1.5 * 1e-6 #m
        self.k_solo = 0.89 # w / (m*K)
        self.T_solo = 15 + 273.15# C
        self.z_solo = 2 # m
        self.l = [i*0.05*self.Lc for i in range(0,20)]
        #[0, 0.1*self.Lc, 0.2*self.Lc, 0.30*self.Lc, 0.4*self.Lc, 0.5*self.Lc, 0.60*self.Lc, 0.70*self.Lc, 0.80*self.Lc, 0.9*self.Lc, self.Lc]

    def fator_friccao(self, Re): 
        return 0.25*float(-4 * np.log10(self.e_D / 3.7 / self.D - 5.02/ Re * np.log10(self.e_D/ 3.7 / self.D - 5.02/Re * np.log10(self.e_D / 3.7  / self.D + 13/Re))))**(-2)
    
    def q_solo(self, Rho, T, U): 
        return float((1/Rho) * (4*U/ self. D) * (self.T_solo - T))

    def coef_cov_fluid(self, kappa, mu, Re, gas):
        P_r = (gas.Cpt*1000/gas.mixture.MM_m*mu)/kappa
        ft = (1.82*np.log10(Re) - 1.64)**(-2)
        Nu = (ft/8)*(Re - 1000)*P_r/(1.07 + 12.7*((ft/8)**(0.5))*(P_r**(2/3) - 1))
        h_t = Nu*kappa/self.D
        return float(h_t)

    def derivada_centrada(self, x, f, i):

        h = x[1] - x[0]  # assume espaçamento uniforme
    
        if i == 0:
        # progressiva de 3 pontos (ordem 2)
            return (-3*f[0] + 4*f[1] - f[2]) / (x[2] - x[0])
        elif i == len(x) - 1:
        # regressiva de 3 pontos (ordem 2)
            return (3*f[-1] - 4*f[-2] + f[-3]) / (x[-1] - x[-3])
        else:
        # centrada de 3 pontos (ordem 2)
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
        #no inicio a derivada de T, V é 0(no tempo)
        for i in range(len(self.l)):
            gas2 = self.gas.copy_change_conditions(T[i], None, V[i], 'gas')
            v_kg = V[i]/gas2.mixture.MM_m
            rho = 1 / v_kg
            gas2.ci_real()
            Cv = float(gas2.Cvt)/gas2.mixture.MM_m*1000 #cv_real ou seria o ci_real? perguntar.
            
            mu = self.visc.evaluate_viscosity(T[i], gas2.P.item())
            Re = rho * w[i] * (self.D / mu) #numero de reynolds, verificar se a equacao ta certa.

            f = self.fator_friccao(Re) #fator de fricção.

            kappa = coef_con_ter(gas2)

            h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)

            U = 1/((1/h_t) + (self.D/2*self.k_solo)*(np.arccosh(2*self.z_solo/self.D)))

            q = self.q_solo(rho, T[i], U)
            #calor.
            dPdT = float(gas2.dPdT)*1000
            dPdV = float(gas2.dPdV)*1000

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
            [-f * w[i] * abs(w[i]) / (2 * self.D)]
            ], dtype=float)

            result = (matrix_a @ matrix_dx) + matrix_b
            
            if i==0:
                dTdt[i] = 0
                dVdt[i] = 0
                dwdt[i] = result[0]
            elif i== len(self.l) - 1:
                dTdt[i] = result[0] 
                dVdt[i] = result[1]
                dwdt[i] = 0
            else:
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
    
        Cv = float(gas2.Cvt)/gas2.mixture.MM_m*1000
        mu = float(self.visc.evaluate_viscosity(T, float(gas2.P)))
        rho = 1 / v_kg
        Re = rho * w * (self.D / mu)
        f = self.fator_friccao(Re)
        kappa = float(coef_con_ter(gas2))
    
        h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)
        U = 1.0 / ((1.0 / h_t) + (self.D / (2 * self.k_solo)) * (np.arccosh(2 * self.z_solo / self.D)))
        q = self.q_solo(rho, T, U)

        
        dPdT = float(gas2.dPdT)*1000
        dPdV = float(gas2.dPdV)*1000
    
        matrix_a = np.array([
            [-w, 0.0, -T * (v_kg * dPdT / Cv)],
            [0.0, -w, V],
            [-v_kg * dPdT, -v_kg * dPdV, -w]
        ], dtype=float)

        matrix_b = np.array([
            [f * w**2 * abs(w) / (2 * self.D * Cv) + q / Cv],
            [0.0],
            [-f * w * abs(w) / (2 * self.D)]
        ], dtype=float)
        
        result = -np.linalg.inv(matrix_a) @ matrix_b
        dTdx = float(result[0].item())
        dVdx = float(result[1].item())
        dwdx = float(result[2].item())

        return dTdx, dVdx, dwdx