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
    def __init__(self, gas, visc, Lc, D, n_points = 21):
        self.visc = visc
        self.gas = gas
        self.Lc = Lc
        self.D = D
        self.e_D = 1.5 * 1e-6 #m
        self.k_solo = 0.89 # w / (m*K)
        self.T_solo = 15 + 273.15# C
        self.z_solo = 2 # m
        # --- Malha de Chebyshev-Gauss ---
        i = np.arange(1, n_points + 1)
        xi = np.cos((2 * i - 1) / (2 * n_points) * np.pi)  # [-1,1]
        self.l = (self.Lc / (xi[-1] - xi[0])) * (xi - xi[0])  # [0, Lc]
        self.n_points = n_points

    def fator_friccao(self, Re):
        
        # Zigrang & Sylvester (1982)
        f = 4 * (-4 * np.log10((self.e_D / (3.7 * self.D))- (5.02 / Re) * np.log10((self.e_D / (3.7 * self.D))
                    - (5.02 / Re) * np.log10(
                    (self.e_D / (3.7 * self.D)) + (13.0 / Re)
                    )
                )
            )
        ) ** (-2)
        
        return float(f)
    
    def q_solo(self, Rho, T, U): 
        #Chaczykowski(2010)
        return float((1/Rho) * (4*U/ self. D) * (self.T_solo - T ))

    def coef_conv_fluid(self, kappa, mu, Re, gas):
        # Número de Prandtl
        P_r = (gas.Cpt * 1000 / gas.mixture.MM_m * mu) / kappa  # Cp [J/mol·K] -> [J/kg·K] via MM_m se necessário

        # Fator de atrito de Petukhov (sem usar ff da Colebrook)
        ft = (1.82 * np.log10(Re) - 1.64) ** (-2)

        # Número de Nusselt
        Nu = (ft / 8) * (Re - 1000) * P_r / (1.07 + 12.7 * ((ft / 8) ** 0.5) * (P_r ** (2 / 3) - 1))

        # Coeficiente convectivo
        h_t = Nu * kappa / self.D

        return float(h_t)

    def derivada_centrada(self, x, f, i):

        h = x[1] - x[0]  
    
        if i == 0:
            return (-3*f[0] + 4*f[1] - f[2]) / (x[2] - x[0])
        elif i == len(x) - 1:
            return (3*f[-1] - 4*f[-2] + f[-3]) / (x[-1] - x[-3])
        else:
            return (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
    
    def estacionario(self, x, y):
        T, V, w = map(float, y) 

        gas2 = self.gas.copy_change_conditions(T, None, V, 'gas')
        v_kg = float(V / gas2.mixture.MM_m)
        gas2.ci_real()

        Cv = float(np.squeeze(gas2.Cvt / gas2.mixture.MM_m * 1000))
        
        mu = self.visc.evaluate_viscosity(T, float(gas2.P))
        rho = 1 / v_kg
        Re = rho * w * (self.D / mu)

        f = self.fator_friccao(Re)
        kappa = float(coef_con_ter(gas2))
        h_t = self.coef_conv_fluid(kappa, mu, Re, gas2)
        U = (1.0 / ((1.0 / h_t) + (self.D / (2 * self.k_solo)) * (np.arccosh(2 * self.z_solo / self.D))))
        q = self.q_solo(rho, T, U)
        dPdT = float(gas2.dPdT) * 1000
        dPdV = float(gas2.dPdV) * 1000


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

        return [dTdx, dVdx, dwdx]
