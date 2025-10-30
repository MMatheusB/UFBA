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
    def __init__(self, gas, visc, compressor, Lc, D, n_points=21):
        self.visc = visc
        self.gas = gas
        self.compressor = compressor
        self.Lc = Lc
        self.D = D
        self.e_D = 1.5 * 1e-6  # rugosidade [m]
        self.k_solo = 0.89  # condutividade térmica do solo [W/(m*K)]
        self.T_solo = 15 + 273.15  # temperatura do solo [K]
        self.z_solo = 2  # profundidade [m]
        
        # --- Malha de Chebyshev-Gauss ---
        i = np.arange(1, n_points + 1)
        xi = np.cos((2 * i - 1) / (2 * n_points) * np.pi)  # [-1,1]
        self.l = (self.Lc / (xi[-1] - xi[0])) * (xi - xi[0])  # [0, Lc]
        self.n_points = n_points

    # ---------------------
    # Propriedades auxiliares
    # ---------------------
    def fator_friccao(self, Re): 
        return 0.25 * float(-4 * np.log10(self.e_D / 3.7 / self.D - 5.02 / Re * np.log10(self.e_D / 3.7 / self.D - 5.02 / Re * np.log10(self.e_D / 3.7 / self.D + 13 / Re)))) ** (-2)
    
    def q_solo(self, Rho, T, U): 
        return float((1 / Rho) * (4 * U / self.D) * (self.T_solo - T))

    def coef_cov_fluid(self, kappa, mu, Re, gas):
        P_r = (gas.Cpt * 1000 / gas.mixture.MM_m * mu) / kappa
        ft = (1.82 * np.log10(Re) - 1.64) ** (-2)
        Nu = (ft / 8) * (Re - 1000) * P_r / (1.07 + 12.7 * ((ft / 8) ** 0.5) * (P_r ** (2 / 3) - 1))
        h_t = Nu * kappa / self.D
        return float(h_t)

    def derivada_centrada(self, x, f, i):
        """
        Derivada de f(x) no ponto x[i] via polinômio de Lagrange de 3 pontos.
        """
        n = len(x)
        if i == 0:
            idx = [0, 1, 2]
        elif i == n - 1:
            idx = [n - 3, n - 2, n - 1]
        else:
            idx = [i - 1, i, i + 1]

        x_sub = np.array([x[j] for j in idx], dtype=float)
        f_sub = np.array([f[j] for j in idx], dtype=float)
        xi = x[i]

        deriv = 0.0
        for j in range(3):
            Lj_deriv = 0.0
            for m in range(3):
                if m != j:
                    prod = 1.0
                    for k in range(3):
                        if k != j and k != m:
                            prod *= (xi - x_sub[k]) / (x_sub[j] - x_sub[k])
                    Lj_deriv += prod / (x_sub[j] - x_sub[m])
            deriv += f_sub[j] * Lj_deriv

        return float(deriv)

    # ---------------------
    # Sistema dinâmico
    # ---------------------
    def evaluate_dae(self, t, y):
        T = y[0::3]
        V = y[1::3]
        w = y[2::3]

        dTdt = np.zeros_like(T)
        dVdt = np.zeros_like(V)
        dwdt = np.zeros_like(w)

        for i in range(len(self.l)):
            gas2 = self.gas.copy_change_conditions(T[i], None, V[i], 'gas')
            v_kg = V[i] / gas2.mixture.MM_m
            rho = 1 / v_kg
            gas2.ci_real()
            Cv = float(gas2.Cvt) / gas2.mixture.MM_m * 1000
            
            mu = self.visc.evaluate_viscosity(T[i], gas2.P.item())
            Re = rho * w[i] * (self.D / mu)
            f = self.fator_friccao(Re)
            kappa = coef_con_ter(gas2)
            h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)
            U = 1 / ((1 / h_t) + (self.D / (2 * self.k_solo)) * np.arccosh(2 * self.z_solo / self.D))
            q = self.q_solo(rho, T[i], U)

            dPdT = float(gas2.dPdT) * 1000
            dPdV = float(gas2.dPdV) * 1000

            dT_dx = self.derivada_centrada(self.l, T, i)
            dV_dx = self.derivada_centrada(self.l, V, i)
            dw_dx = self.derivada_centrada(self.l, w, i)

            matrix_dx = np.array([[dT_dx], [dV_dx], [dw_dx]])
            matrix_a = np.array([
                [-w[i], 0.0, -T[i] * (v_kg * dPdT / Cv)],
                [0.0, -w[i], V[i]],
                [-v_kg * dPdT, -v_kg * dPdV, -w[i]]
            ])
            matrix_b = np.array([
                [f * w[i]**2 * abs(w[i]) / (2 * self.D * Cv) + q / Cv],
                [0.0],
                [-f * w[i] * abs(w[i]) / (2 * self.D)]
            ])
            result = matrix_a @ matrix_dx + matrix_b

            dTdt[i] = result[0]
            dVdt[i] = result[1]
            dwdt[i] = result[2]

            if i == 0:
                dTdt[i] = 0
                dVdt[i] = 0
            elif i == len(self.l) - 1:
                dwdt[i] = 0

        dydt = np.empty_like(y)
        dydt[0::3] = dTdt
        dydt[1::3] = dVdt
        dydt[2::3] = dwdt
        return dydt

    # ---------------------
    # Estacionário (com integração do compressor)
    # ---------------------
    def estacionario(self, x, y, compressor =None):
        """
        Calcula o regime estacionário do duto.
        Se o compressor for passado, usa sua saída (T2, V2) como entrada do duto.
        """
        # --- Condições de contorno ---

        T_inicial, V_inicial = y[0], y[1]

        # --- Calcula derivadas no ponto x ---
        T, V, w = map(float, y)
        T = T_inicial
        V = V_inicial

        gas2 = self.gas.copy_change_conditions(T, None, V, 'gas')
        v_kg = float(V / gas2.mixture.MM_m)
        gas2.ci_real()

        Cv = float(gas2.Cvt) / gas2.mixture.MM_m * 1000
        mu = float(self.visc.evaluate_viscosity(T, float(gas2.P)))
        rho = 1 / v_kg
        Re = rho * w * (self.D / mu)
        f = self.fator_friccao(Re)
        kappa = float(coef_con_ter(gas2))

        h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)
        U = 1.0 / ((1.0 / h_t) + (self.D / (2 * self.k_solo)) * (np.arccosh(2 * self.z_solo / self.D)))
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

        return dTdx, dVdx, dwdx
