from casadi import MX, vertcat, cos, pi, log10, sqrt, inv, arccosh, fabs
import numpy as np
from libs.gc_eos_soave import *
from libs.viscosity import *
from libs.eos_database import *
from libs.composicaogas import *


class duto_casadi:
    def __init__(self, gas, visc, Lc, compressor, D, n_points=15):
        self.visc = visc
        self.gas = gas
        self.Lc = Lc
        self.D = D
        self.e_D = 1.5 * 1e-6  # m
        self.k_solo = 0.89  # w / (m*K)
        self.T_solo = 15 + 273.15  # K
        self.z_solo = 2  # m
        self.compressor = compressor
        # --- Malha de Chebyshev-Gauss ---
        i = np.arange(1, n_points + 1)
        xi = np.cos((2 * i - 1) / (2 * n_points) * np.pi)  # [-1,1]
        self.l = (self.Lc / (xi[-1] - xi[0])) * (xi - xi[0])  # [0, Lc]
        self.n_points = n_points

    def fator_friccao(self, Re):
        """
        Implementação simbólica do fator de fricção usando CasADi.
        """
        eD = self.e_D / self.D
        expr = -4 * log10(eD / 3.7 - 5.02 / Re * log10(eD / 3.7 - 5.02 / Re * log10(eD / 3.7 + 13 / Re)))
        return 4 * (expr)**(-2)

    def q_solo(self, Rho, T, U):
        return (1 / Rho) * (4 * U / self.D) * (self.T_solo - T)

    def coef_cov_fluid(self, kappa, mu, Re, gas):
        Pr = (gas.Cpt * 1000 / gas.mixture.MM_m * mu) / kappa
        ft = (1.82 * log10(Re) - 1.64)**(-2)
        Nu = (ft / 8) * (Re - 1000) * Pr / (1.07 + 12.7 * sqrt(ft / 8) * (Pr**(2/3) - 1))
        h_t = Nu * kappa / self.D
        return h_t

    def derivada_centrada(self, x, f, i):
        """
        Adaptado para CasADi: f é uma lista de MX e x é numpy.
        """
        n = len(x)
        if i == 0:
            idx = [0, 1, 2]
        elif i == n - 1:
            idx = [n - 3, n - 2, n - 1]
        else:
            idx = [i - 1, i, i + 1]

        x_sub = [x[j] for j in idx]
        f_sub = [f[j] for j in idx]
        xi = x[i]

        deriv = SX(0)
        for j in range(3):
            Lj_deriv = 0
            for m in range(3):
                if m != j:
                    prod = 1
                    for k in range(3):
                        if k != j and k != m:
                            prod *= (xi - x_sub[k]) / (x_sub[j] - x_sub[k])
                    Lj_deriv += prod / (x_sub[j] - x_sub[m])
            deriv += f_sub[j] * Lj_deriv
        return deriv

    def evaluate_dae(self, t, y, z, u):
        T = [y[3*i + 0] for i in range(self.n_points)]
        V = [y[3*i + 1] for i in range(self.n_points)]
        w = [y[3*i + 2] for i in range(self.n_points)]

        Timp = z[0]   # Temperatura na entrada do compressor
        Vimp = z[1]   # Vazão na entrada do compressor
        Tdif = z[2]   # Temperatura na saída do compressor
        Vdif = z[3]   # Vazão na saída do compressor
        T2s  = z[4]   
        V2s  = z[5]   
        T2   = z[6]   
        V2   = z[7]   
        V1   = z[8]   
        
        rot = u[0] 
        P1 = u[1]
        T1 = u[2]
        A = np.pi * (self.D**2) / 4

        MM = self.gas.mixture.MM_m  # massa molar em kg/mol
        v_kg = V[0] / MM
        rho = 1 / v_kg
        m_dot = rho * A * w[0]
        a3, a4, a5, a6, a7, a8, a9, a10, a11 = self.compressor.character_dae(
            [Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1],
            [rot, (m_dot)/4, P1, T1]   
        )
        # --- Aplica   essas como condições de contorno do duto ---
        # T[0] = T2   # Temperatura na entrada do duto = Temperatura de saída do compressor
        # V[0] = V2 
        dTdt, dVdt, dwdt = [], [], []

        for i in range(self.n_points):
            # Atualiza propriedades do gás
            gas2 = self.gas.copy_change_conditions(T[i], None, V[i], 'gas')
            v_kg = V[i] / gas2.mixture.MM_m
            rho = 1 / v_kg
            gas2.ci_real()
            Cv = gas2.Cvt / gas2.mixture.MM_m * 1000
            mu = self.visc.evaluate_viscosity(T[i], gas2.P)
            Re = rho * w[i] * self.D / mu
            f = self.fator_friccao(Re)
            kappa = coef_con_ter(gas2)
            h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)
            U = 1 / ((1 / h_t) + (self.D / (2 * self.k_solo)) * arccosh(2 * self.z_solo / self.D))
            q = self.q_solo(rho, T[i], U)
            dPdT = gas2.dPdT * 1000
            dPdV = gas2.dPdV * 1000

            # Derivadas espaciais
            dT_dx = self.derivada_centrada(self.l, T, i)
            dV_dx = self.derivada_centrada(self.l, V, i)
            dw_dx = self.derivada_centrada(self.l, w, i)

            # Matriz A simbólica
            A = vertcat(
                horzcat(-w[i], 0, -T[i] * (v_kg * dPdT / Cv)),
                horzcat(0, -w[i], V[i]),
                horzcat(-v_kg * dPdT, -v_kg * dPdV, -w[i])
            )

            dx = vertcat(dT_dx, dV_dx, dw_dx)

            b = vertcat(
                f * w[i]**2 * fabs(w[i]) / (2 * self.D * Cv) + q / Cv,
                SX(0),
                -f * w[i] * fabs(w[i]) / (2 * self.D)
            )

            result = A @ dx + b

            dTdt.append(result[0])
            dVdt.append(result[1])
            dwdt.append(result[2])

            # Condições de fronteira
            if i == 0:
                dTdt[i] = (T2 - T[i])
                dVdt[i] = (V2 - V[i])
            elif i == self.n_points - 1:
                dwdt[i] = SX(0)

        # Concatenar tudo
        dydt = []
        for i in range(self.n_points):
            dydt += [dTdt[i], dVdt[i], dwdt[i]]
        
        alg = vertcat(a3, a4, a5, a6, a7, a8, a9, a10, a11)

        return vertcat(*dydt), alg
    
    def estacionario(self, x, y):
        """
        Implementação simbólica da EDO estacionária para o duto.
        Entradas:
            x : MX (posição)
            y : MX [T, V, w]
        Saída:
            [dTdx, dVdx, dwdx] como MX
        """
        T, V, w = y[0], y[1], y[2]

        # Copia o gás e atualiza as condições
        gas2 = self.gas.copy_change_conditions(T, None, V, 'gas')
        v_kg = V / gas2.mixture.MM_m
        rho = 1 / v_kg
        gas2.ci_real()

        Cv = gas2.Cvt / gas2.mixture.MM_m * 1000
        mu = self.visc.evaluate_viscosity(T, gas2.P)
        Re = rho * w * self.D / mu

        # Fator de fricção
        f = self.fator_friccao(Re)
        kappa = coef_con_ter(gas2)
        h_t = self.coef_cov_fluid(kappa, mu, Re, gas2)
        U = 1 / ((1 / h_t) + (self.D / (2 * self.k_solo)) * arccosh(2 * self.z_solo / self.D))
        q = self.q_solo(rho, T, U)

        dPdT = gas2.dPdT * 1000
        dPdV = gas2.dPdV * 1000

        # Matriz A e vetor b simbólicos
        A = vertcat(
            horzcat(-w, 0, -T * (v_kg * dPdT / Cv)),
            horzcat(0, -w, V),
            horzcat(-v_kg * dPdT, -v_kg * dPdV, -w)
        )

        b = vertcat(
            f * w**2 * fabs(w) / (2 * self.D * Cv) + q / Cv,
            0,
            -f * w * fabs(w) / (2 * self.D)
        )

        # Sistema linear A * [dTdx, dVdx, dwdx] = -b
        deriv = -inv(A) @ b

        dTdx = deriv[0]
        dVdx = deriv[1]
        dwdx = deriv[2]

        return vertcat(dTdx, dVdx, dwdx)