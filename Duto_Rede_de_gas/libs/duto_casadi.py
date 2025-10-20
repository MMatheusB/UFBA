from casadi import MX, vertcat, cos, pi, log10, sqrt, inv, arccosh, fabs
import numpy as np
from libs.gc_eos_soave import *
from libs.viscosity import *
from libs.eos_database import *
from libs.composicaogas import *


class duto_casadi:
    def __init__(self, gas, visc, Lc, D, n_points=21):
        self.visc = visc
        self.gas = gas
        self.Lc = Lc
        self.D = D
        self.e_D = 1.5 * 1e-6  # m
        self.k_solo = 0.89  # w / (m*K)
        self.T_solo = 15 + 273.15  # K
        self.z_solo = 2  # m

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
        return 0.25 * (expr)**(-2)

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

    def evaluate_dae(self, t, y):

        # Separar variáveis
        T = [y[3*i + 0] for i in range(self.n_points)]
        V = [y[3*i + 1] for i in range(self.n_points)]
        w = [y[3*i + 2] for i in range(self.n_points)]

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
                dTdt[i] = SX(0)
                dVdt[i] = SX(0)
            elif i == self.n_points - 1:
                dwdt[i] = SX(0)

        # Concatenar tudo
        dydt = []
        for i in range(self.n_points):
            dydt += [dTdt[i], dVdt[i], dwdt[i]]

        return vertcat(*dydt)

