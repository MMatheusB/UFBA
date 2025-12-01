from casadi import MX, vertcat, cos, pi, log10, sqrt, inv, arccosh, fabs
import numpy as np
from libs.gc_eos_soave import *
from libs.viscosity import *
from libs.eos_database import *
from libs.composicaogas import *


class duto_casadi:
    def __init__(self, gas, visc, Lc, compressor, D, n_points=21):
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
        eD = self.e_D / self.D
        expr = -4 * log10(eD / 3.7 - 5.02 / Re * log10(eD / 3.7 - 5.02 / Re * log10(eD / 3.7 + 13 / Re)))
        return 4 * (expr)**(-2)

    def q_solo(self, Rho, T, U):
        return (1 / Rho) * (4 * U / self.D) * (self.T_solo - T)

    def coef_cov_fluid(self, kappa, mu, Re, gas):
        gas.ci_real()
        Pr = mu * (gas.Cpt / gas.mixture.MM_m) / kappa
        ft = (1.82 * log10(Re) - 1.64)**(-2)
        Nu = (ft / 8) * (Re - 1000) * Pr / (1 + 12.7 * sqrt(ft / 8) * (Pr**(2/3) - 1))
        h_t = Nu * kappa / self.D
        return h_t

    def T_ML(self, T2, Tq, Taq, T_af):
        a = T2 - Taq
        b = Tq - T_af
        c = np.log(a/b)
        return (a - b)/c

    def derivada_centrada(self, x, f, i):
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

    def evaluate_alg_duto2(self, T_in, V_in, w_in, w_aspas, V_aspas, T_aspas):
        A = ca.pi * (self.D**2) / 4
        MM = self.gas.mixture.MM_m  # massa molar em kg/mol
        v_kg = V_in / MM
        rho = 1 / v_kg
        m_dot_final = rho * A * w_in
        v_kg_aspas = V_aspas / MM
        rho_aspas = 1 / v_kg_aspas
        m_dot_aspas = rho_aspas * A * w_aspas
        ag1 = m_dot_final * (self.gas.h - self.gas.h) + T_in * self.gas.dPdT * V_in* (m_dot_final - m_dot_aspas)
        gas_in = self.gas.copy_change_conditions(T_in, None, V_in, 'gas')
        gas_aspas = self.gas.copy_change_conditions(T_aspas, None, V_aspas, 'gas')
        ag2 = gas_in.P - gas_aspas.P
        ag3 = m_dot_final - m_dot_aspas
        
        return  ag1, ag2, ag3

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
        T_q = z[9]
        T_aq = z[10]
        V_q = z[11]
        m_dot_inicio = z[12]
        P_final = z[13]

        rot = u[0]
        m_dot_agua = u[1]
        P1 = u[2]
        T1 = u[3]
        Q_final = u[4]
        T_af = u[5]

        A = np.pi * (self.D**2) / 4

        MM = self.gas.mixture.MM_m  # massa molar em kg/mol
        v_kg = V[0] / MM
        rho = 1 / v_kg
        m_dot = rho * A * w[0]
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = self.compressor.character_dae(
            [Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1],
            [rot, (m_dot)/4, P1, T1]   
        )
        gas_T2 = self.gas.copy_change_conditions(T2, None, V2, 'gas')
        mu_T2 = self.visc.evaluate_viscosity(T2, gas_T2.P)
        Re_T2 = rho * w[0] * self.D / mu_T2
        kappa_T2 = coef_con_ter(gas_T2)
        h_t_T2 = self.coef_cov_fluid(kappa_T2, mu_T2, Re_T2, gas_T2)
       
        gas_Tq = self.gas.copy_change_conditions(T_q, None, V_q, 'gas')
        mu_Tq = self.visc.evaluate_viscosity(T_q, gas_Tq.P)
        Re_Tq = rho * w[0] * self.D / mu_Tq
        kappa_Tq = coef_con_ter(gas_Tq)
        h_t_Tq = self.coef_cov_fluid(kappa_Tq, mu_Tq, Re_Tq, gas_Tq)
       
        a10 = m_dot*(h_t_T2 - h_t_Tq) - 300*self.T_ML(T2, T_q, T_aq, T_af)

        a11 = m_dot_agua*4184 - 300*self.T_ML(T2, T_q, T_aq, T_af)

        a12 = gas_T2.P - gas_Tq.P

        a13 = m_dot_inicio - m_dot
        
        gas_temp = self.gas.copy_change_conditions(T[-1], None, V[-1], 'gas')
        
        a14 = P_final - gas_temp.P

        w_final = Q_final/A
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
                dwdt[i] = (w_final - w[i])

        # Concatenar tudo
        dydt = []
        for i in range(self.n_points):
            dydt += [dTdt[i], dVdt[i], dwdt[i]]
        
        alg = vertcat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)

        return vertcat(*dydt), alg
    
    def evaluate_dae_duto2(self, t, y, z, u):
        T = [y[3*i + 0] for i in range(self.n_points * 2)]
        V = [y[3*i + 1] for i in range(self.n_points * 2)]
        w = [y[3*i + 2] for i in range(self.n_points * 2)]

        Timp_1 = z[0]   # Temperatura na entrada do compressor
        Vimp_1 = z[1]   # Vazão na entrada do compressor
        Tdif_1 = z[2]   # Temperatura na saída do compressor
        Vdif_1 = z[3]   # Vazão na saída do compressor
        T2s_1 = z[4]   
        V2s_1 = z[5]   
        T2_1 = z[6]   
        V2_1 = z[7]   
        V1_1 = z[8]   
        T_aspas = z[9]
        V_aspas = z[10]
        w_aspas = z[11]
        Timp_2 = z[12]   # Temperatura na entrada do compressor
        Vimp_2 = z[13]   # Vazão na entrada do compressor
        Tdif_2 = z[14]   # Temperatura na saída do compressor
        Vdif_2 = z[15]   # Vazão na saída do compressor
        T2s_2 = z[16]   
        V2s_2 = z[17]   
        T2_2 = z[18]   
        V2_2 = z[19]   
        V1_2 = z[20]
        
        rot = u[0] 
        P1 = u[1]
        T1 = u[2]
        P_duto = u[3]
        rot2 = u[4]
        Q_final = u[5]

        A = np.pi * (self.D**2) / 4

        MM = self.gas.mixture.MM_m  # massa molar em kg/mol
        v_kg = V[0] / MM
        rho = 1 / v_kg
        m_dot = rho * A * w[0]
        a3, a4, a5, a6, a7, a8, a9, a10, a11 = self.compressor.character_dae(
            [Timp_1, Vimp_1, Tdif_1, Vdif_1, T2s_1, V2s_1, T2_1, V2_1, V1_1],
            [rot, (m_dot)/4, P1, T1]   
        )
        a12, a13, a14 = self.evaluate_alg(T[-1], V[-1],w[-1], w_aspas, V_aspas, T_aspas)
        w_final = Q_final/A
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
            dT_dx = self.derivada_centrada(self.l, T[:self.n_points+1], i)
            dV_dx = self.derivada_centrada(self.l, V[:self.n_points+1], i)
            dw_dx = self.derivada_centrada(self.l, w[:self.n_points+1], i)

            # Matriz A simbólica
            A = ca.vertcat(
                ca.horzcat(-w[i], 0, -T[i] * (v_kg * dPdT / Cv)),
                ca.horzcat(0, -w[i], V[i]),
                ca.horzcat(-v_kg * dPdT, -v_kg * dPdV, -w[i])
            )

            dx = ca.vertcat(dT_dx, dV_dx, dw_dx)

            b = ca.vertcat(
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
                dTdt[i] = (T2_1 - T[i])
                dVdt[i] = (V2_1 - V[i])
            elif i == self.n_points - 1:
                dwdt[i] = (w_final - w[i])
        
        v_kg = V[self.n_points] / MM
        rho = 1 / v_kg
        m_dot = rho * A * w[self.n_points]
        gas_temp = self.gas.copy_change_conditions(T_aspas, None, V_aspas, 'gas')

        a15, a16, a17, a18, a19, a20, a21, a22, a23 = self.compressor.character_dae(
            [Timp_2, Vimp_2, Tdif_2, Vdif_2, T2s_2, V2s_2, T2_2, V2_2, V1_2],
            [rot2, (m_dot)/4, gas_temp.P, T_aspas]
        )
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
            dT_dx = self.derivada_centrada(self.l, T[self.n_points:], i)
            dV_dx = self.derivada_centrada(self.l, V[self.n_points:], i)
            dw_dx = self.derivada_centrada(self.l, w[self.n_points:], i)

            # Matriz A simbólica
            A = ca.vertcat(
                ca.horzcat(-w[i], 0, -T[i] * (v_kg * dPdT / Cv)),
                ca.horzcat(0, -w[i], V[i]),
                ca.horzcat(-v_kg * dPdT, -v_kg * dPdV, -w[i])
            )

            dx = ca.vertcat(dT_dx, dV_dx, dw_dx)

            b = ca.vertcat(
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
                dTdt[i] = (T2_2 - T[i])
                dVdt[i] = (V2_2 - V[i])
            elif i == self.n_points - 1:
                dwdt[i] = (w_final - w[i])
        # Concatenar tudo
        dydt = []
        for i in range(self.n_points):
            dydt += [dTdt[i], dVdt[i], dwdt[i]]
        
        alg = ca.vertcat(a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23)

        return ca.vertcat(*dydt), alg
    
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
            f * w**2 * fabs(w) / (2 * self.D * Cv) + (q / Cv),
            0,
            -f * w * fabs(w) / (2 * self.D)
        )

        # Sistema linear A * [dTdx, dVdx, dwdx] = -b
        deriv = -inv(A) @ b

        dTdx = deriv[0]
        dVdx = deriv[1]
        dwdx = deriv[2]

        return vertcat(dTdx, dVdx, dwdx)