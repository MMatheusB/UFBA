# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:31:58 2024

@author: rodri
"""
from numpy import array, zeros, log, exp, argmax
from scipy.optimize import fsolve, minimize
from matplotlib.pyplot import plot, figure
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

def config_plot(axes):
    """
    Configures the axes of Figures
    """
    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_scientific(False)
    formatter.set_powerlimits((-1, 1))
    axes.yaxis.set_major_formatter(formatter)

    axes.xaxis.set_minor_locator(AutoMinorLocator())
    axes.yaxis.set_minor_locator(AutoMinorLocator())
    axes.tick_params(which='both', direction='out', bottom=True, left=True)
    axes.tick_params(which='major', width=2)
    axes.tick_params(which='minor', width=1)
    # axes.xaxis.set_tick_params(which='both', right='off', top='off', direction='out', width=1)
    # axes.yaxis.set_tick_params(which='both', right='off', top='off', direction='out', width=1)

    # axes.spines['top'].set_visible(False)

class compression:

    def __init__(self,suction_fluid,compressor,vis_model, compressible = True):

        self.suction_fluid = suction_fluid
        self.compressor = compressor
        self.vis_model = vis_model
        self.compressible = compressible

    def set_suction_fluid(self,new_suction_fluid):

        self.suction_fluid = new_suction_fluid

    def characterization(self, N, divi, m_max = 100):
        nN = len(N)
        Gi1 = self.suction_fluid.copy_change_conditions(323, 'icog', self.suction_fluid.V*0.8, 'gas')

        Phi = [[] for i in range(nN)]  # Supondo que 100 seja suficiente para a matriz
        eta = [[] for i in range(nN)]
        M = [[] for i in range(nN)]

        self.T = [[] for i in range(nN)]
        self.P = [[] for i in range(nN)]
        self.Mach = [[] for i in range(nN)]

        self.Timp = [[] for i in range(nN)]
        self.Pimp = [[] for i in range(nN)]
        self.Vimp = [[] for i in range(nN)]
        self.Zimp = [[] for i in range(nN)]

        self.Tdif = [[] for i in range(nN)]
        self.Pdif = [[] for i in range(nN)]
        self.Vdif = [[] for i in range(nN)]
        self.Zdif = [[] for i in range(nN)]

        self.PHImax = [[] for i in range(nN)]
        self.PHIid = [[] for i in range(nN)]
        self.PHIfd = [[] for i in range(nN)]
        self.PHIii = [[] for i in range(nN)]
        self.PHIfi = [[] for i in range(nN)]
        self.PHItotal = [[] for i in range(nN)]

        pos = []
        for i in range(nN):
            Mach = 0
            k = 0
            m = 0
            Ethan = 0.5
            print(i)
            if i > 0:
                print(Phi[i])
            while Mach < 1.05 and Ethan > 0.05 and m < m_max:
                Phi_cal, eta_cal, Mach, Gi1, G2, Gdif, PHI = self.character(m, N[i], Gi1)

                Phi[i].append(Phi_cal)
                eta[i].append(eta_cal)

                Ethan = eta[i][k]

                self.Mach[i].append(Mach)
                self.T[i].append(G2.T)
                self.P[i].append(G2.P)
                M[i].append(m)

                self.Timp[i].append(Gi1.T)
                self.Vimp[i].append(Gi1.V)
                self.Pimp[i].append(Gi1.P)
                self.Zimp[i].append(Gi1.Z)

                self.Tdif[i].append(Gdif.T)
                self.Vdif[i].append(Gdif.V)
                self.Pdif[i].append(Gdif.P)
                self.Zdif[i].append(Gdif.Z)

                # self.PHImax[i].append(self.character_phi(m, N[i], 0) * self.suction_fluid.P)
                # self.PHIid[i].append(self.PHImax[i, k] - self.character_phi(m, N[i], PHI[0]) * self.suction_fluid.P)
                # self.PHIfd[i].append(self.PHImax[i, k] - self.character_phi(m, N[i], PHI[1]) * self.suction_fluid.P)
                # self.PHIii[i].append(self.PHImax[i, k] - self.character_phi(m, N[i], PHI[2]) * self.suction_fluid.P)
                # self.PHIfi[i].append(self.PHImax[i, k] - self.character_phi(m, N[i], PHI[3]) * self.suction_fluid.P)
                # self.PHItotal[i].append(self.PHImax[i, k] - self.character_phi(m, N[i], sum(PHI)) * self.suction_fluid.P)

                if Mach < 0.64:
                    m += divi
                else:
                    m += divi / 10

                k += 1

            pos.append(k - 1)
            Gi1 = self.suction_fluid.copy_change_conditions(max(self.Timp[i]), None, min(self.Vimp[i]), 'gas')

        self.Phi = Phi
        self.eta = eta
        self.M = M
        Gi2 = G2
        self.Gs = []
        self.Gw = []

        self.m_pwall = []
        m_surge = []
        Phi_surge = zeros(nN)
        self.Phi_pwall = []
        m_op = zeros(nN)
        Phi_op = zeros(nN)

        for i in range(nN - 1):
            if self.Mach[i][-1] > 0.9:
                self.m_pwall.append(fsolve(lambda m: 1 - self.character_wall(m, N[i], Gi2), M[i][pos[i]])[0])
                phi_wall = self.character_phi([self.m_pwall[-1]], N[i], Gi2)
                self.Phi_pwall.append(phi_wall)

            psur = argmax(Phi[i])
            msu0 = M[i][psur]
            Gi1 = self.suction_fluid.copy_change_conditions(self.Timp[i][psur], None, self.Vimp[i][psur],'gas')

            result = minimize(lambda m: -self.character_phi(m, N[i], Gi1), msu0)
            m_surge.append(result.x[0])
            Phi_surge[i] = self.character_phi([m_surge[i]], N[i], Gi1)
            self.Gs.append(Gi1)

            m_op[i] = m_surge[i] * 1.1
            Phi_op[i] = self.character_phi([m_op[i]], N[i], Gi1)

        i = nN - 1
        pos[i] = len(M[i])-1
        if self.Mach[i][-1] > 0.9:
            self.m_pwall.append(fsolve(lambda m: 1 - self.character_wall(m, N[i], Gi2), M[i][pos[i]])[0])
            phi_wall = self.character_phi([self.m_pwall[-1]], N[i], Gi2)
            self.Phi_pwall.append(phi_wall)

        psur = argmax(Phi[i])
        msu0 = M[i][psur]
        print(psur,msu0)
        Gi1 = self.suction_fluid.copy_change_conditions(self.Timp[i][psur], None, self.Vimp[i][psur],'gas')

        result = minimize(lambda m: -self.character_phi(m, N[i], Gi1), msu0)

        print('era para ser esse', result)
        m_surge.append(result.x[0])
        Phi_surge[i] = self.character_phi([m_surge[i]], N[i], Gi1)
        self.Gs.append(Gi1)

        m_op[i] = m_surge[i] * 1.1
        Phi_op[i] = self.character_phi([m_op[i]], N[i], Gi1)

        self.m_op = m_op
        self.Phi_op = Phi_op
        self.pos = pos
        self.Phi_surge = Phi_surge
        self.m_surge = m_surge

    def character(self, m, N, Gi_1):
        self.compressor.update_speed(N)
        W = self.compressor.Ah_ideal

        # Impeller
        Co1 = 0
        Co2 = self.compressor.Ca2
        Gi = self.suction_fluid

        Ca1 = m * Gi.V / self.suction_fluid.mixture.MM_m / self.compressor.A1

        C0 = (Ca1**2 + Co2**2)**0.5
        Yo = array([self.suction_fluid.T, self.suction_fluid.V, C0/2])

        var = array([Gi_1.T, Gi_1.V, C0]) / Yo
        l = self.compressor.li
        Ahii = self.compressor.losses_incimp(m, self.suction_fluid)

        if self.compressible:
            var = fsolve(lambda var: self.imp_dif(var, m, W, l, Gi, Ahii, Co1, Co2, C0), var)
            var = array(var) * Yo
            Timp, Vimp, Cimp = var
            Gimp = self.suction_fluid.copy_change_conditions(Timp, None, Vimp, 'gas')
        else:
            Gimp = self.suction_fluid

        Ahfi = self.compressor.perdas_fricimp(Gimp,self.vis_model, m)


        # Diffuser
        Co1 = self.compressor.Ca2
        Co2 = self.compressor.Ca2
        Gi = Gimp

        l = self.compressor.ld
        Ahid = self.compressor.losses_incdif(m, Gimp)

        if self.compressible:
            var = array([Timp*1.5, Vimp*0.8, var[2]]) / Yo
            var = fsolve(lambda var: self.imp_dif(var, m, 0, l, Gi, Ahid, Co1, Co2, C0), var)

            var = array(var) * Yo

            Tdif, Vdif, Cdifs = var
            Gdif = self.suction_fluid.copy_change_conditions(Tdif, None, Vdif, 'gas')
        else:
            Gdif = self.suction_fluid

        Ahfd = self.compressor.perdas_fricdif(Gdif, self.vis_model, m)
        Gi_2 = Gdif

        # Pressure Ratio
        eta = (W - Ahid - Ahfd - Ahii - Ahfi) / W - 0.065

        self.suction_fluid.ci_real()
        k = self.suction_fluid.Cpt / self.suction_fluid.Cvt
        R = 8.314472
        PMt = self.suction_fluid.mixture.MM_m
        P2 = self.suction_fluid.P * (1 + eta * W / 1000 * PMt / self.suction_fluid.Cpt / self.suction_fluid.T) ** (k / (k - 1))
        V2s = self.suction_fluid.V * (self.suction_fluid.P / P2) ** (1 / k)
        G2s = self.suction_fluid.copy_change_conditions(None,P2,V2s,'gas')
        T2s = G2s.T
        T2 = T2s + (1 - eta) * W / 1000 * PMt / self.suction_fluid.Cpt
        V2 = T2 * R / P2

        G2s = self.suction_fluid.copy_change_conditions(92+273.15,46000,None,'gas')

        var = [G2s.T, G2s.V, G2s.T, G2s.V]

        var = fsolve(lambda var: self.thermal(var, W / 1000 * PMt, eta), var)

        T2, V2, T2s, V2s = var
        C2 = Ca1
        C2s = Ca1
        G2 = self.suction_fluid.copy_change_conditions(T2, None, V2, 'gas')
        G2s = self.suction_fluid.copy_change_conditions(T2s, None, V2s, 'gas')
        Phi = G2.P / self.suction_fluid.P

        # Mach Number in diffuser
        Cdif = (Co2**2 + (m * Gdif.V / PMt / self.compressor.A1) ** 2) ** 0.5
        Mach = Cdif / Gdif.sound_speed()

        PHI = [Ahid, Ahfd, Ahii, Ahfi]

        return Phi, eta, Mach, Gimp, G2, Gdif, PHI #, G2s, Cimp, Cdifs, C2, C2s

    def character_dae(self,z,u):

        Timp, Vimp, Tdif, Vdif, T2s, V2s, T2, V2, V1 = z

        N, m, P1, T1 = u

        G1 = self.suction_fluid.copy_change_conditions(T1, None, V1, 'gas')

        Gimp = self.suction_fluid.copy_change_conditions(Timp, None, Vimp, 'gas')

        Gdif = self.suction_fluid.copy_change_conditions(Tdif, None, Vdif, 'gas')

        G2s = self.suction_fluid.copy_change_conditions(T2s, None, V2s, 'gas')

        G2 = self.suction_fluid.copy_change_conditions(T2, None, V2, 'gas')

        self.compressor.update_speed(N)

        W = self.compressor.Ah_ideal

        # Impeller
        Co1 = 0
        Co2 = self.compressor.Ca2

        Gi = G1

        Ca1 = m * Gi.V / self.suction_fluid.mixture.MM_m / self.compressor.A1

        C0 = (Ca1**2 + Co2**2)**0.5
        Yo = array([self.suction_fluid.T, self.suction_fluid.V, C0/2])

        l = self.compressor.li

        Ahii = self.compressor.losses_incimp(m, Gi)

        var = [Timp/Yo[0], Vimp/Yo[1], C0/Yo[2]]

        a1, a2 = self.imp_dif_dae(var, m, W, l, Gi, Ahii, Co1, Co2, C0)

        Ahfi = self.compressor.perdas_fricimp(Gimp,self.vis_model, m)

        # Diffuser
        Co1, Co2 = [self.compressor.Ca2, self.compressor.Ca2]

        Gi = Gimp

        l = self.compressor.ld
        Ahid = self.compressor.losses_incdif(m, Gimp)

        var = var = [Tdif/Yo[0], Vdif/Yo[1], C0/Yo[2]]

        a3, a4 = self.imp_dif_dae(var, m, 0, l, Gi, Ahid, Co1, Co2, C0)

        Ahfd = self.compressor.perdas_fricdif(Gdif, self.vis_model, m)

        # Pressure Ratio
        eta = (W - Ahid - Ahfd - Ahii - Ahfi) / W - 0.065

        var = [T2s, V2s, T2, V2]

        a5, a6, a7, a8 = self.thermal_dae(var, W / 1000 * self.suction_fluid.mixture.MM_m, eta, G1)

        a9 = P1 - G1.P

        return a1, a2, a3, a4, a5, a6, a7, a8, a9



    def character_phi(self, m, N, Gi_1):

        tuple_phi = self.character(m[0], N, Gi_1)

        return tuple_phi[0]

    def character_wall(self, m, N, Gi_1):

        tuple_wall = self.character(m[0], N, Gi_1)

        return tuple_wall[2]


    def imp_dif(self,var,m,W,l,Gi,Ahi,Co1,Co2,C0):

        T = var[0]*self.suction_fluid.T
        V = var[1]*self.suction_fluid.V
        C2 = var[2]*C0/2;

        rho0 = Gi.mixture.MM_m/Gi.V

        G = Gi.copy_change_conditions(T,None,V,'gas')

        rho = G.mixture.MM_m/G.V

        Gi.h_gas()
        G.h_gas()

        SrhodP = G.int_rhodP(Gi)*1000*G.mixture.MM_m

        Ca1 = m/rho/self.compressor.A1
        Ca2 = m/rho0/self.compressor.A1

        C1 = (Ca1**2 + Co1**2)**0.5

        Ahf = self.compressor.friction(Gi,self.vis_model,l,m,1)

        loss = (Ahi + Ahf)*rho**2

        Aht = (G.h - Gi.h)*1000/G.mixture.MM_m

        a1 = ((C2**2 - C1**2)/2 + Aht - W)/C0**2
        a2 = ((Ca2*rho)**2*log(V/Gi.V) + (Co2**2 - Co1**2)/2*rho**2 +
              SrhodP + loss - W*rho**2)/C0**2/rho0**2
        a3 = (C2**2 - Ca2**2 - Co2**2)/C0**2

        return array([a1,a2,a3])

    def imp_dif_dae(self,var,m,W,l,Gi,Ahi,Co1,Co2,C0):

        T = var[0]*self.suction_fluid.T
        V = var[1]*self.suction_fluid.V

        rho0 = Gi.mixture.MM_m/Gi.V

        G = Gi.copy_change_conditions(T,None,V,'gas')

        rho = G.mixture.MM_m/G.V

        Gi.h_gas()
        G.h_gas()

        SrhodP = G.int_rhodP(Gi)*1000*G.mixture.MM_m

        Ca1 = m/rho/self.compressor.A1
        Ca2 = m/rho0/self.compressor.A1

        C1 = (Ca1**2 + Co1**2)**0.5

        Ahf = self.compressor.friction(Gi,self.vis_model,l,m,1)

        loss = (Ahi + Ahf)*rho**2

        Aht = (G.h - Gi.h)*1000/G.mixture.MM_m

        a1 = ((Ca2**2 + Co2**2 - C1**2)/2 + Aht - W)/C0**2
        a2 = ((Ca2*rho)**2*log(V/Gi.V) + (Co2**2 - Co1**2)/2*rho**2 +
              SrhodP + loss - W*rho**2)/C0**2/rho0**2

        return array([a1,a2])

    def thermal(self,var, W, eta):

        T2, V2, T2s, V2s = var

        G2 = self.suction_fluid.copy_change_conditions(T2, None, V2, 'gas')
        G2s = self.suction_fluid.copy_change_conditions(T2s, None, V2s, 'gas')

        self.suction_fluid.h_gas()
        self.suction_fluid.s_gas()

        G2.h_gas()
        G2.s_gas()

        G2s.h_gas()
        G2s.s_gas()

        a1 = G2s.h - self.suction_fluid.h - W*eta
        a2 = G2s.s - self.suction_fluid.s
        a3 = G2.h - G2s.h - W*(1-eta)
        a4 = G2.P - G2s.P

        return array([a1,a2,a3,a4])

    def thermal_dae(self,var, W, eta, G1):

        T2, V2, T2s, V2s = var

        G2 = self.suction_fluid.copy_change_conditions(T2, None, V2, 'gas')
        G2s = self.suction_fluid.copy_change_conditions(T2s, None, V2s, 'gas')

        G1.h_gas()
        G1.s_gas()

        G2.h_gas()
        G2.s_gas()

        G2s.h_gas()
        G2s.s_gas()

        a1 = G2s.h - G1.h - W*eta
        a2 = G2s.s - G1.s
        a3 = G2.h - G2s.h - W*(1-eta)
        a4 = G2.P - G2s.P

        return array([a1,a2,a3,a4])

    def plot_map(self):

        fig1 = figure(dpi = 150)
        ax = fig1.add_subplot(1, 1, 1)
        for i in range(len(self.M)):
            p1, = plot(self.M[i], self.Phi[i], 'k')
        p2, = plot(self.m_op,self.Phi_op,'o-')


    def plot_map_q(self):

        fig1 = figure(dpi = 150)
        ax = fig1.add_subplot(1, 1, 1)
        cte = self.compressor.A1/self.suction_fluid.mass_rho
        for i in range(len(self.M)):
            p1, = plot([m*cte*3600 for m in self.M[i]], self.Phi[i], 'k')
        p2, = plot([m*cte*3600 for m in self.m_op],self.Phi_op,'o-')


    def plot_map_multiples(self,list_of_maps,colors):

        fig1 = figure(dpi = 150)
        ax = fig1.add_subplot(1, 1, 1)
        cte = 1/self.suction_fluid.mass_rho
        for i,map in enumerate(list_of_maps):
            for j in range(len(map.M)):
                p1, = plot([m*cte*3600 for m in map.M[j]], map.Phi[j], colors[i]+'-')
            p2, = plot([m*cte*3600 for m in map.m_surge],map.Phi_surge,colors[i]+'o-')
        config_plot(ax)
        ax.set_ylabel('Pressure ratio', fontsize=12)
        ax.set_xlabel('Volumetric flow rate /(m$^3$ $\cdot$ s$^{-1}$)', fontsize=12)
        fig1.tight_layout()
        fig1.show()


    def plot_internal(self):
        pmt = self.suction_fluid.mixture.MM_m
        fig1 = figure(dpi = 150)
        ax = fig1.add_subplot(1, 1, 1)
        cte = 1/self.suction_fluid.mass_rho
        for i in range(len(self.M)):
            p1, = plot([cte*3600*m for m in self.M[i]],
                       [pmt/V for V in self.Vimp[i]], 'r--')
            p2, = plot([cte*3600*m for m in self.M[i]],
                       [pmt/V for V in self.Vdif[i]], 'b--')
        p2, = plot([0,cte*3600*self.M[-1][-1]],
                   [pmt/self.suction_fluid.V,pmt/self.suction_fluid.V], 'k--')
        config_plot(ax)
        #ax.set_ylim(bottom=380)
        ax.set_ylabel('Specific mass /(kg $\cdot$ m$^{-3}$)', fontsize=12)
        ax.set_xlabel('Volumetric flow rate /(m$^3$ $\cdot$ h$^{-1}$)', fontsize=12)
        fig1.tight_layout()
        fig1.show()























