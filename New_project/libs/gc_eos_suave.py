# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 01:30:58 2024

@author: Rodrigo Meira
"""
from eos_database import *
from casadi import *
from numpy import exp, log, array, roots, zeros, linalg
from math import isnan
from scipy.optimize import fsolve
from species_builder import R, Species, Mixture
from matplotlib.pyplot import plot, figure
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import scipy.linalg as la
from numpy.linalg import svd, det

class gc_eos_class:

    def __init__(self, mixture, T, P, V, u, w, Aij, delta_v, phase):
        self.mixture = mixture
        self.T = T
        self.P = P
        self.V = V
        self.Aij = Aij
        self.phase = phase
        self.csi = min([y.real for y in roots(
            [u*(w+u)-w, 3*(w+u), 3, -1]) if y.imag == 0])
        # self.csi = 0.2774
        self.u = u
        self.w = w
        self.delta_v = delta_v
        self.delta_vm = array(delta_v).dot(array(mixture.x).transpose())

        if [T, P, V].count(None) > 1:
            print("It is necessary 2 values in T, P or V")

        self.soave_parameters()

        if self.T == None:
            if self.phase == 'liquid':
                self.Veos = self.V-self.delta_vm
                def t_solver(Z): return array(self.evaluate_eos_T_liq(Z))
                res = fsolve(t_solver, 0.25)
                Z_eos = res[0]
                self.T = self.P*(self.Veos)/R/Z_eos
                self.Z = self.P*self.V/R/self.T
            elif self.phase == 'gas':
                self.Veos = self.V-self.delta_vm
                def t_solver(Z): return array(self.evaluate_eos_T_gas(Z))
                res = fsolve(t_solver, 0.85)
                Z_eos = res[0]
                self.T = self.P*(self.Veos)/R/Z_eos
                self.Z = self.P*self.V/R/self.T
                
        elif self.P == None:
            self.Veos = self.V-self.delta_vm
            self.P = self.evaluate_eos_P()
            self.Z = self.P*self.V/self.T/R

        elif self.V == None:
            if self.phase == 'liquid':
                self.Zeos = min(
                    [y.real for y in self.evaluate_z_roots() if y.imag == 0 and y.real > 0])
                self.Veos = self.Zeos*self.T*R/self.P
                self.V = self.Veos+self.delta_vm
                self.Z = self.P*self.V/R/self.T
            elif self.phase == 'gas':
                self.Zeos = max(
                    [y.real for y in self.evaluate_z_roots() if y.imag == 0 and y.real > 0])
                self.Veos = self.Zeos*self.T*R/self.P
                self.V = self.Veos+self.delta_vm
                self.Z = self.P*self.V/R/self.T
            # self.Z = res[0]

        #self.critical_parameters()

        self.rho = 1/self.V
        self.mass_rho = self.mixture.MM_m/self.V
        
        
    
    def critical_case_T(self,T0,V0,F,B,eval):
        crit_gas = self.copy_change_conditions(T0[0], None, V0, 'gas')
        
        x = array(crit_gas.mixture.x)
        
        A = crit_gas.Q.dot(x)/crit_gas.a_m
        mat_Q = zeros([len(x), len(x)])
        for i in range(len(x)):
            for j in range(len(x)):
                delta = 0
                if i==j:
                    delta = 1
                par_1 = 1/sum(x)*(delta/x[i]+(B[i]+B[j])*F[1]+B[i]*B[j]*F[1]**2)
                par_2 = (B[i]*B[j]-A[i]*B[j]-A[j]*B[i])*F[6];
                par_3 = crit_gas.a_m/(self.b_m*R*T0*sum(x))*(B[i]*B[j]*F[3]-crit_gas.Q[i][j]/crit_gas.a_m*F[5]+par_2);
                mat_Q[i][j] = (par_1+par_3)*T0/100;
        if eval:
            return mat_Q
        else:
            return array(det(mat_Q))
    
    def critical_case_V(self, T0, V0,eval):
        
        V0 = V0[0]
        
        D1 = self.u/2 + (self.u**2-4*self.w)**0.5/2
        D2 = self.u/2 - (self.u**2-4*self.w)**0.5/2
        crit_gas = self.copy_change_conditions(T0, None, V0, 'gas')
        K = crit_gas.V/self.b_m
        F1 = 1/(K-1)
        F2 = 2/(D1-D2)*(D1/(K+D1)-D2/(K+D2))
        F3 = 1/(D1-D2)*(D1**2/(K+D1)**2-D2**2/(K+D2)**2)
        F4 = 1/(D1-D2)*(D1**3/(K+D1)**3-D2**3/(K+D2)**3)
        F5 = 2/(D1-D2)*log((K+D1)/(K+D2))
        F6 = F2 - F5
        
        F = [0, F1, F2, F3, F4, F5, F6]
        
        B = self.b/self.b_m
        
        t_solver = lambda T: self.critical_case_T(T, V0, F, B, False)
        res = fsolve(t_solver, x0 = T0)
        T0 = res[0]
        x = array(crit_gas.mixture.x)
        crit_gas = self.copy_change_conditions(T0, None, V0, 'gas')
        
        mat_Q = self.critical_case_T(array([T0]), V0, F, B, True)
        
        U, S, Vh = np.linalg.svd(mat_Q, full_matrices=True)
        
        delta_y0 = U[:,-1]
        if abs(min(delta_y0))>max(delta_y0):
            delta_y0 = delta_y0*-1
        #print(U)
        delta_y0 = delta_y0/sum([i**2 for i in delta_y0])**0.5
        
        N_ = sum(delta_y0)
        B_ = sum(B*delta_y0)
        A = crit_gas.Q.dot(x)/crit_gas.a_m
        A_ = sum(A*delta_y0)
        
        a_ = delta_y0.transpose().dot(crit_gas.Q.dot(delta_y0))/crit_gas.a_m
        
        t1 = -sum(delta_y0**3/x**2)+3*N_*(B_*F[1])**2+2*(B_*F[1])**3
        t2 = 3*B_**2*(2*A_-B_)*(F[3]+F[6])-2*B_**3*F[4]-3*B_*a_*F[6]
        
        Cf = t1+crit_gas.a_m/(crit_gas.b_m*R*crit_gas.T)*t2
        Cf = Cf*(self.V-self.b_m)/self.b_m**2

        if eval:
            return T0
        else:
            return Cf
    
    def critical_case(self, T0, V0):
        
        V_solver = lambda V: self.critical_case_V(T0, V, False)
        
        res = fsolve(V_solver, x0 = V0)
        
        Tc = self.critical_case_V(T0, res, True)
        
        Vc = res[0]
        
        self.crit_gas = self.copy_change_conditions(Tc, None, Vc, 'gas')
    

    def evaluate_eos_T_gas(self, Z):

        self.T = self.P*self.V/Z[0]/R
        self.evaluate_par_a()

        beta = self.b_m*self.P/self.T/R

        q = self.a_m/self.b_m/self.T/R

        return Z - 1 - beta + q*beta*(Z-beta)/(Z**2 + 2*Z*beta - beta**2)

    def evaluate_eos_T_liq(self, Z):

        self.T = self.P*(self.Veos)/Z[0]/R
        self.evaluate_par_a()

        beta = self.b_m*self.P/self.T/R

        q = self.a_m/self.b_m/self.T/R

        return Z - beta + (Z**2 + 2*Z*beta - beta**2)*(1 + beta - Z)/q/beta

    def evaluate_eos_P(self):
        ##revisada
        self.evaluate_par_a()

        V_eos = self.V + self.delta_vm

        return self.T*R/(V_eos-self.b_m) - self.a_m/(V_eos**2 + V_eos*self.b_m)
    
    def evaluate_der_eos_P(self):
        #revisada, falta descobrir oq é dadt
        self.evaluate_der_a()
        
        self.dPdT = R/(self.Veos-self.b_m) - 1/(self.Veos**2 + self.Veos*self.b_m)
        
        self.dPdV = -R*self.T/(self.Veos-self.b_m)**2 + \
                    self.a_m * (self.b_m  + 2*self.Veos)/(self.Veos**2 + self.b_m*self.Veos)**2
    
    def evaluate_der_rho(self):
        
        self.evaluate_der_eos_P()
        
        dPdrho = -self.Veos**2/self.mixture.MM_m*self.dPdV
        
        self.drhodP = 1/dPdrho
        
        self.drhodT = -self.dPdT/dPdrho
        
    def evaluate_eos_V_gas(self, Z):

        self.V = self.T*Z*R/self.P + self.delta_vm 
        self.evaluate_par_a()

        beta = self.b_m*self.P/self.T/R
        q = self.a_m/self.b_m/self.T/R

        return Z - 1 - beta + q*beta*(Z-beta)/(Z**2 + 2*Z*beta - beta**2)

    def evaluate_eos_V_liq(self, Z):

        self.V = self.T*Z*R/self.P
        self.evaluate_par_a()

        beta = self.b_m*self.P/self.T/R
        q = self.a_m/self.b_m/self.T/R

        return Z - beta - (Z**2 + 2*Z*beta - beta**2)*(1 + beta - Z)/q/beta

    def soave_parameters(self):

        self.b = [R*sp.Tc*0.07780/sp.Pc for sp in self.mixture.list_of_species]

        self.a_c = [R**2*sp.Tc**2*0.45724 /
                    sp.Pc for sp in self.mixture.list_of_species]

        self.kappa = [sp.kappa for sp in self.mixture.list_of_species]

    def evaluate_a(self):
        self.Tr = [self.T/sp.Tc for sp in self.mixture.list_of_species]
        self.a = [self.a_c[i]*(1+self.kappa[i]*(1-self.Tr[i]**0.5))**2 \
                  for i in range(len(self.Tr))]

    def evaluate_der_a(self):
        
        A_c = array([[(i*j)**0.5 for i in self.a_c] for j in self.a_c])
        
        sqrt_alfa = [(1+self.kappa[i]*(1-self.Tr[i]**0.5)) \
                     for i in range(len(self.mixture.x))]
        
        der_sqrt_a = [-0.5**self.kappa[i]/(self.T*self.mixture.list_Tc[i])**0.5 \
                      for i in range(len(self.kappa))]
        
        der_A = [[sqrt_alfa[i]*der_sqrt_a[j]+sqrt_alfa[j]*der_sqrt_a[i] \
                  for i in range(len(der_sqrt_a))]
                 for j in range(len(der_sqrt_a))]
        
        Q = A_c*array(der_A)*(1-array(self.Aij))
        
        x = array(self.mixture.x)
        
        self.der_a = x.dot(Q.dot(x.transpose()))

    def evaluate_der_sec_a(self):
        
        A_c = array([[(i*j)**0.5 for i in self.a_c] for j in self.a_c])
        ## linha 188 do outro arquivo.
        der_sec_A = [[0.25*self.T**-1.5*((1+self.kappa[i])*self.kappa[j]/self.mixture.list_Tc[j]**0.5 +
                                         (1+self.kappa[j])*self.kappa[i]/self.mixture.list_Tc[i]**0.5) \
                      for i in range(len(self.mixture.x))]
                     for j in range(len(self.mixture.x))]

        Q = A_c*array(der_sec_A)*(1-array(self.Aij))

        x = array(self.mixture.x)

        self.der_sec_a = x.dot(Q.dot(x.transpose()))

    def evaluate_par_a(self):

        self.evaluate_a()

        self.Q = array([[(self.a[i]*self.a[j])**0.5*(1-self.Aij[i][j]) for i in range(len(self.a))]
                        for j in range(len(self.a))])
        x = array(self.mixture.x)
        self.a_m = x.dot(self.Q.dot(x.transpose()))
        self.b_m = sum(array(self.b)*x)

        self.b_bar = array(self.b)

        self.a_bar = 2*self.Q.dot(x.transpose())-self.a_m

    def evaluate_fugacity_coef(self):

        beta = self.b_m*self.P/self.T/R
        q = self.a_m/self.b_m/self.T/R

        ln_phi_c_i = self.b_bar/self.b_m*(self.Zeos-1) - log(self.Zeos-beta) \
            - q*(1 + self.a_bar/self.a_m - array(self.b_bar)/self.b_m) \
            / 2**1.5*log((self.Zeos+beta*(1+2**.5))/(self.Zeos+beta*(1-2**.5)))
        
        return exp(ln_phi_c_i)
    
    def ci_ideal(self):
        
        self.cpi = self.mixture.evaluate_cp_ig(self.T)
        
        self.cvi = self.cpi-R
    
    def ci_real(self):
            
        self.evaluate_der_sec_a()
        
        self.evaluate_der_eos_P()
        
        self.ci_ideal()
        
        # Cv_real = Cv_ideal - T*\int_V^Inf(d2PdT2)dV
        
        self.Cvt = self.cvi + self.T*self.der_sec_a/(2*2**0.5*self.b_m)*log((self.Veos+self.b_m*(1+2**0.5))/(self.Veos+self.b_m*(1-2**0.5)))
        
        # Cp_real = Cv_real - T*(dPdT)^2/dPdV
        
        self.Cpt = self.Cvt - self.T*(self.dPdT**2)/self.dPdV
    
    def h_gas(self):
        #revisado parcialmente
        hi = self.mixture.evaluate_enthalpy_ig(self.T)
        
        self.evaluate_der_a()
        ## linha 291
        hr = self.P*self.Veos - R*self.T - (self.a_m - self.der_a*self.T)*(1/self.b_m)*(np.log(1 + self.b_m/self.Veos))
        self.h = hr + hi;
    
    def s_gas(self):
        #revisado parcialmente
        si = self.mixture.evaluate_entropy_ig(self.T)
        
        self.evaluate_der_a()
        #linha 305
        sr = R*log((self.Veos-self.b_m)*self.P/(R*self.T)) - self.der_a*(1/self.b_m)* np.log(1 + self.b_m/self.Veos)
        
        self.s = sr + si - R*log(self.P)
        
    def sound_speed(self):
        
        self.ci_real()
        
        self.evaluate_der_eos_P()
        
        rhoeos = 1/self.Veos;
        
        c2 = -self.Veos**2*self.dPdV + self.T*self.Veos**2*(self.dPdT)**2/(self.Cvt)

        return (c2*1000/self.mixture.MM_m)**0.5
    
    def evaluate_z_roots(self):
        
        self.evaluate_par_a()

        beta = self.b_m*self.P/self.T/R
        q = self.a_m/self.b_m/self.T/R

        return roots([1, (self.u-1)*beta-1, (q-self.u+(self.w-self.u)*beta)*beta, -(self.w*beta**3+self.w*beta**2+q*beta**2)])
    
    import math

    def int_rhodP(self, g01):
        
        A = -R * self.T / self.b_m**2 * (
            (self.V / (self.b_m - self.V)-g01.V / (self.b_m - g01.V)) - 
            log((self.b_m - self.V)/(self.b_m - g01.V)) + log(self.V/g01.V))
    
        tanh = 0.5 * log((self.V + self.b_m * (1 + 2**0.5)) / (self.V + self.b_m * (1 - 2**0.5)))
        tanhg01 = 0.5 * log((g01.V + self.b_m * (1 + 2**0.5)) / (g01.V + self.b_m * (1 - 2**0.5)))
        
        B = self.a_m * (
            (
                2 * self.b_m * (2 * self.b_m + self.V) / (self.b_m**2 - 2 * self.b_m * self.V - self.V**2)
                - 2 * log(-self.b_m**2 + 2 * self.b_m * self.V + self.V**2)
                + 3 * 2**0.5 * tanh
                + 4 * log(self.V)
            ) / (2 * self.b_m**3)
            - (
                2 * self.b_m * (2 * self.b_m + g01.V) / (self.b_m**2 - 2 * self.b_m * g01.V - g01.V**2)
                - 2 * log(-self.b_m**2 + 2 * self.b_m * g01.V + g01.V**2)
                + 3 * 2**0.5 * tanhg01
                + 4 * log(g01.V)
            ) / (2 * self.b_m**3)
        )
    
        rodV = A + B
    
        A1 = (1 / self.V) * (R / (self.V - self.b_m)) * (self.T - g01.T)
        B1 = -(1 / self.V) * (1 / (self.V * (self.V + self.b_m) + self.b_m * (self.V - self.b_m))) * (self.a_m - g01.a_m)
    
        rodT = A1 + B1
        
        rodP = rodV + rodT
        return rodP

    
    def copy_change_x_and_conditions(self, T, P, V, x, state):

        new_mixture = self.mixture.copy_and_change_composition(x)

        return gc_eos_class(new_mixture, T, P, V, self.u, self.w, self.Aij, self.delta_v, state)

    def copy_change_conditions(self, T, P, V, state):

        return gc_eos_class(self.mixture, T, P, V, self.u, self.w, self.Aij, self.delta_v, state)
    
    def evaluate_pure_fugacity(self):
        
        beta = self.b_m*self.P/self.T/R
        q = self.a_m/self.b_m/self.T/R

        ln_phi = (self.Zeos-1) - log(self.Zeos-beta) \
                - q/2**1.5*log((self.Zeos+beta*(1+2**.5))/(self.Zeos+beta*(1-2**.5)))

        return exp(ln_phi) 

