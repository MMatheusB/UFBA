from libs.gc_eos_soave import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from libs.viscosity import *
from libs.eos_database import *


class duto:
    def _init_(self, gas, Lc, A1, D, U, T_solo):
        self.gas = gas
        self.Lc = Lc
        self.A1 = A1
        self.D = D
        self.U = U
        self.T_solo = T_solo
    
    def fator_friccao(self, Re): 
        e_D = self.epsilon / self.D
        return (1/ (-4 * np.log((e_D / 3.7) - (5.02/ Re) * (np.log((e_D/ 3.7) - (5.02/Re) * (np.log((e_D / 3.7) + (13/Re))))))))**2
    
    def q_solo(self, Rho, T): 
        return (1/Rho) * (4*self.U/ self. D) * (T - self.T_solo)

    def evaluate_dae(self, t, x, z, u):
        """
        Aqui eu devo calcular as variaveis diferenciais, sendo elas,
        Temperatura, Volume especifico, velocidade.
        """
        T, V, w = x
        gas2 = self.gas.copy_change_conditions(T, None, V, 'gas') # calculando o P via PVT.
        rho = 1 / V
        
        Cv = self.gas.cvT #cv_real ou seria o ci_real? perguntar.
        
        mu = viscosity.evaluate_viscosity(T, gas2.P.item()) #viscosidate? para calcular o numero de reynolds.

        Re = rho * w * (self.D / mu) #numero de reynolds, verificar se a equacao ta certa.

        f = self.fator_friccao(Re) #fator de friccao.

        q = self.q_solo(rho, T) #calor.

        dT_dt = f * w**2 * w / (2 * self.D * Cv) + q / Cv #Temperatura e coisas.
        dV_dt = 0 # Nao sei
        dw_dt = f * w * w / (2 * self.D) # Velocidade, utilizar para calcular a vazao no futuro.

        return np.array([dT_dt, dV_dt, dw_dt])
