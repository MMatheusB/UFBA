import numpy as np
import casadi as ca
import time
from scipy.optimize import fsolve
import torch

class Simulation:
    def __init__(self, A1, Lc, kv, P1, P_out, C, alphas, N_RotS, nAlphas, nData, perturb,tempo, dt, interpolation, timestep):
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.alphas = alphas
        self.N_RotS = N_RotS
        self.nAlphas = nAlphas
        self.nData = nData
        self.perturb = perturb
        self.dt = dt
        self.tempo = tempo
        self.timestep = timestep
        
        #Interpolação
        self.interpolation = interpolation
        self.data = None
        self.N_rot = None
        self.Mass = None
        self.Phi = None

        self.interval = [np.linspace(i * self.tempo, (i + 1) * self.tempo, self.nData) for i in range(self.nAlphas)]
        self.time = 0
        
        self.alpha_values = []
        self.N_values = []
        self.massFlowrate = []
        self.PlenumPressure = []
        self.Phi_values = []
        self.RNN_train = []
        self.RNN_trainFut = []
        self.X_train = []
        self.y_train = []        

    def fun(self, variables, alpha, N):
        (x, y) = variables  # x e y são escalares
        phi_value = float(self.interpolation([N, x]))  # Garantir que phi_value é escalar
        eqn_1 = (self.A1 / self.Lc) * ((phi_value * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - alpha * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]


    def run(self):
        lut = self.interpolation
        # Condições iniciais
        result = fsolve(self.fun, (10, 10), args=(self.alphas[0],self.N_RotS[0]))
        init_m, init_p = result

        # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        p = ca.MX.sym('p', 2)  # Parâmetros (alpha e N)
        alpha, N = p[0], p[1]  # Divisão dos parâmetros

        # Solução Numérica
        tm1 = time.time()
        for i in range(self.nAlphas):
            alpha_value = self.alphas[i] + np.random.normal(0, self.perturb, self.nData)
            N_value = self.N_RotS[i] + np.random.normal(0, 50, self.nData)
            self.alpha_values.append(alpha_value)
            self.N_values.append(N_value)

            rhs = ca.vertcat((self.A1 / self.Lc) * ((lut(ca.vertcat(N, x[0])) * self.P1) - x[1]) * 1e3,
                             (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))
            
            ode = {'x': x, 'ode': rhs, 'p': p}

            F = ca.integrator('F', 'cvodes', ode, self.interval[0][0],self.dt)

            for j in range(self.nData):
                params = [alpha_value[j], N_value[j]]
                sol = F(x0=[init_m, init_p], p=params)
                xf_values = np.array(sol["xf"])
                aux1, aux2 = xf_values
                self.massFlowrate.append(aux1)
                self.PlenumPressure.append(aux2)
                self.Phi_values.append(lut(ca.vertcat(N_value[j], aux1)))
                init_m = aux1[-1]
                init_p = aux2[-1]
                self.RNN_train.append([aux1[0], aux2[0], alpha_value[j], N_value[j]])
                self.RNN_trainFut.append([aux1[0], aux2[0], alpha_value[j], N_value[j]])

        tm2 = time.time()
        self.time = tm2-tm1
        self.massFlowrate = np.reshape(self.massFlowrate, [self.nAlphas, self.nData])
        self.PlenumPressure = np.reshape(self.PlenumPressure, [self.nAlphas, self.nData])
        self.Phi_values = np.reshape(self.Phi_values, [self.nAlphas, self.nData])
        
        self.RNN_train = np.array(self.RNN_train)

        for i in range(len(self.RNN_train) - self.timestep):
            self.X_train.append(self.RNN_train[i:i + self.timestep])  
            if i + self.timestep < len(self.RNN_train):           
                self.y_train.append(self.RNN_train[i + self.timestep, :2])  

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)

        self.x_min = self.X_train.amin(dim=(0, 1), keepdim=True)
        self.x_max = self.X_train.amax(dim=(0, 1), keepdim=True)

        self.y_train = self.y_train.unsqueeze(1)
