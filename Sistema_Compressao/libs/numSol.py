import numpy as np
import casadi as ca
from scipy.optimize import fsolve
import time

class Simulation:
    def __init__(self, A1, Lc, kv, P1, P_out, C, alphas, nAlphas, nData, perturb,tempo, dt):
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.alphas = alphas
        self.nAlphas = nAlphas
        self.nData = nData
        self.perturb = perturb
        self.dt = dt
        self.tempo = tempo

        self.interval = [np.linspace(i * self.tempo, (i + 1) * self.tempo, self.nData) for i in range(self.nAlphas)]
        self.time = 0
        
        self.alpha_values = []
        self.massFlowrate = []
        self.PlenumPressure = []
        self.RNN_train = []
        self.RNN_trainFut = []

    def fun(self, variables, alpha):
        (x, y) = variables
        eqn_1 = (self.A1 / self.Lc) * ((1.5 * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - alpha * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]

    def run(self):
        # Condições iniciais
        result = fsolve(self.fun, (0, 10), args=(self.alphas[0],))
        init_m, init_p = result

        # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        alpha = ca.MX.sym('alpha', 1)

        rhs = ca.vertcat((self.A1 / self.Lc) * ((1.5 * self.P1) - x[1]) * 1e3,
                    (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))
        
        ode = {'x': x, 'ode': rhs, 'p': alpha}

        F = ca.integrator('F', 'idas', ode, self.interval[0][0],self.dt)

        # Solução Numérica
        tm1 = time.time()
        for i in range(self.nAlphas):
            alpha_value = self.alphas[i] + np.random.normal(0, self.perturb, self.nData)
            self.alpha_values.append(alpha_value)

            for j in range(self.nData):
                sol = F(x0=[init_m, init_p], p=alpha_value[j])
                xf_values = np.array(sol["xf"])
                aux1, aux2 = xf_values
                self.massFlowrate.append(aux1)
                self.PlenumPressure.append(aux2)
                init_m = aux1[-1]
                init_p = aux2[-1]
                self.RNN_train.append([aux1[0], aux2[0], alpha_value[j]])
                self.RNN_trainFut.append([aux1[0], aux2[0], alpha_value[j]])

        tm2 = time.time()
        self.time = tm2-tm1
        self.massFlowrate = np.reshape(self.massFlowrate, [self.nAlphas, self.nData])
        self.PlenumPressure = np.reshape(self.PlenumPressure, [self.nAlphas, self.nData])