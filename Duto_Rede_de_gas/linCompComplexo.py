from libs.eos_database import *
from libs.gc_eos_soave import *
from libs.viscosity import *
from libs.duto_casadi import *
import casadi as ca
from scipy.stats import qmc
import control as ctrl
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

class linDiscretizeComp():
    def __init__(self):
        self.list_names = ["CH4", "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12", 
                        "nC6H14", "nC7H16", "nC8H18", "nC9H20", "nC10H22", "nC11H24", 
                        "nC12H26", "nC14H30", "N2", "H2O", "CO2", "C15+"]

        self.nwe = [0.9834, 0.0061, 0.0015, 0.0003, 0.0003, 0.00055, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0008, 0.0]

        self.dict_composition= {list_names[i]: self.nwe[i] for i in range(len(self.nwe))}

        self.mixture = Mixture(list_of_species, self.dict_composition)

        self.volumn_desviation = [0] * 19

        self.Vpp = 2.0 
        A1 = 2.6e-3
        gas = gc_eos_class(self.mixture, 300, 8400, None, 1, 0, Aij, self.volumn_desviation, 'gas')
        visc = viscosity(self.mixture, self.volumn_desviation)
        D = 0.42
        # Criar instância do duto com os parâmetros necessários  # ajuste conforme seu import real
        meu_duto = self.duto_casadi(gas=gas, visc=visc,  Lc=200000, D=D)

        # Número de nós (comprimento de self.l)
        n_nodes = len(meu_duto.l)
        #self.timestep = 3

        self.x0 = [3.00000000e+02, 2.61601541e-01, 1.96792505e+01, 2.99750360e+02,
        2.62187303e-01, 1.97233151e+01, 2.99261401e+02, 2.63374723e-01,
        1.98126400e+01, 2.98550800e+02, 2.65192052e-01, 1.99493505e+01,
        2.97647356e+02, 2.67684185e-01, 2.01368238e+01, 2.96578155e+02,
        2.70896961e-01, 2.03785082e+01, 2.95383985e+02, 2.74893793e-01,
        2.06791741e+01, 2.94090879e+02, 2.79720154e-01, 2.10422422e+01,
        2.92744196e+02, 2.85438443e-01, 2.14724065e+01, 2.91361455e+02,
        2.92067847e-01, 2.19711104e+01, 2.89987463e+02, 2.99637468e-01,
        2.25405431e+01, 2.88629655e+02, 3.08101151e-01, 2.31772325e+01,
        2.87330071e+02, 3.17402,240e-01, 2.38769167e+01, 2.86090146e+02,
        3.27370360e-01, 2.46267789e+01, 2.84948609e+02, 3.37788458e-01,
        2.54104913e+01, 2.83907808e+02, 3.48292584e-01, 2.62006752e+01,
        2.83002,477e+02, 3.58441830e-01, 2.69641628e+01, 2.82240861e+02,
        3.67663890e-01, 2.7657902,0e+01, 2.81650995e+02, 3.75349875e-01,
        2.82360883e+01, 2.81244962e+02, 3.80890037e-01, 2.86528527e+01,
        2.81038776e+02, 3.83798501e-01, 2.88716450e+01]
        #self.z0 = [6245.39, 6245.39, 321.672, 0.445562, 319.423, 0.503621, 320.097, 0.396345, 339.69, 0.42885, 0.514917]   comp -> T2, V2 -> [duto] ->
        self.u0 = [3.00000000e+02, 2.61601541e-01] #-> T2, V2
        self.nx = len(self.x0)
        self.nu = len(self.u0)

        pickle_filename = "MPC_to_NN/MPCvsNN - Sistema Compressão/libs/estEstacionario.pkl"

        if os.path.exists(pickle_filename):
            # Se o arquivo existir, carrega as variáveis
            with open(pickle_filename, 'rb') as f:
                data = pickle.load(f)
            
            self.x_ss = data['x_ss']
            self.z_ss = data['z_ss']
            self.u_ss = data['u_ss']
            self.SysD = data['SysD']

        else:
            # Se não existir, executa os cálculos e salva os resultados
            self.x_ss, self.z_ss = self.compSteadyState()
            self.x_ss,  self.z_ss = self.x_ss.reshape(-1,1), self.z_ss.reshape(-1,1)
            self.u_ss = array(self.u0).reshape(-1,1)
            self.SysD = self.discretize(self.linearize())

            # Agrupa as variáveis em um dicionário para salvar
            data_to_save = {
                'x_ss': self.x_ss,
                'z_ss': self.z_ss,
                'u_ss': self.u_ss,
                'SysD': self.SysD
            }

            # Salva o dicionário no arquivo pickle
            with open(pickle_filename, 'wb') as f:
                pickle.dump(data_to_save, f)
    
    
    def linearize(self):
        y = ca.SX.sym("y", 3 * self.meu_duto.n_points)
        t = ca.SX.sym("t")
        u  =
        dydt = self.meu_duto.evaluate_dae(t, y)
        dae = {
            'x': x_sym,
            'z': z_sym,
            'p': u_sym,
            'ode': vertcat(*ode_sym),
            'alg': vertcat(*alg_sym)
        }

        f_expr = dae['ode']
        g_expr = dae["alg"]

        # EDOs
        Axx_sym = ca.jacobian(f_expr, x_sym) # df/dx
        Axz_sym = ca.jacobian(f_expr, z_sym) # df/dz
        Bx_sym = ca.jacobian(f_expr, u_sym) # df/du

        eval_Axx = ca.Function('eval_Axx', [x_sym, z_sym, u_sym], [Axx_sym])
        eval_Axz = ca.Function('eval_Axz', [x_sym, z_sym, u_sym], [Axz_sym])
        eval_Bx = ca.Function('eval_Bx', [x_sym, z_sym, u_sym], [Bx_sym])

        # ALG
        Azx_sym = ca.jacobian(g_expr, x_sym) # dg/dx
        Azz_sym = ca.jacobian(g_expr, z_sym) # dg/dz
        Bz_sym = ca.jacobian(g_expr, u_sym) # dg/du

        eval_Azx = ca.Function('eval_Axx', [x_sym, z_sym, u_sym], [Azx_sym])
        eval_Azz = ca.Function('eval_Axz', [x_sym, z_sym, u_sym], [Azz_sym])
        eval_Bz = ca.Function('eval_Bx', [x_sym, z_sym, u_sym], [Bz_sym])

        Axx = np.squeeze(eval_Axx(self.x_ss, self.z_ss, self.u_ss))
        Axz = np.squeeze(eval_Axz(self.x_ss, self.z_ss, self.u_ss))
        Bx = np.squeeze(eval_Bx(self.x_ss, self.z_ss, self.u_ss))
        Azx = np.squeeze(eval_Azx(self.x_ss, self.z_ss, self.u_ss))
        Azz = np.squeeze(eval_Azz(self.x_ss, self.z_ss, self.u_ss))
        Bz = np.squeeze(eval_Bz(self.x_ss, self.z_ss, self.u_ss))

        # dotX = Axx @ X + Axz @ Z + Bx @ U
        # 0 = Azx @ X + Azz @ Z + Bz @ U
        # Z = - Azz^{-1} @ Azx @ X - Azz^{-1} @ Bz @ U
        # dotX = (Axx - Axz @ Azz^{-1} @ Azx) @ X + (Bx - Axz @ Azz^{-1} @ Bz) @ U

        Ac = Axx - (Axz @ np.linalg.inv(Azz) @ Azx)
        Bc = Bx - Axz @ np.linalg.inv(Azz) @ Bz

        return Ac, Bc

    def discretize(self, linSys):
        Ac, Bc = linSys

        # Normalização
        Dx = numpy.diag(self.x_ss.flatten())
        Du = numpy.diag(self.u_ss.flatten())
        
        A_ = numpy.linalg.inv(Dx) @ Ac @ Dx
        B_ = numpy.linalg.inv(Dx) @ Bc @ Du

        dt = 0.5
        
        Cc = np.eye(len(Ac))
        Dc = np.zeros((Ac.shape[0], Bc.shape[1]))

        sys_c = ctrl.ss(A_,B_,Cc,Dc)
        sys_d = ctrl.c2d(sys_c,dt, method = 'zoh')
        A, B, C, D = ctrl.ssdata(sys_d)

        print(np.linalg.eigvals(A))

        return A, B, C, D
    
    def normalize(self, var, type = 'x'):
        if type == 'x':
            var_ = var / self.x_ss - 1
        elif type == 'u':
            var_ = var / self.u_ss - 1
        return var_
    
    def denormalize(self, var_, type = 'x'):
        if type == 'x':
            var = self.x_ss * var_ + self.x_ss
        elif type == 'u':
            var_ = self.u_ss * var_ + self.u_ss
        return var_
    
if __name__ == "__main__":
    # --- Passo 1: Obtenção do sistema discretizado ---
    model = linDiscretizeComp()
    A, B, C, D = model.SysD

    # --- Passo 2: Configuração da Simulação ---
    N_sim = 80
    dt = 0.5
    
    Y_hist = np.zeros((C.shape[0], N_sim)) # Histórico para as SAÍDAS
    
    # Condição inicial de estado (X)
    X_k = model.normalize(model.x_ss)
    
    # Define a entrada de degrau (U)
    u_step = model.normalize(model.u_ss, 'u')
    u_step[2] = model.normalize(model.u_ss, 'u')[2] + 0.15 # Aumenta a rotação (terceira entrada) em 15%
    
    # --- Passo 3: Execução do Loop de Simulação ---
    x_k_dev = X_k - model.normalize(model.x_ss) # Desvio inicial é zero
    u_k_dev = u_step - model.normalize(model.u_ss, 'u') # Desvio da entrada é o degrau

    for k in range(N_sim):
        # Equação de saída (calcula a saída no tempo k)
        # y_dev[k] = C * x_dev[k] + D * u_dev[k]
        y_k_dev = C @ x_k_dev + D @ u_k_dev
        
        # Armazena a saída real (desvio + ponto de operação)
        # Assumindo y_ss = x_ss, já que C=I e D=0
        Y_hist[:, k] = (model.denormalize(y_k_dev) + model.x_ss).flatten()
        
        # Equação de estados (calcula o próximo estado)
        # x_dev[k+1] = A * x_dev[k] + B * u_dev[k]
        x_k_dev = A @ x_k_dev + B @ u_k_dev
        
    # --- Passo 4: PLOT com Matplotlib (Gráficos Separados) ---
    
    # 4.1. Criar o vetor de tempo
    time_vec = np.linspace(0, (N_sim - 1) * dt, N_sim)

    # 4.2. Criar a figura e os subplots
    # 3 linhas, 1 coluna. sharex=True liga o eixo X de todos os gráficos.
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Resposta das Saídas do Sistema a um Degrau na Entrada', fontsize=16)

    # Nomes das saídas para os títulos (ajuste conforme o significado de cada uma)
    output_names = ['Vazão Mássica', 'Pressão', 'Volume']

    # 4.3. Loop para plotar cada saída em seu respectivo subplot
    for i in range(Y_hist.shape[0]):
        axes[i].plot(time_vec, Y_hist[i, :], label=f'Resposta de Y$_{i+1}$')
        # Adiciona uma linha tracejada para indicar o valor inicial (estado estacionário)
        axes[i].axhline(y=model.x_ss[i], color='r', linestyle='--', label=f'Y$_{i+1}$ (ss)')
        
        axes[i].set_title(output_names[i])
        axes[i].set_ylabel('Valor')
        axes[i].grid(True)
        axes[i].legend()

    # 4.4. Adicionar rótulo do eixo X apenas no último gráfico
    axes[-1].set_xlabel('Tempo (s)')

    # 4.5. Ajustar o layout e exibir o gráfico
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta para o supertítulo não sobrepor
    plt.show()
