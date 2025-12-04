import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import torch


class SimuladorDuto:
    def __init__(self, sistema):
        self.sistema = sistema
        self.integrador = None
        self.resultados = {}

    def run(self, y0, z0, u0, N_perturb, N_data, Rot, dt):
        y = ca.SX.sym("y", 3 * self.sistema.n_points)
        z = ca.SX.sym("z", 11)  
        u = ca.SX.sym("u", 4)
        t = ca.SX.sym("t")

        dydt, alg_eqs = self.sistema.evaluate_dae(t, y, z, u)
        dae = {"x": y, "z": z, "p": u, "ode": dydt, "alg": alg_eqs}

        self.integrador = ca.integrator(
            "integrador", "idas", dae,
            {"tf": dt, "abstol": 1e-8, "reltol": 1e-8, "calc_ic": True}
        )

        y0 = np.array(y0, dtype=float)
        z0 = np.array(z0, dtype=float).flatten()
        u_current = np.array(u0, dtype=float)

        n_points = self.sistema.n_points
        n_diff = len(y0)
        n_alg = len(z0)

        y_sol = np.zeros((N_data*N_perturb, n_diff))
        z_sol = np.zeros((N_data*N_perturb, n_alg))

        y_sol[0, :] = y0
        z_sol[0, :] = z0

        y_current, z_current = y0, z0
        rot_array = []

        for i in range(N_perturb):
            u_current[0] = Rot[i]

            for j in range(N_data):

                sol = self.integrador(
                    x0=ca.DM(y_current),
                    z0=ca.DM(z_current),
                    p=ca.DM(u_current)
                )

                y_current = np.array(sol["xf"]).flatten()
                z_current = np.array(sol["zf"]).flatten()

                idx = i * N_data + j 

                y_sol[idx, :] = y_current
                z_sol[idx, :] = z_current
                rot_array.append(u_current[0])

        T_sol = np.zeros((N_data*N_perturb, n_points))
        V_sol = np.zeros((N_data*N_perturb, n_points))
        w_sol = np.zeros((N_data*N_perturb, n_points))

        for j in range(n_points):
            T_sol[:, j] = y_sol[:, 3*j + 0]
            V_sol[:, j] = y_sol[:, 3*j + 1]
            w_sol[:, j] = y_sol[:, 3*j + 2]

        P_out = np.zeros(N_data*N_perturb)
        m_dot_sol = np.zeros((N_data*N_perturb, n_points))
        A = np.pi * (self.sistema.D / 2)**2 

        for i in range(N_data*N_perturb):
            T_out = T_sol[i, -1]
            V_out = V_sol[i, -1]
            v_kg = V_sol[i, :] / self.sistema.gas.mixture.MM_m
            gas2 = self.sistema.gas.copy_change_conditions(T_out, None, V_out, 'gas')
            P_out[i] = gas2.P  
            m_dot_sol[i, :] = (1 / v_kg) * w_sol[i, :] * A

        # ARMAZENAR RESULTADOS + Z[9] E Z[10]
        self.resultados = {
            "tempo": np.linspace(0, N_data*N_perturb*dt, N_data*N_perturb),
            "T_sol": T_sol,
            "V_sol": V_sol,
            "w_sol": w_sol,
            "P_out": P_out,
            "m_dot": m_dot_sol,
            "z_sol": z_sol,
            "rot": rot_array,
            "z10": z_sol[:, 9],   # penúltima variável algébrica
            "z11": z_sol[:, 10]   # última variável algébrica
        }

        return self.resultados
    
    
    def train_dataset(self, time_step, horizon=10, output_dims=3):
        RNN_train = np.column_stack((
            self.resultados["z10"],             # m_dot_in
            self.resultados["z11"],             # P_out
            self.resultados["T_sol"][:, 0],     # T_in (nó 1)
            self.resultados["rot"]              # rotação/entrada
        ))

        # número total de observações
        N_total = RNN_train.shape[0]
        n_features = RNN_train.shape[1]

        x_list = []
        y_list = []

        # garantir que exista espaço para time_step + horizon
        max_i = N_total - (time_step + horizon) + 1
        if max_i <= 0:
            raise ValueError("Dados insuficientes: aumente N_data/N_perturb ou reduza time_step/horizon.")

        for i in range(max_i):
            # janela de entrada
            x_window = RNN_train[i : i + time_step, :]                     # shape (time_step, n_features)
            # sequência de saída com 'horizon' passos, apenas as primeiras output_dims colunas
            y_window = RNN_train[i + time_step : i + time_step + horizon, :output_dims]  # shape (horizon, output_dims)

            x_list.append(x_window)
            y_list.append(y_window)

        # converter para arrays/tensors
        x_np = np.array(x_list)   # (n_samples, time_step, n_features)
        y_np = np.array(y_list)   # (n_samples, horizon, output_dims)

        x_train = torch.tensor(x_np, dtype=torch.float32)
        y_train = torch.tensor(y_np, dtype=torch.float32)

        # calcular min/max para normalização (por feature)
        # para x: min/max por coluna de feature ao longo de todas as amostras e time_steps
        x_min = x_train.amin(dim=(0, 1), keepdim=True)  # shape (1,1,n_features)
        x_max = x_train.amax(dim=(0, 1), keepdim=True)

        # para y: min/max por feature ao longo de todas as amostras e horizon
        y_min = y_train.amin(dim=(0, 1), keepdim=True)  # shape (1,1,output_dims)
        y_max = y_train.amax(dim=(0, 1), keepdim=True)

        return RNN_train, x_train, y_train, x_min, x_max, y_min, y_max


    

    def plotar(self):
        if not self.resultados:
            print("⚠️ Nenhum resultado disponível. Rode a simulação primeiro.")
            return

        t_h = self.resultados["tempo"] / 3600
        T_sol = self.resultados["T_sol"]
        w_sol = self.resultados["w_sol"]
        V_sol = self.resultados["V_sol"]
        P_out = self.resultados["P_out"]
        m_dot_sol = self.resultados["m_dot"]

        z10 = self.resultados["z10"]
        z11 = self.resultados["z11"]

        n_points = self.sistema.n_points

        # --- Temperatura ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução da Temperatura ao longo do duto")
        for i in range(n_points):
            plt.plot(t_h, T_sol[:, i], label=f"Nó {i+1}")
        plt.xlabel("Tempo / h")
        plt.ylabel("Temperatura / K")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.show()

        # --- Velocidade ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução da Velocidade do Gás")
        for i in range(n_points):
            plt.plot(t_h, w_sol[:, i], label=f"Nó {i+1}")
        plt.xlabel("Tempo / h")
        plt.ylabel("Velocidade / m/s")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.show()

        # --- Volume específico ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução do Volume Específico")
        for i in range(n_points):
            plt.plot(t_h, V_sol[:, i], label=f"Nó {i+1}")
        plt.xlabel("Tempo / h")
        plt.ylabel("Volume específico / m³/mol")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.show()

        # --- Vazão mássica ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução da Vazão Mássica")
        for i in range(n_points):
            plt.plot(t_h, m_dot_sol[:, i], label=f"Nó {i+1}")
        plt.xlabel("Tempo / h")
        plt.ylabel("Vazão mássica / kg/s")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.show()

        # # --- Pressão na saída ---
        # plt.figure(figsize=(9, 6))
        # plt.plot(t_h, P_out, color='tab:red', linewidth=2)
        # plt.title("Pressão na Saída do Duto")
        # plt.xlabel("Tempo / h")
        # plt.ylabel("Pressão / kPa")
        # plt.grid(True)
        # plt.show()

        # --- z[10] ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução da Vazão Mássica no No 1")
        plt.plot(t_h, z10, linewidth=2)
        plt.xlabel("Tempo / h")
        plt.ylabel("z[10]")
        plt.grid(True)
        plt.show()

        # --- z[11] ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução da Pressão na Saída do Duto")
        plt.plot(t_h, z11, linewidth=2, color="purple")
        plt.xlabel("Tempo / h")
        plt.ylabel("z[11]")
        plt.grid(True)
        plt.show()
