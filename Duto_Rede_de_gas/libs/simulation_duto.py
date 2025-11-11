import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

class SimuladorDuto:
    def __init__(self, sistema, dt=60, n_steps=5000):
        self.sistema = sistema
        self.dt = dt
        self.n_steps = n_steps
        self.integrador = None
        self.resultados = {}

    def run(self, y0, z0, u0):
        y = ca.SX.sym("y", 3 * self.sistema.n_points)
        z = ca.SX.sym("z", 9)
        u = ca.SX.sym("u", 4)
        t = ca.SX.sym("t")

        dydt, alg_eqs = self.sistema.evaluate_dae(t, y, z, u)
        dae = {"x": y, "z": z, "p": u, "ode": dydt, "alg": alg_eqs}

        self.integrador = ca.integrator(
            "integrador", "idas", dae,
            {"tf": self.dt, "abstol": 1e-8, "reltol": 1e-8, "calc_ic": True}
        )

        y0 = np.array(y0, dtype=float)
        z0 = np.array(z0, dtype=float).flatten()
        u_current = np.array(u0, dtype=float)

        n_points = self.sistema.n_points
        n_diff = len(y0)
        n_alg = len(z0)

        y_sol = np.zeros((self.n_steps, n_diff))
        z_sol = np.zeros((self.n_steps, n_alg))

        y_sol[0, :] = y0
        z_sol[0, :] = z0

        y_current, z_current = y0, z0

        for i in range(1, self.n_steps):
            if i == 1200:
                u_current[0] = 700.0
                u_current[-1] = 1.92
            elif i == 2400:
                u_current[0] = 670.0
            elif i == 3600:
                u_current[0] = 710.0

            sol = self.integrador(x0=ca.DM(y_current), z0=ca.DM(z_current), p=ca.DM(u_current))
            y_current = np.array(sol["xf"]).flatten()
            z_current = np.array(sol["zf"]).flatten()

            y_sol[i, :] = y_current
            z_sol[i, :] = z_current

        T_sol = np.zeros((self.n_steps, n_points))
        V_sol = np.zeros((self.n_steps, n_points))
        w_sol = np.zeros((self.n_steps, n_points))

        for j in range(n_points):
            T_sol[:, j] = y_sol[:, 3*j + 0]
            V_sol[:, j] = y_sol[:, 3*j + 1]
            w_sol[:, j] = y_sol[:, 3*j + 2]

        P_out = np.zeros(self.n_steps)
        m_dot_sol = np.zeros((self.n_steps, n_points))
        A = np.pi * (self.sistema.D / 2)**2 

        for i in range(self.n_steps):
            T_out = T_sol[i, -1]
            V_out = V_sol[i, -1]
            v_kg = V_sol[i, :]/self.sistema.gas.mixture.MM_m
            gas2 = self.sistema.gas.copy_change_conditions(T_out, None, V_out, 'gas')
            P_out[i] = gas2.P  
            m_dot_sol[i, :] = (1 / v_kg) * w_sol[i, :] * A

        # armazenar resultados
        self.resultados = {
            "tempo": np.linspace(0, self.n_steps*self.dt, self.n_steps),
            "T_sol": T_sol,
            "V_sol": V_sol,
            "w_sol": w_sol,
            "P_out": P_out,
            "m_dot": m_dot_sol,
            "z_sol": z_sol
        }

        return self.resultados

    def plotar(self):
        """Gera plots básicos das variáveis simuladas."""
        if not self.resultados:
            print("⚠️ Nenhum resultado disponível. Rode a simulação primeiro.")
            return

        t_h = self.resultados["tempo"] / 3600
        T_sol = self.resultados["T_sol"]
        w_sol = self.resultados["w_sol"]
        V_sol = self.resultados["V_sol"]
        P_out = self.resultados["P_out"]
        m_dot_sol = self.resultados["m_dot"]

        n_points = self.sistema.n_points

        # --- Temperatura ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução da Temperatura ao longo do duto")
        for i in range(n_points):
            plt.plot(t_h, T_sol[:, i], label=f"Nó {i+1}")
        plt.xlabel("Tempo [h]")
        plt.ylabel("Temperatura [K]")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.show()

        # --- Velocidade ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução da Velocidade do Gás")
        for i in range(n_points):
            plt.plot(t_h, w_sol[:, i], label=f"Nó {i+1}")
        plt.xlabel("Tempo [h]")
        plt.ylabel("Velocidade [m/s]")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.show()

        # --- Volume específico ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução do Volume Específico")
        for i in range(n_points):
            plt.plot(t_h, V_sol[:, i], label=f"Nó {i+1}")
        plt.xlabel("Tempo [h]")
        plt.ylabel("Volume específico [m³/kmol]")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.show()

        # --- Vazão mássica ---
        plt.figure(figsize=(9, 6))
        plt.title("Evolução da Vazão Mássica")
        for i in range(n_points):
            plt.plot(t_h, m_dot_sol[:, i], label=f"Nó {i+1}")
        plt.xlabel("Tempo [h]")
        plt.ylabel("Vazão mássica [kg/s]")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.show()

        # --- Pressão na saída ---
        plt.figure(figsize=(9, 6))
        plt.plot(t_h, P_out, color='tab:red', linewidth=2)
        plt.title("Pressão na Saída do Duto")
        plt.xlabel("Tempo [h]")
        plt.ylabel("Pressão [Pa]")
        plt.grid(True)
        plt.show()

