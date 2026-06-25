import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from libs.composicaogas import coef_con_ter
from concurrent.futures import ThreadPoolExecutor
import os

class RNNModelWrapper(nn.Module):
    def __init__(self, sistema, input_dim, hidden_dim, output_dim, num_layers,
                 x_min, x_max, y_min, y_max, T_norm, V_norm, P_norm, w_norm, lr=1e-3, device="cpu"):

        super().__init__()
        self.device = device
        self.sistema = sistema
        # Guardar limites para normalização
        self.x_min = x_min.clone().detach().float().to(device)
        self.x_max = x_max.clone().detach().float().to(device)

        self.y_min = y_min.clone().detach().float().to(device)
        self.y_max = y_max.clone().detach().float().to(device)
        self.T_norm= T_norm
        self.V_norm= V_norm
        self.P_norm= P_norm
        self.w_norm= w_norm
        # --------------------- MODELO ---------------------
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bias=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.D_lagrange = self.build_lagrange_matrix()
        # Otimizador
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self._executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    def normalize_x(self, x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1

    def normalize_y(self, y):
        return 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1

    def denormalize_y(self, y_norm):
        return (y_norm + 1) * 0.5 * (self.y_max - self.y_min) + self.y_min

    def build_lagrange_matrix(self):

        x = np.asarray(self.sistema.l)

        n_points = len(x)

        D = np.zeros((n_points, n_points))

        for i in range(n_points):

            if i == 0:
                idx = [0, 1, 2]
            elif i == n_points - 1:
                idx = [n_points - 3, n_points - 2, n_points - 1]
            else:
                idx = [i - 1, i, i + 1]

            x_sub = x[idx]
            xi = x[i]

            coeffs = np.zeros(3)

            for j in range(3):

                Lj_deriv = 0.0

                for m in range(3):

                    if m != j:

                        prod = 1.0

                        for k in range(3):

                            if k != j and k != m:
                                prod *= (
                                    (xi - x_sub[k])
                                    /
                                    (x_sub[j] - x_sub[k])
                                )

                        Lj_deriv += (
                            prod
                            /
                            (x_sub[j] - x_sub[m])
                        )

                coeffs[j] = Lj_deriv

            D[i, idx] = coeffs

        return torch.tensor(
            D,
            dtype=torch.float32
        )
    
    def derivada_lagrange_3p(self, var):

        D = self.D_lagrange.to(
            device=var.device,
            dtype=var.dtype
        )

        return var @ D.T

    def physics_point(self, T, V, P, w):

        gas_temp = self.sistema.gas.copy_change_conditions(
            T,
            None,
            V,
            "gas"
        )

        gas_temp.ci_real()

        dPdT = gas_temp.dPdT * 1000
        dPdV = gas_temp.dPdV * 1000

        Cv = (
            gas_temp.Cvt /
            gas_temp.mixture.MM_m *
            1000
        )

        P_phys = gas_temp.P

        MM = gas_temp.mixture.MM_m

        v_kg = V / MM
        rho = 1.0 / v_kg

        A = np.pi * (self.sistema.D**2) / 4

        m_phys = rho * w * A

        mu = self.sistema.visc.evaluate_viscosity(
            T,
            P
        )

        Re = rho * w * self.sistema.D / mu

        f = self.sistema.fator_friccao(Re)

        kappa = coef_con_ter(gas_temp)

        h_t = self.sistema.coef_cov_fluid(
            kappa,
            mu,
            Re,
            gas_temp
        )

        U = 1.0 / (
            (1.0 / h_t)
            +
            (
                self.sistema.D /
                (2 * self.sistema.k_solo)
            )
            *
            np.arccosh(
                2 * self.sistema.z_solo /
                self.sistema.D
            )
        )

        q = self.sistema.q_solo(
            rho,
            T,
            U
        )

        return (
            dPdT,
            dPdV,
            Cv,
            f,
            q,
            P_phys,
            m_phys
        )


    def physics_rhs(self, T, V, w, P):

        n_points = self.sistema.n_points
        batch_size = T.shape[0]
        assert batch_size % n_points == 0
        n_times = batch_size // n_points

        # =====================================
        # Derivadas espaciais
        # =====================================

        T_grid = T.view(n_times, n_points)
        V_grid = V.view(n_times, n_points)
        w_grid = w.view(n_times, n_points)

        dT_dx = self.derivada_lagrange_3p(T_grid).reshape(-1)
        dV_dx = self.derivada_lagrange_3p(V_grid).reshape(-1)
        dw_dx = self.derivada_lagrange_3p(w_grid).reshape(-1)

        # =====================================
        # CPU arrays (uma única transferência)
        # =====================================

        T_np = T.detach().cpu().numpy()
        V_np = V.detach().cpu().numpy()
        P_np = P.detach().cpu().numpy()
        w_np = w.detach().cpu().numpy()

        # =====================================
        # EOS paralela com ThreadPoolExecutor
        # =====================================

        def call_physics(i):
            return self.physics_point(T_np[i], V_np[i], P_np[i], w_np[i])

        results = list(self._executor.map(call_physics, range(len(T_np))))

        (
            dPdT_list,
            dPdV_list,
            Cv_list,
            f_list,
            q_list,
            P_phys_list,
            m_phys_list,
        ) = zip(*results)

        # =====================================
        # Conversão em bloco (evita N alocações)
        # =====================================

        def to_tensor(lst):
            return torch.as_tensor(
                np.array(lst, dtype=np.float32),
                device=self.device
            )

        dPdT  = to_tensor(dPdT_list)
        dPdV  = to_tensor(dPdV_list)
        Cv    = to_tensor(Cv_list)
        f     = to_tensor(f_list)
        q     = to_tensor(q_list)
        P_phys = to_tensor(P_phys_list)
        m_phys = to_tensor(m_phys_list)

        # =====================================
        # Equações governantes (igual ao original)
        # =====================================

        F_T = (
            -w * dT_dx
            - T * (V * dPdT / Cv) * dw_dx
            + f * w**2 * torch.abs(w) / (2 * self.sistema.D * Cv)
            + q / Cv
        )

        F_V = (
            -w * dV_dx
            + V * dw_dx
        )

        F_w = (
            -V * dPdT * dT_dx
            - V * dPdV * dV_dx
            - w * dw_dx
            - f * w * torch.abs(w) / (2 * self.sistema.D)
        )

        return F_T, F_V, F_w, P_phys, m_phys
    
    def forward(self, x):
        x = self.normalize_x(x)
        out, _ = self.rnn(x)

        out_last = out[:, -1, :]
        out = self.fc(out_last)
        out = self.denormalize_y(out)

        return out


    def train_model(self, dt, train_loader, epochs, lambda_phys=1e1):

        self.train()

        warmup_epochs = 1000 # física entra a partir da segunda época

        for ep in range(epochs):

            total_loss = 0.0
            total_data = 0.0
            total_phys = 0.0

            for xb, yb in train_loader:

                self.optimizer.zero_grad()

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                pred = self.forward(xb)

                T_k1 = pred[:,0]
                V_k1 = pred[:,1]
                w_k1 = pred[:,2]
                P_k1 = pred[:,3]
                m_k1 = pred[:,4]

                T_k2 = pred[:,5]
                V_k2 = pred[:,6]
                w_k2 = pred[:,7]
                P_k2 = pred[:,8]
                m_k2 = pred[:,9]


                MM = self.sistema.gas.mixture.MM_m

                A = np.pi * (self.sistema.D**2) / 4

                v_kg_nn = self.V_norm / MM

                rho_nn = 1.0 / v_kg_nn

                m_scale = rho_nn * self.w_norm * A

                if ep < warmup_epochs:

                    loss_phys = torch.tensor(
                        0.0,
                        device=self.device
                    )

                else:

                    # ----------------------------------------------
                    # RHS da física
                    # ----------------------------------------------

                    F_T_k1, F_V_k1, F_w_k1, _, _ = self.physics_rhs(
                        T_k1,
                        V_k1,
                        w_k1,
                        P_k1
                    )

                    F_T_k2, F_V_k2, F_w_k2, P_phys, m_phys = self.physics_rhs(
                        T_k2,
                        V_k2,
                        w_k2,
                        P_k2
                    )

                    # ----------------------------------------------
                    # Adams-Moulton 2ª ordem
                    # ----------------------------------------------

                    res_T = (
                        (T_k2 - T_k1)/dt
                        -
                        0.5*(F_T_k2 + F_T_k1)
                    )

                    res_V = (
                        (V_k2 - V_k1)/dt
                        -
                        0.5*(F_V_k2 + F_V_k1)
                    )

                    res_w = (
                        (w_k2 - w_k1)/dt
                        -
                        0.5*(F_w_k2 + F_w_k1)
                    )
                    # ----------------------------------------------
                    # Resíduos algébricos
                    # ----------------------------------------------

                    res_m = 0*(m_k2-m_phys)
                    res_P = 0*(P_k2-P_phys)
                    # ----------------------------------------------
                    # Loss física
                    # ----------------------------------------------

                    loss_phys = (
                        1e-4 * (res_T**2).mean()
                        +
                        1e-4 * (res_V**2).mean()
                        +
                        1e-4 * (res_w**2).mean()
                        +
                        1e-4 * (res_m**2).mean()
                        +
                        1e-4 * (res_P**2).mean()
                    )

                # ==================================================
                # Loss dos dados
                # ==================================================

                w_T = 5e0 / self.T_norm 
                w_V = 1e3 / self.V_norm 
                w_w = 3e0 / self.w_norm 
                w_P = 1e0 / self.P_norm 
                w_m = 15e0 / m_scale 
                weights = torch.tensor(
                [
                    w_T,w_V,w_w,w_P,w_m,
                    w_T,w_V,w_w,w_P,w_m
                ],
                device=self.device)

                err_in = (pred - yb) ** 2

                loss_data = (err_in*weights).mean()

                # ==================================================
                # Loss total
                # ==================================================

                loss = loss_data + lambda_phys * loss_phys

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()
                total_data += loss_data.item()
                total_phys += loss_phys.item()

            print(
                f"Epoch {ep+1}/{epochs} | "
                f"Total = {total_loss/len(train_loader):.6f} | "
                f"Data = {total_data/len(train_loader):.6f} | "
                f"Phys = {total_phys/len(train_loader):.6f}"
            )
