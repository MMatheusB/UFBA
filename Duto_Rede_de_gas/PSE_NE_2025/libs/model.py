import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from libs.composicaogas import coef_con_ter

class RNNModelWrapper(nn.Module):
    def __init__(self, sistema, input_dim, hidden_dim, output_dim, num_layers,
                 x_min, x_max, y_min, y_max, lr=1e-3, device="cpu"):

        super().__init__()
        self.device = device
        self.sistema = sistema
        # Guardar limites para normalização
        self.x_min = torch.tensor(x_min, dtype=torch.float32).to(device)
        self.x_max = torch.tensor(x_max, dtype=torch.float32).to(device)
        self.y_min = torch.tensor(y_min, dtype=torch.float32).to(device)
        self.y_max = torch.tensor(y_max, dtype=torch.float32).to(device)

        # --------------------- MODELO ---------------------
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bias=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

        # Otimizador
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()


    def normalize_x(self, x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1

    def normalize_y(self, y):
        return 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1

    def denormalize_y(self, y_norm):
        return (y_norm + 1) * 0.5 * (self.y_max - self.y_min) + self.y_min

    def derivada_lagrange_3p(self, var):
        """
        Derivada espacial usando Lagrange 3 pontos (malha não uniforme)
        
        var: (n_times, n_points)
        """

        device = var.device
        dtype = var.dtype

        x = torch.tensor(self.sistema.l, device=device, dtype=dtype)

        n_times, n_points = var.shape
        dvar_dx = torch.zeros_like(var)

        for i in range(n_points):

            # stencil
            if i == 0:
                idx = [0, 1, 2]
            elif i == n_points - 1:
                idx = [n_points - 3, n_points - 2, n_points - 1]
            else:
                idx = [i - 1, i, i + 1]

            x_sub = x[idx]          # (3,)
            f_sub = var[:, idx]     # (n_times, 3)
            xi = x[i]

            deriv = torch.zeros(n_times, device=device, dtype=dtype)

            for j in range(3):

                Lj_deriv = 0

                for m in range(3):
                    if m != j:

                        prod = 1
                        for k in range(3):
                            if k != j and k != m:
                                prod *= (xi - x_sub[k]) / (x_sub[j] - x_sub[k])

                        Lj_deriv += prod / (x_sub[j] - x_sub[m])

                deriv += f_sub[:, j] * Lj_deriv

            dvar_dx[:, i] = deriv

        return dvar_dx
    
    def derivada_t_5p(self, dt, var):
        """
        Derivada temporal com 5 pontos (ordem 4 no centro)

        var: (5, n_points)
        """

        dvar_dt = torch.zeros_like(var)


        dvar_dt[0] = (var[1] - var[0]) / dt


        dvar_dt[1] = (var[2] - var[0]) / (2 * dt)

  
        dvar_dt[2] = (
            -var[4] + 8*var[3] - 8*var[1] + var[0]
        ) / (12 * dt)


        dvar_dt[3] = (var[4] - var[2]) / (2 * dt)


        dvar_dt[4] = (var[4] - var[3]) / dt

        return dvar_dt
    
    def forward(self, x):
        x = self.normalize_x(x)
        out, _ = self.rnn(x)

        out_last = out[:, -1, :]
        out = self.fc(out_last)
        out = self.denormalize_y(out)

        return out


    def train_model(self, dt, train_loader, epochs=50, lambda_phys=1e1):

        self.train()

        for ep in range(epochs):
            total_loss = 0.0
            total_data = 0.0
            total_phys = 0.0

            for xb, yb in train_loader:

                self.optimizer.zero_grad()

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                pred = self.forward(xb)

                n = pred.shape[0]

                w_T = 1e2
                w_V = 1e4   
                w_w = 1e5   
                w_P = 1e1
                w_m = 1e3

                weights = torch.tensor([w_T, w_V, w_w, w_P, w_m], device=self.device)

                err_in = (pred[0] - yb[0])**2
                loss_in = (err_in * weights).mean()

                err_out = (pred[n-1] - yb[n-1])**2
                loss_out = (err_out * weights).mean()

                loss_data = loss_in + loss_out

                n_points = self.sistema.n_points
                batch_size = pred.shape[0]

                assert batch_size % n_points == 0, "Batch desalinhado!"

                n_times = batch_size // n_points

                T = pred[:, 0].view(n_times, n_points)
                V = pred[:, 1].view(n_times, n_points)
                w = pred[:, 2].view(n_times, n_points)
                P = pred[:, 3].view(n_times, n_points)
                m = pred[:, 4].view(n_times, n_points)

                dT_dx = self.derivada_lagrange_3p(T)
                dV_dx = self.derivada_lagrange_3p(V)
                dw_dx = self.derivada_lagrange_3p(w)

                dT_dt = self.derivada_t_5p(dt, T)
                dV_dt = self.derivada_t_5p(dt, V)
                dw_dt = self.derivada_t_5p(dt, w)

                T = T.reshape(-1)
                V = V.reshape(-1)
                w = w.reshape(-1)
                P = P.reshape(-1)
                m = m.reshape(-1)

                dT_dx = dT_dx.reshape(-1)
                dV_dx = dV_dx.reshape(-1)
                dw_dx = dw_dx.reshape(-1)

                dT_dt = dT_dt.reshape(-1)
                dV_dt = dV_dt.reshape(-1)
                dw_dt = dw_dt.reshape(-1)

                T_np = T.detach().cpu().numpy()
                V_np = V.detach().cpu().numpy()
                P_np = P.detach().cpu().numpy()

                dPdT_list = []
                dPdV_list = []
                Cv_list = []
                f_list = []
                q_list = []
                m_phys_list = []
                P_phys_list = []

                for i in range(len(T_np)):

                    gas_temp = self.sistema.gas.copy_change_conditions(
                        T_np[i], None, V_np[i], 'gas'
                    )
                    gas_temp.ci_real()

                    dPdT_list.append(gas_temp.dPdT * 1000)
                    dPdV_list.append(gas_temp.dPdV * 1000)
                    Cv_list.append(gas_temp.Cvt / gas_temp.mixture.MM_m * 1000)
                    
                    P_phys_list.append(gas_temp.P)
                    
                    MM = gas_temp.mixture.MM_m
                    v_kg = V_np[i] / MM
                    rho = 1 / v_kg
                    A = np.pi * (self.sistema.D**2) / 4
                    m_phys = rho * w[i].item() * A        
                    m_phys_list.append(m_phys)
                    mu = self.sistema.visc.evaluate_viscosity(T_np[i], P_np[i])
                    Re = rho * w[i].item() * self.sistema.D / mu

                    f = self.sistema.fator_friccao(Re)

                    kappa = coef_con_ter(gas_temp)
                    h_t = self.sistema.coef_cov_fluid(kappa, mu, Re, gas_temp)

                    U = 1 / (
                        (1 / h_t)
                        + (self.sistema.D / (2 * self.sistema.k_solo))
                        * np.arccosh(2 * self.sistema.z_solo / self.sistema.D)
                    )

                    q = self.sistema.q_solo(rho, T_np[i], U)

                    f_list.append(f)
                    q_list.append(q)

                dPdT = torch.tensor(dPdT_list, device=self.device).float()
                dPdV = torch.tensor(dPdV_list, device=self.device).float()
                Cv   = torch.tensor(Cv_list, device=self.device).float()
                f    = torch.tensor(f_list, device=self.device).float()
                q    = torch.tensor(q_list, device=self.device).float()
                m_phys = torch.tensor(m_phys_list, device=self.device).float()
                P_phys = torch.tensor(P_phys_list, device=self.device).float()
                # =========================
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
                    -V * dPdV * dV_dx
                    -w * dw_dx
                    -f * w * torch.abs(w) / (2 * self.sistema.D)
                )

                res_T = dT_dt - F_T
                res_V = dV_dt - F_V
                res_w = dw_dt - F_w
                res_m = m - m_phys
                res_P = P - P_phys
                
                res_T = torch.nan_to_num(res_T, nan=0.0, posinf=0.0, neginf=0.0)
                res_V = torch.nan_to_num(res_V, nan=0.0, posinf=0.0, neginf=0.0)
                res_w = torch.nan_to_num(res_w, nan=0.0, posinf=0.0, neginf=0.0)

                loss_phys = (
                    (res_T**2).mean() +
                    (res_V**2).mean() +
                    (res_w**2).mean() +
                    1e-2 * (res_m**2).mean() + 
                    1e-2 * (res_P**2).mean()
                )

                loss = 1e-3*loss_data + lambda_phys * loss_phys

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

    def predict(self, x):
        """Função que faz sei lá o que"""
        self.eval()
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x_norm = self.normalize_x(x)

        with torch.no_grad():
            y_norm = self.forward(x_norm)

        y = self.denormalize_y(y_norm)
        return y.cpu().numpy()
