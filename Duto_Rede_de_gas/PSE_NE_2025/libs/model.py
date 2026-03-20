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


    def forward(self, x):

        x = x.clone()

        indice = x[:, :, 0].long()

        x_real = torch.tensor(self.sistema.l, device=x.device).float()

        x_pos = x_real[indice]

        x = torch.cat([
            x_pos.unsqueeze(-1),
            x[:, :, 1:]
        ], dim=-1)

        out, _ = self.rnn(x)

        out_last = out[:, -1, :]
        out = self.fc(out_last)

        T = torch.nn.functional.softplus(out[:, 0]) + 200.0
        V = torch.nn.functional.softplus(out[:, 1]) + 1e-3
        w = out[:, 2]
        P = torch.nn.functional.softplus(out[:, 3]) + 1e5
        m = out[:, 4]

        # clamp extra
        T = torch.clamp(T, 200.0, 2000.0)
        V = torch.clamp(V, 1e-6, 1.0)
        P = torch.clamp(P, 1e5, 1e7)

        out = torch.stack([T, V, w, P, m], dim=1)

        return out


    def train_model(self, train_loader, epochs=50, lambda_phys=1.0):

        self.train()

        for ep in range(epochs):

            total_loss = 0.0
            total_data = 0.0
            total_phys = 0.0

            for xb, yb in train_loader:

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                xb = xb.clone().detach().requires_grad_(True)

                self.optimizer.zero_grad()

                pred = self.forward(xb)

                pos = xb[:, 0, 0]

                mask = (pos == 0) | (pos == self.sistema.l[-1])

                if mask.any():
                    loss_data = ((pred - yb)[mask]**2).mean()
                else:
                    loss_data = torch.tensor(0.0, device=self.device)

                T = pred[:, 0]
                V = pred[:, 1]
                w = pred[:, 2]
                P = pred[:, 3]
                m = pred[:, 4]

                dT_dx = torch.autograd.grad(
                    T, xb,
                    grad_outputs=torch.ones_like(T),
                    create_graph=True
                )[0][:, -1, 0]

                dV_dx = torch.autograd.grad(
                    V, xb,
                    grad_outputs=torch.ones_like(V),
                    create_graph=True
                )[0][:, -1, 0]

                dw_dx = torch.autograd.grad(
                    w, xb,
                    grad_outputs=torch.ones_like(w),
                    create_graph=True
                )[0][:, -1, 0]

                T_np = T.detach().cpu().numpy()
                V_np = V.detach().cpu().numpy()
                P_np = P.detach().cpu().numpy()

                dPdT_list = []
                dPdV_list = []
                Cv_list = []
                mu_list = []
                f_list = []
                q_list = []

                for i in range(len(T_np)):

                    gas_temp = self.sistema.gas.copy_change_conditions(
                        T_np[i], None, V_np[i], 'gas'
                    )
                    gas_temp.ci_real()
                    dPdT_list.append(gas_temp.dPdT * 1000)
                    dPdV_list.append(gas_temp.dPdV * 1000)
                    Cv_list.append(gas_temp.Cvt / gas_temp.mixture.MM_m * 1000)

                    MM = gas_temp.mixture.MM_m
                    v_kg = V_np[i] / MM
                    rho = 1 / v_kg

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

                    mu_list.append(mu)
                    f_list.append(f)
                    q_list.append(q)

                # voltar pra torch
                dPdT = torch.tensor(dPdT_list, device=self.device).float()
                dPdV = torch.tensor(dPdV_list, device=self.device).float()
                Cv   = torch.tensor(Cv_list, device=self.device).float()
                f    = torch.tensor(f_list, device=self.device).float()
                q    = torch.tensor(q_list, device=self.device).float()

                res_T = (
                    -w * dT_dx
                    - T * (V * dPdT / Cv) * dw_dx
                    + f * w**2 * torch.abs(w) / (2 * self.sistema.D * Cv)
                    + q / Cv
                )

                res_V = (
                    -w * dV_dx
                    + V * dw_dx
                )

                res_w = (
                    -V * dPdT * dT_dx
                    -V * dPdV * dV_dx
                    -w * dw_dx
                    -f * w * torch.abs(w) / (2 * self.sistema.D)
                )
                
                res_T = torch.nan_to_num(res_T, nan=0.0)
                res_V = torch.nan_to_num(res_V, nan=0.0)
                res_w = torch.nan_to_num(res_w, nan=0.0)
                
                loss_phys = (
                    (res_T**2).mean() +
                    (res_V**2).mean() +
                    (res_w**2).mean()
                )

                loss = loss_data + lambda_phys * loss_phys

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_data += loss_data.item()
                total_phys += loss_phys.item()

            if (ep + 1) % 10 == 0:
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
