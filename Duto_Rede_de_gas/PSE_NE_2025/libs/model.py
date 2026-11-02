import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from libs.composicaogas import coef_con_ter
from concurrent.futures import ThreadPoolExecutor
from casadi import SX, Function, jacobian
import os


class PhysicsCasADiFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, physics_fn_batch, jac_fn_batch, batch_size):
        y_np = y.detach().cpu().numpy().T

        dydt_np = np.array(physics_fn_batch(y_np))

        ctx.save_for_backward(y)
        ctx.jac_fn_batch = jac_fn_batch
        ctx.batch_size = batch_size

        return torch.as_tensor(dydt_np.T, dtype=y.dtype, device=y.device)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        y_np = y.detach().cpu().numpy().T

        jac_np = np.array(ctx.jac_fn_batch(y_np))
        jac_list = np.hsplit(jac_np, ctx.batch_size)
        jac_stack = np.stack(jac_list, axis=0)

        grad_output_np = grad_output.detach().cpu().numpy()

        grad_input_np = np.einsum('bi,bij->bj', grad_output_np, jac_stack)

        grad_input = torch.as_tensor(grad_input_np, dtype=y.dtype, device=y.device)

        return grad_input, None, None, None

class RNNModelWrapper(nn.Module):
    def __init__(self, sistema, input_dim, hidden_dim, num_layers,
                 x_min, x_max, y_min, y_max, T_norm, V_norm, P_norm, w_norm,
                 batch_size_phys, lr=1e-3, device="cpu"):

        super().__init__()
        self.device = device
        self.sistema = sistema
        self.n_points = sistema.n_points

        self.x_min = x_min.clone().detach().float().to(device)
        self.x_max = x_max.clone().detach().float().to(device)
        self.y_min = y_min.clone().detach().float().to(device)
        self.y_max = y_max.clone().detach().float().to(device)

        self.T_norm = T_norm
        self.V_norm = V_norm
        self.P_norm = P_norm
        self.w_norm = w_norm

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bias=True)

        self.fc = nn.Linear(hidden_dim, self.n_points * 10)

        self.D_lagrange = self.build_lagrange_matrix()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # ---------------------------------------------------------
        # Construção da física via CasADi (uma vez só)
        # batch_size_phys = batch_size do treino * 2 (k1+k2 concatenados)
        # ---------------------------------------------------------
        self.n_states = 3 * self.n_points
        self.batch_size_phys = batch_size_phys

        y_sym = SX.sym("y", self.n_states)
        dydt_sym = sistema.evaluate_physics_only(y_sym)
        jac_sym = jacobian(dydt_sym, y_sym)

        physics_fn = Function("physics_only", [y_sym], [dydt_sym])
        jac_fn = Function("physics_jacobian", [y_sym], [jac_sym])

        self.physics_fn_batch = physics_fn.map(batch_size_phys)
        self.jac_fn_batch = jac_fn.map(batch_size_phys)

    def normalize_x(self, x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1

    def normalize_y(self, y):
        return 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1

    def denormalize_y(self, y_norm):
        return (y_norm + 1) * 0.5 * (self.y_max - self.y_min) + self.y_min

    def forward(self, x):
        x = self.normalize_x(x)
        out, _ = self.rnn(x)

        out_last = out[:, -1, :]
        out = self.fc(out_last)
        out = out.view(-1, self.n_points, 10)
        out = self.denormalize_y(out)

        return out

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

    def physics_rhs_casadi(self, T, V, w):
        # T, V, w: (batch, n_points)
        batch = T.shape[0]

        # entrelaça T,V,w por nó: [T0,V0,w0, T1,V1,w1, ...]
        y_cat = torch.stack([T, V, w], dim=-1).reshape(batch, -1)  # (batch, n_states)

        dydt = PhysicsCasADiFunction.apply(
            y_cat, self.physics_fn_batch, self.jac_fn_batch, self.batch_size_phys
        )  # (batch, n_states)

        dydt = dydt.reshape(batch, self.n_points, 3)

        F_T = dydt[:, :, 0]
        F_V = dydt[:, :, 1]
        F_w = dydt[:, :, 2]

        return F_T, F_V, F_w

    def train_model(self, dt, train_loader, epochs, lambda_phys=1e0, warmup_epochs=0):

        self.train()

        for ep in range(epochs):

            total_loss = 0.0
            total_data = 0.0
            total_phys = 0.0

            for batch_idx, (xb, yb) in enumerate(train_loader):

                self.optimizer.zero_grad()

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                pred = self.forward(xb)

                T_k1 = pred[:, :, 0]
                V_k1 = pred[:, :, 1]
                w_k1 = pred[:, :, 2]
                m_k1 = pred[:, :, 4]

                T_k2 = pred[:, :, 5]
                V_k2 = pred[:, :, 6]
                w_k2 = pred[:, :, 7]
                m_k2 = pred[:, :, 9]

                MM = self.sistema.gas.mixture.MM_m
                A = np.pi * (self.sistema.D**2) / 4
                v_kg_nn = self.V_norm / MM
                rho_nn = 1.0 / v_kg_nn
                m_scale = rho_nn * self.w_norm * A

                if ep < warmup_epochs:
                    loss_phys = torch.tensor(0.0, device=self.device)
                else:
                    T_cat = torch.cat([T_k1, T_k2], dim=0)
                    V_cat = torch.cat([V_k1, V_k2], dim=0)
                    w_cat = torch.cat([w_k1, w_k2], dim=0)

                    F_T_cat, F_V_cat, F_w_cat = self.physics_rhs_casadi(T_cat, V_cat, w_cat)

                    batch = T_k1.shape[0]
                    F_T_k1, F_T_k2 = F_T_cat[:batch], F_T_cat[batch:]
                    F_V_k1, F_V_k2 = F_V_cat[:batch], F_V_cat[batch:]
                    F_w_k1, F_w_k2 = F_w_cat[:batch], F_w_cat[batch:]

                    res_T = (T_k2 - T_k1) / dt - 0.5 * (F_T_k2 + F_T_k1)
                    res_V = (V_k2 - V_k1) / dt - 0.5 * (F_V_k2 + F_V_k1)
                    res_w = (w_k2 - w_k1) / dt - 0.5 * (F_w_k2 + F_w_k1)

                    loss_phys = (
                         (res_T**2).mean()
                        + (res_V**2).mean()
                        +  (res_w**2).mean()
                    )

                w_T = 5e0 / self.T_norm
                w_V = 1e3 / self.V_norm
                w_w = 3e0 / self.w_norm
                w_P = 1e0 / self.P_norm
                w_m = 15e0 / m_scale

                weights = torch.tensor(
                    [w_T, w_V, w_w, w_P, w_m, w_T, w_V, w_w, w_P, w_m],
                    device=self.device
                )

                err_in = (pred - yb) ** 2
                loss_data = (err_in * weights).mean()

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