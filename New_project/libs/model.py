import time
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import fsolve
from joblib import Parallel, delayed
from functools import partial

class MyModel(nn.Module):
    def __init__(self, units, dt, x_max, x_min, y_min, y_max, plenum):
        super(MyModel, self).__init__()

        self.x_max = x_max
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.plenum = plenum
        self.dt = dt
        self.coeff = torch.tensor([11, -18, 9, -2], dtype=torch.float32) / (6 * dt)
        
        # A LSTM espera entrada de dimensão 7 por exemplo.
        self.lstm = nn.LSTM(input_size=7, hidden_size=100, batch_first=True)
        # A camada de saída gera 14 features
        self.output_layer = nn.Linear(100, 14)
    
    def forward(self, inputs):
        # Normaliza a entrada
        normalized_inputs = self.normalize(inputs, self.x_min, self.x_max)
        # Passa pela LSTM (batch, seq_len, 7)
        lstm_out, (hidden, cell) = self.lstm(normalized_inputs)
        # Usa o último estado oculto para gerar a saída (batch, 14)
        output = self.output_layer(hidden[-1])
        # Desnormaliza a saída
        desnormalized_output = self.desnormalize(output, self.y_min, self.y_max)
        return desnormalized_output

    def normalize(self, inputs, x_min, x_max):
        # Normalização simples no intervalo [-1, 1]
        return 2 * (inputs - x_min) / (x_max - x_min) - 1

    def desnormalize(self, inputs, y_min, y_max):
        return ((inputs + 1) / 2) * (y_max - y_min) + y_min

    def system_residuals(self, z, x0, u0, plenum_sys):
        """Calcula apenas os resíduos algébricos, mantendo x0 fixo"""
        _, alg_sym = plenum_sys.evaluate_dae(None, x0, z, u0)
        return np.array([alg_sym[i] for i in range(11)])  # Retorna apenas as equações algébricas

    def compute_steady_state(self, u0, plenum_sys, x0, z0):
        """Calcula o estado estacionário das variáveis algébricas"""
        sol = fsolve(self.system_residuals, z0, args=(x0, u0, plenum_sys))
        return x0, sol
    
    @staticmethod
    def _process_steady_state(args, plenum_sys, self):
        u0, x0, z0 = args
        return self.compute_steady_state(u0, plenum_sys, x0, z0)
    
    def compute_steady_state_batch(self, u0_batch, plenum_sys, x0_batch, z0_batch, n_jobs=-1):
        process_fn = partial(self._process_steady_state, plenum_sys=plenum_sys, self=self)
        args_list = list(zip(u0_batch, x0_batch, z0_batch))
        
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(delayed(process_fn)(args) for args in args_list)
        
        x_ss_batch, z_ss_batch = zip(*results)
        return np.stack(x_ss_batch), np.stack(z_ss_batch)
    
    @staticmethod
    def _process_gas(args, gas_template):
        y_pred_i, inputs_i, gas_template = args
        gas = gas_template.copy_change_conditions(y_pred_i[1].item(), None, y_pred_i[2].item(), 'gas')
        gas2 = gas_template.copy_change_conditions(y_pred_i[1].item(), y_pred_i[3].item(), None, 'gas')
        gas.evaluate_der_eos_P()
        return gas2.V.item(), gas.dPdV, gas.dPdT
    
    def process_gas_batch(self, y_pred, inputs, gas_template, n_jobs=-1):
        args_list = [(y_pred[i], inputs[i], gas_template) for i in range(y_pred.shape[0])]
        
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(delayed(self._process_gas)(args, gas_template) for args in args_list)
        
        Vp, dP_dV, dP_dT = zip(*results)
        return (torch.tensor(Vp, dtype=torch.float32), 
                torch.tensor(dP_dV, dtype=torch.float32), 
                torch.tensor(dP_dT, dtype=torch.float32))
    
    def generate_free_inputs(self, initial_input, steps=None):
        """
        Gera entradas em modo livre (autoregressivo) para o professor forcing.
        A lógica é:
          - A entrada da rede tem 7 features.
          - Os 5 primeiros são referentes às variáveis que o modelo prevê (usando índices [0], [1], [3], [4] e [11]),
            enquanto os 2 últimos (índices 5 e 6) são controles constantes, extraídos da entrada original.
        """
        # initial_input: tensor de shape (batch, seq_len, 7)
        batch = initial_input.size(0)
        if steps is None:
            steps = initial_input.size(1)
        # Extraímos os controles (supondo que eles estejam nas colunas 5 e 6)
        control = initial_input[:, 0, 5:]  # shape: (batch, 2)
        # Usamos o último instante da entrada teacher-forcing como ponto de partida
        current_input = initial_input[:, -1, :].clone()  # shape: (batch, 7)
        free_inputs = [current_input]
        
        with torch.no_grad():
            for _ in range(steps - 1):
                # A entrada para o modelo deve ter 7 features
                pred = self(current_input.unsqueeze(1))  # pred shape: (batch, 14)
                # Constrói nova entrada usando os outputs relevantes:
                # new_input[0] = pred[:, 0]
                # new_input[1] = pred[:, 1]
                # new_input[2] = pred[:, 3]
                # new_input[3] = pred[:, 4]
                # new_input[4] = pred[:, 11]
                # new_input[5:7] = controle constante (do input original)
                new_input = torch.cat([
                    pred[:, [0, 1, 3, 4, 11]],  # shape: (batch, 5)
                    control                     # shape: (batch, 2)
                ], dim=1)  # new_input shape: (batch, 7)
                free_inputs.append(new_input)
                current_input = new_input
        
        # Retorna a sequência de entradas em free run (batch, steps, 7)
        return torch.stack(free_inputs, dim=1)

    def train_model(self, model, train_loader, val_loader, lr, epochs, optimizers, patience, factor, gas):
        optimizer = optimizers(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience
        )

        model.train()
        train_loss_values = []
        val_loss_values = []
        physics_loss_values = []
        # Peso do loss do professor forcing (valor ajustável)
        alpha = 0.1

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_loss_physics = 0
            
            for batch_idx, (inputs, y_true) in enumerate(train_loader):
                start_time = time.time()
                optimizer.zero_grad()
                
                # ============================
                # Passagem Teacher Forcing
                # ============================
                y_pred_teacher = self(inputs)  # saída com teacher forcing (batch, 14)
                
                # ============================
                # Passagem Free-Running (autoregressivo)
                # ============================
                # Gera entradas free-running com o mesmo número de steps que a sequência original
                free_inputs = self.generate_free_inputs(inputs, steps=inputs.size(1))  # shape: (batch, seq_len, 7)
                y_pred_free = self(free_inputs)  # saída free-run (batch, 14)
                
                # ============================
                # Loss do Professor Forcing
                # Penalizamos a diferença entre a saída teacher forced e a free-run para os índices relevantes
                # ============================
                indices = [0, 1, 3, 4, 11]
                prof_loss = 0.0
                for i in indices:
                    prof_loss += nn.functional.mse_loss(y_pred_teacher[:, i], y_pred_free[:, i])
                prof_loss = prof_loss / len(indices)
                
                # ============================
                # Loss de dados (já presente)
                # ============================
                loss_data = 2*(
                    1e2 * torch.mean((y_true[:, 0] - y_pred_teacher[:, 0]) ** 2) +
                    1e2 * torch.mean((y_true[:, 1] - y_pred_teacher[:, 1]) ** 2) +
                    1e-5 * torch.mean((y_true[:, 3] - y_pred_teacher[:, 3]) ** 2) +
                    1e-5 * torch.mean((y_true[:, 4] - y_pred_teacher[:, 4]) ** 2) +
                    1e2 * torch.mean((y_true[:, 11] - y_pred_teacher[:, 11]) ** 2)
                )
                
                # ============================
                # Cálculo das derivadas temporais (vetorizado)
                # ============================
                m_t = torch.sum(self.coeff.view(1, -1) * torch.cat([
                    y_true[:, 0:1],
                    inputs[:, -3:, 0]
                ], dim=1), dim=1)
                
                t_t = torch.sum(self.coeff.view(1, -1) * torch.cat([
                    y_true[:, 1:2],
                    inputs[:, -3:, 1]
                ], dim=1), dim=1)
                
                P_t = torch.sum(self.coeff.view(1, -1) * torch.cat([
                    y_true[:, 3:4],
                    inputs[:, -3:, 2]
                ], dim=1), dim=1)
                
                # ============================
                # Processamento paralelo das propriedades do gás
                # ============================
                Vp, dP_dV, dP_dT = self.process_gas_batch(y_pred_teacher, inputs, gas)
                
                # ============================
                # Cálculo do estado estacionário (free-run) em paralelo
                # ============================
                with torch.no_grad():
                    u0_batch = np.stack([
                        np.array([4500, 300, inputs[i, -1, -1].item(),
                                  inputs[i, -1, -2].item(), 5000])
                        for i in range(inputs.shape[0])
                    ])
                    x0_batch = y_pred_teacher[:, :3].detach().numpy()
                    z0_batch = y_true[:, 3:].detach().numpy()
                    
                    x_ss, z_ss = self.compute_steady_state_batch(u0_batch, self.plenum, x0_batch, z0_batch)
                    z_ss = torch.tensor(z_ss, dtype=torch.float32)
                
                # ============================
                # Cálculo das perdas físicas
                # ============================
                dVp_dt = (P_t - dP_dT*t_t) / dP_dV
                
                # Avaliação das equações DAE em paralelo
                ode_list = []
                for i in range(inputs.shape[0]):
                    u0 = np.array([4500, 300, inputs[i, -1, -1].item(),
                                   inputs[i, -1, -2].item(), 5000])
                    x0 = y_pred_teacher[i, :3].detach().numpy()
                    z0 = y_true[i, 3:].detach().numpy()
                    ode, _ = self.plenum.evaluate_dae(None, x0, z0, u0)
                    ode_list.append(ode)
                
                soma_ode = torch.tensor(ode_list, dtype=torch.float32)
                
                loss_physics_x_mt = torch.mean((soma_ode[:, 0] - m_t)**2)
                loss_physics_t_t = torch.mean((soma_ode[:, 1] - t_t)**2)
                loss_physics_Vp = torch.mean((soma_ode[:, 2] - dVp_dt)**2)
                loss_physics_x = (
                    1e-1 * (loss_physics_x_mt + loss_physics_t_t + loss_physics_Vp) +
                    5e4 * torch.mean((Vp - y_pred_teacher[:, 2])**2)
                )
                
                loss_physics_z = 1e-3*(
                    torch.mean((z_ss[:, 2] - y_pred_teacher[:, 5])**2) +
                    1e2 * torch.mean((z_ss[:, 3] - y_pred_teacher[:, 6])**2) +
                    1e2 * torch.mean((z_ss[:, 4] - y_pred_teacher[:, 7])**2) +
                    1e2 * torch.mean((z_ss[:, 5] - y_pred_teacher[:, 8])**2) +
                    1e2 * torch.mean((z_ss[:, 6] - y_pred_teacher[:, 9])**2) +
                    1e2 * torch.mean((z_ss[:, 7] - y_pred_teacher[:, 10])**2) +
                    1e0 * torch.mean((z_ss[:, 8] - y_pred_teacher[:, 11])**2) +
                    1e2 * torch.mean((z_ss[:, 9] - y_pred_teacher[:, 12])**2) +
                    1e2 * torch.mean((z_ss[:, 10] - y_pred_teacher[:, 13])**2)
                )
                
                loss_physics = loss_physics_x + loss_physics_z
                
                # ============================
                # Loss total: dados + físicas + professor forcing
                # ============================
                loss = loss_data + loss_physics + alpha * prof_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss_data.item()
                total_loss_physics += loss_physics.item()
                
                print(f"Batch {batch_idx} completed in {time.time() - start_time:.2f}s | "
                      f"Data: {loss_data.item():.4f} | Phys: {loss_physics.item():.4f} | Prof: {prof_loss.item():.4f}")

            scheduler.step(total_loss / len(train_loader))
            train_loss_values.append(total_loss / len(train_loader))
            physics_loss_values.append(total_loss_physics / len(train_loader))
            
            # Validação
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_inputs, val_y_true in val_loader:
                    val_y_pred = model(val_inputs)
                    val_loss += nn.functional.mse_loss(val_y_pred, val_y_true).item()
            
            val_loss /= len(val_loader)
            val_loss_values.append(val_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {train_loss_values[-1]:.6f}, Val Loss: {val_loss_values[-1]:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}, Phys Loss: {physics_loss_values[-1]:.6f}")

        return train_loss_values, val_loss_values, physics_loss_values
