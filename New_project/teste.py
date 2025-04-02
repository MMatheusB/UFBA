import time
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import fsolve
from joblib import Parallel, delayed
from functools import partial
from libs.compression import *

class MyModel(nn.Module):
    def __init__(self, units, dt, x_max, x_min, y_min, y_max, plenum):
        self.x_max = x_max
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.plenum = plenum
        self.dt = dt
        self.coeff = torch.tensor([11, -18, 9, -2], dtype=torch.float32) / (6 * dt)
        
        super(MyModel, self).__init__()
        
        self.rnn_layer = nn.LSTM(
            input_size=7,
            hidden_size=units,
            batch_first=True,
            bidirectional=False,
            bias=True,
            num_layers=1
        )
        
        self.dense_layers = nn.Sequential(
            nn.Linear(units, 64),
            nn.Tanh(),
            nn.Linear(64, 14),
        )

    def system_residuals(self, y, u0, plenum_sys):
        x = y[:3]
        z = y[3:]
        ode_sym, alg_sym = plenum_sys.evaluate_dae(None, x, z, u0)
        res_ode = np.array([ode_sym[i].item() for i in range(3)])
        res_alg = np.array([alg_sym[i] for i in range(11)])
        return np.concatenate((res_ode, res_alg))

    def compute_steady_state(self, u0, plenum_sys, x0, z0):
        y0 = np.concatenate((x0, z0))
        start_time = time.time()
        sol = fsolve(self.system_residuals, y0, args=(u0, plenum_sys))
        fsolve_time = time.time() - start_time
        print(f"Tempo fsolve: {fsolve_time:.4f}s")
        return sol[:3], sol[3:], fsolve_time
    
    @staticmethod
    def _process_steady_state(args, plenum_sys, self):
        u0, x0, z0 = args
        x_ss, z_ss, fsolve_time = self.compute_steady_state(u0, plenum_sys, x0, z0)
        return x_ss, z_ss, fsolve_time
    
    def compute_steady_state_batch(self, u0_batch, plenum_sys, x0_batch, z0_batch, n_jobs=-1):
        process_fn = partial(self._process_steady_state, plenum_sys=plenum_sys, self=self)
        args_list = list(zip(u0_batch, x0_batch, z0_batch))
        
        start_time = time.time()
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(delayed(process_fn)(args) for args in args_list)
        parallel_time = time.time() - start_time
        
        x_ss_batch, z_ss_batch, fsolve_times = zip(*results)
        avg_fsolve = np.mean(fsolve_times)
        print(f"\nTempo total steady-state batch: {parallel_time:.4f}s")
        print(f"Tempo médio por fsolve: {avg_fsolve:.4f}s")
        print(f"Número de itens processados: {len(args_list)}")
        print(f"Throughput: {len(args_list)/parallel_time:.2f} itens/s\n")
        
        return np.stack(x_ss_batch), np.stack(z_ss_batch)

    @staticmethod
    def _process_gas(args, gas_template):
        y_pred_i, inputs_i, gas_template = args
        start_time = time.time()
        gas = gas_template.copy_change_conditions(y_pred_i[1].item(), None, y_pred_i[2].item(), 'gas')
        gas2 = gas_template.copy_change_conditions(y_pred_i[1].item(), y_pred_i[3].item(), None, 'gas')
        gas.evaluate_der_eos_P()
        proc_time = time.time() - start_time
        return gas2.V.item(), gas.dPdV, gas.dPdT, proc_time
    
    def process_gas_batch(self, y_pred, inputs, gas_template, n_jobs=-1):
        args_list = [(y_pred[i], inputs[i], gas_template) for i in range(y_pred.shape[0])]
        
        start_time = time.time()
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(delayed(self._process_gas)(args, gas_template) for args in args_list)
        parallel_time = time.time() - start_time
        
        Vp, dP_dV, dP_dT, proc_times = zip(*results)
        avg_proc_time = np.mean(proc_times)
        print(f"\nTempo total process_gas_batch: {parallel_time:.4f}s")
        print(f"Tempo médio por item: {avg_proc_time:.4f}s")
        print(f"Número de itens processados: {len(args_list)}")
        print(f"Throughput: {len(args_list)/parallel_time:.2f} itens/s\n")
        
        return (torch.tensor(Vp, dtype=torch.float32), 
                torch.tensor(dP_dV, dtype=torch.float32), 
                torch.tensor(dP_dT, dtype=torch.float32))

    def train_model(self, model, train_loader, val_loader, lr, epochs, optimizers, patience, factor, gas):
        optimizer = optimizers(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience
        )

        # Variáveis para armazenar métricas de tempo
        time_metrics = {
            'forward': [],
            'data_loss': [],
            'derivatives': [],
            'gas_processing': [],
            'steady_state': [],
            'dae_evaluation': [],
            'physics_loss': [],
            'backward': [],
            'batch_total': []
        }

        train_loss_values = []
        val_loss_values = []
        physics_loss_values = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_loss_physics = 0
            total_fsolve_time = 0
            total_gas_time = 0
            total_dae_time = 0

            for batch_idx, (inputs, y_true) in enumerate(train_loader):
                batch_start_time = time.time()
                
                # Forward Pass
                optimizer.zero_grad()
                forward_start = time.time()
                y_pred = self(inputs)
                time_metrics['forward'].append(time.time() - forward_start)

                # Loss de dados
                data_loss_start = time.time()
                loss_data = (
                    1e2 * torch.mean((y_true[:, 0] - y_pred[:, 0]) ** 2) +
                    1e1 * torch.mean((y_true[:, 1] - y_pred[:, 1]) ** 2) +
                    1e-7 * torch.mean((y_true[:, 3] - y_pred[:, 3]) ** 2) +
                    1e-7 * torch.mean((y_true[:, 4] - y_pred[:, 4]) ** 2) +
                    1e-4 * torch.mean((y_true[:, 11] - y_pred[:, 11]) ** 2)
                )
                time_metrics['data_loss'].append(time.time() - data_loss_start)

                # Cálculo das derivadas temporais
                derivatives_start = time.time()
                m_t = torch.sum(self.coeff.view(1, -1) * torch.cat([
                    y_true[:, 0:1], inputs[:, -3:, 0]], dim=1), dim=1)
                t_t = torch.sum(self.coeff.view(1, -1) * torch.cat([
                    y_true[:, 1:2], inputs[:, -3:, 1]], dim=1), dim=1)
                P_t = torch.sum(self.coeff.view(1, -1) * torch.cat([
                    y_true[:, 3:4], inputs[:, -3:, 2]], dim=1), dim=1)
                time_metrics['derivatives'].append(time.time() - derivatives_start)

                # Processamento do gás em paralelo
                gas_start = time.time()
                Vp, dP_dV, dP_dT = self.process_gas_batch(y_pred, inputs, gas)
                time_metrics['gas_processing'].append(time.time() - gas_start)
                total_gas_time += time_metrics['gas_processing'][-1]

                # Cálculo do estado estacionário em paralelo
                steady_state_start = time.time()
                with torch.no_grad():
                    u0_batch = np.stack([
                        np.array([4500, 300, inputs[i, -1, -1].item(), 
                                inputs[i, -1, -2].item(), 5000])
                        for i in range(inputs.shape[0])
                    ])
                    x0_batch = y_pred[:, :3].detach().numpy()
                    z0_batch = y_true[:, 3:].detach().numpy()
                    
                    z_ss = self.compute_steady_state_batch(u0_batch, self.plenum, x0_batch, z0_batch)
                    z_ss = torch.tensor(z_ss, dtype=torch.float32)
                time_metrics['steady_state'].append(time.time() - steady_state_start)
                total_fsolve_time += sum([t for t in time_metrics['steady_state'] if isinstance(t, float)])

                # Avaliação das equações DAE
                dae_start = time.time()
                ode_list = []
                for i in range(inputs.shape[0]):
                    u0 = np.array([4500, 300, inputs[i, -1, -1].item(), 
                                 inputs[i, -1, -2].item(), 5000])
                    x0 = y_pred[i, :3].detach().numpy()
                    z0 = y_true[i, 3:].detach().numpy()
                    
                    start_dae = time.time()
                    ode, _ = self.plenum.evaluate_dae(None, x0, z0, u0)
                    ode_list.append(ode)
                    total_dae_time += time.time() - start_dae
                
                soma_ode = torch.tensor(ode_list, dtype=torch.float32)
                time_metrics['dae_evaluation'].append(time.time() - dae_start)

                # Cálculo das perdas físicas
                physics_start = time.time()
                dVp_dt = (P_t - dP_dT*t_t)/dP_dV
                
                loss_physics_x_mt = torch.mean((soma_ode[:, 0] - m_t)**2)
                loss_physics_t_t = torch.mean((soma_ode[:, 1] - t_t)**2)
                loss_physics_Vp = torch.mean((soma_ode[:, 2] - dVp_dt)**2)
                loss_physics_x = (
                    1e-1 * (loss_physics_x_mt + loss_physics_t_t + loss_physics_Vp) + 
                    5e4 * torch.mean((Vp - y_pred[:, 2])**2)
                )
                
                loss_physics_z = (
                    1e-3 * torch.mean((z_ss[:, 0] - y_pred[:, 3])**2) +
                    1e-6 * torch.mean((z_ss[:, 1] - y_pred[:, 4])**2) +
                    torch.mean((z_ss[:, 2] - y_pred[:, 5])**2) +
                    1e2 * torch.mean((z_ss[:, 3] - y_pred[:, 6])**2) +
                    1e2 * torch.mean((z_ss[:, 4] - y_pred[:, 7])**2) +
                    1e2 * torch.mean((z_ss[:, 5] - y_pred[:, 8])**2) +
                    1e2 * torch.mean((z_ss[:, 6] - y_pred[:, 9])**2) +
                    1e2 * torch.mean((z_ss[:, 7] - y_pred[:, 10])**2) +
                    1e2 * torch.mean((z_ss[:, 8] - y_pred[:, 11])**2) +
                    1e2 * torch.mean((z_ss[:, 9] - y_pred[:, 12])**2) +
                    1e2 * torch.mean((z_ss[:, 10] - y_pred[:, 13])**2)
                )
                
                loss_physics = loss_physics_x + loss_physics_z
                time_metrics['physics_loss'].append(time.time() - physics_start)

                # Backward pass
                backward_start = time.time()
                loss = loss_data + loss_physics
                loss.backward()
                optimizer.step()
                time_metrics['backward'].append(time.time() - backward_start)

                total_loss += loss_data.item()
                total_loss_physics += loss_physics.item()
                time_metrics['batch_total'].append(time.time() - batch_start_time)

                # Log a cada batch
                print(f"\nBatch {batch_idx} Timings:")
                print(f"- Forward pass: {time_metrics['forward'][-1]:.4f}s")
                print(f"- Data loss: {time_metrics['data_loss'][-1]:.4f}s")
                print(f"- Derivatives: {time_metrics['derivatives'][-1]:.4f}s")
                print(f"- Gas processing: {time_metrics['gas_processing'][-1]:.4f}s")
                print(f"- Steady state: {time_metrics['steady_state'][-1]:.4f}s")
                print(f"- DAE evaluation: {time_metrics['dae_evaluation'][-1]:.4f}s")
                print(f"- Physics loss: {time_metrics['physics_loss'][-1]:.4f}s")
                print(f"- Backward pass: {time_metrics['backward'][-1]:.4f}s")
                print(f"Total batch time: {time_metrics['batch_total'][-1]:.4f}s")

            # Resumo da época
            avg_times = {k: np.mean(v) for k, v in time_metrics.items()}
            print(f"\nEpoch {epoch+1} Average Timings:")
            for k, v in avg_times.items():
                print(f"- {k.replace('_', ' ').title()}: {v:.4f}s")
            
            print(f"\nTotal fsolve time: {total_fsolve_time:.2f}s")
            print(f"Total gas processing time: {total_gas_time:.2f}s")
            print(f"Total DAE evaluation time: {total_dae_time:.2f}s")

            # Atualizar scheduler e armazenar métricas
            epoch_loss = total_loss / len(train_loader)
            scheduler.step(epoch_loss)
            train_loss_values.append(epoch_loss)
            physics_loss_values.append(total_loss_physics / len(train_loader))

            # Validação
            val_start = time.time()
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_inputs, val_y_true in val_loader:
                    val_y_pred = model(val_inputs)
                    val_loss += nn.functional.mse_loss(val_y_pred, val_y_true).item()
            
            val_loss /= len(val_loader)
            val_loss_values.append(val_loss)
            print(f"Validation time: {time.time() - val_start:.2f}s")

            print(f"\nEpoch [{epoch + 1}/{epochs}]")
            print(f"Train Loss: {train_loss_values[-1]:.6f}")
            print(f"Physics Loss: {physics_loss_values[-1]:.6f}")
            print(f"Val Loss: {val_loss_values[-1]:.6f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        return train_loss_values, val_loss_values