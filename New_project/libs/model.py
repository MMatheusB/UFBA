import time
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import fsolve

class MyModel(nn.Module):
    def __init__(self, units, dt, x_max, x_min, y_min, y_max, plenum):
        
        self.x_max = x_max
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.plenum = plenum
        self.dt = dt
        
        super(MyModel, self).__init__()
        
        self.rnn_layer = nn.LSTM(
            input_size=7,
            hidden_size=units,
            batch_first=True,
            bidirectional=False,
            bias=True,
            num_layers= 1
        )
        
        self.dense_layers = nn.Sequential(
            nn.Linear(units, 64),
            nn.Tanh(),
            nn.Linear(64, 14),
        )
    
    def normalize(self, inputs, x_min, x_max):
        return 2 * (inputs - x_min) / (x_max - x_min) - 1
    
    def desnormalize(self, inputs, y_min, y_max):
        return ((inputs + 1) / 2) * (y_max - y_min) + y_min
    
    def system_residuals(self, y, u0, plenum_sys):
    # Separa as variáveis
        x = y[:3]
        z = y[3:]
    
    # Substitua as expressões abaixo pelas suas equações do modelo em estado estacionário.
        ode_sym, alg_sym = plenum_sys.evaluate_dae(None, x, z, u0)
    
        res_ode = np.array([ode_sym[i].item() for i in range(3)])
    
    # Calcula os resíduos das equações algébricas
        res_alg = np.array([alg_sym[i] for i in range(11)])

        res = np.concatenate((res_ode, res_alg))
        return res

    def compute_steady_state(self, u0, plenum_sys, x0, z0):
    # Vetor inicial concatenado
        y0 = np.concatenate((x0, z0))
    
    # Chama o fsolve para encontrar os zeros da função de resíduos
        sol = fsolve(self.system_residuals, y0, args=(u0, plenum_sys))
    
    # Separa a solução em x e z
        x_ss = sol[:3]
        z_ss = sol[3:]
        return x_ss, z_ss
    
    def forward(self, inputs):
        rnn_inputs = self.normalize(inputs, self.x_min, self.x_max)
        rnn_output, _ = self.rnn_layer(rnn_inputs)
        dense_output = self.dense_layers(rnn_output[:, -1, :])
        
        # Desnormalizando primeiro
        desnormalized_output = self.desnormalize(dense_output, self.y_min, self.y_max)
        
        # Limitando os valores dentro do intervalo permitido
        clamped_output = torch.clamp(desnormalized_output, min=self.y_min, max=self.y_max)
        
        return desnormalized_output
    
    def train_model(self, model, train_loader, val_loader, lr, epochs, optimizers, patience, factor, gas):
        optimizer = optimizers(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = factor, patience = patience, verbose=True)

        model.train()

        # Listas para armazenar as perdas
        train_loss_values = []
        val_loss_values = []
        physics_loss_values = []
        for epoch in range(epochs):
            # Treinamento
            model.train()
            total_loss = 0
            total_loss_physics = 0
            for inputs, y_true in train_loader:
                optimizer.zero_grad()
                y_pred = self(inputs)
                loss_data =  torch.mean((y_true[:, 0] - y_pred[:, 0]) ** 2) + \
                            1e-3  * torch.mean((y_true[:, 1] - y_pred[:, 1]) ** 2) + \
                            1e-7 * torch.mean((y_true[:, 3] - y_pred[:, 3]) ** 2) + \
                            1e-7 * torch.mean((y_true[:, 4] - y_pred[:, 4]) ** 2) + \
                            1e-4 *  torch.mean((y_true[:, 11] - y_pred[:, 11]) ** 2)
                soma_ode = torch.zeros(inputs.shape[0], 3, dtype=torch.float32)
                alg1 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg2 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg3 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg4 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg5 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg6 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg7 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg8 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg9 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg10 = torch.zeros(inputs.shape[0], dtype=torch.float32)
                alg11 = torch.zeros(inputs.shape[0], dtype=torch.float32)

                m_t = (11 * y_true[:, 0] - 18 * inputs[:, 2, 0] + 9 * inputs[:, 1, 0] - 2 * inputs[:, 0, 0]) / (6 * self.dt)
                t_t = (11 * y_true[:, 1] - 18 * inputs[:, 2, 1] + 9 * inputs[:, 1, 1] - 2 * inputs[:, 0, 1]) / (6 * self.dt)
                P_t = (11 * y_true[:, 3] - 18 * inputs[:, 2, 2] + 9 * inputs[:, 1, 2] - 2 * inputs[:, 0, 2]) / (6 * self.dt)
                dP_dV = torch.zeros(inputs.shape[0], dtype=torch.float32)
                dP_dT = torch.zeros(inputs.shape[0], dtype=torch.float32)
                Vp = torch.zeros(inputs.shape[0], dtype=torch.float32)
                for i in range(inputs.shape[0]):
                    gas = gas.copy_change_conditions(y_pred[i, 1].detach().numpy(), None, y_pred[i, 2].detach().numpy(), 'gas')
                    gas2 = gas.copy_change_conditions(y_pred[i, 1].detach().numpy(), y_pred[i, 3].detach().numpy(), None , 'gas')
                    gas.evaluate_der_eos_P()
                    Vp[i] = gas2.V.item()
                    dP_dV[i] = gas.dPdV
                    dP_dT[i] = gas.dPdT
                    u0 = [4500, 300, inputs[i, -1, -1], inputs[i, -1, -2], 5000]
                    x0 = [y_pred[i, 0].detach().numpy(), y_pred[i, 1].detach().numpy(), y_pred[i, 2].detach().numpy()]
                    z0 = [y_true[i, 3].detach().numpy(), y_true[i, 4].detach().numpy(), y_true[i, 5].detach().numpy(), y_true[i, 6].detach().numpy(), y_true[i,7].detach().numpy(),
                          y_true[i, 8].detach().numpy(), y_true[i, 9].detach().numpy(), y_true[i, 10].detach().numpy(),y_true[i, 11].detach().numpy(), y_true[i, 12].detach().numpy(),
                          y_true[i, 13].detach().numpy()]
                    u0 = np.array(u0)
                    x0 = np.array(x0)
                    z0 = np.array(z0)
                    ode, alg = self.plenum.evaluate_dae(None, x0, z0, u0)
                    ode = torch.tensor(ode, dtype=torch.float32)
                    soma_ode[i] = ode
                    x_ss, z_ss = self.compute_steady_state(u0, self.plenum, x0, z0)
                    alg1[i] = z_ss[0]
                    alg2[i] = z_ss[1]
                    alg3[i] = z_ss[2]
                    alg4[i] = z_ss[3]
                    alg5[i] = z_ss[4]
                    alg6[i] = z_ss[5]
                    alg7[i] = z_ss[6]
                    alg8[i] = z_ss[7]
                    alg9[i] = z_ss[8]
                    alg10[i] = z_ss[9]
                    alg11[i] = z_ss[10]

                    
                dVp_dt = (P_t - dP_dT*t_t)/dP_dV 
                loss_physics_x_mt = torch.mean((soma_ode[:, 0] - m_t)**2)
                loss_physics_t_t = torch.mean((soma_ode[:, 1] - t_t)**2)
                loss_physics_Vp = torch.mean(((soma_ode[:, 2] - dVp_dt)**2))
                loss_physics_x = loss_physics_x_mt + loss_physics_t_t + loss_physics_Vp + 1e4* torch.mean((Vp - y_pred[:, 2])**2)
                loss_physics_z = 1e-4 * torch.mean((alg1 - y_pred[:, 3])**2) + 1e-7 * torch.mean((alg2 - y_pred[:, 4])**2) + torch.mean((alg3 - y_pred[:, 5])**2) + 1e2*torch.mean((alg4 - y_pred[:, 6])**2) + \
                                 torch.mean((alg5 - y_pred[:, 7])**2) + torch.mean((alg6 - y_pred[:, 8])**2) + torch.mean((alg7 - y_pred[:, 9])**2) + torch.mean((alg8 - y_pred[:, 10])**2) + \
                                 torch.mean((alg9 - y_pred[:, 11])**2) + torch.mean((alg10 - y_pred[:, 12])**2) + torch.mean((alg11 - y_pred[:, 13])**2)
                loss_physics = loss_physics_x + loss_physics_z
                loss =  loss_data + loss_physics
                loss.backward()  
                optimizer.step()

                total_loss += loss_data.item()
                total_loss_physics += loss_physics.item() 
                print(loss_data, loss_physics_z, loss_physics_x_mt, loss_physics_t_t, loss_physics_Vp, torch.mean((Vp - y_pred[:, 2])**2))
            # Atualizar o scheduler
            scheduler.step(total_loss / len(train_loader))

            train_loss_values.append(total_loss / len(train_loader))
            physics_loss_values.append(total_loss_physics / len(train_loader))
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_inputs, val_y_true in val_loader:
                    val_y_pred = model(val_inputs)
                    val_data_loss = nn.functional.mse_loss(val_y_pred, val_y_true)
                    val_loss += val_data_loss.item()

                val_loss /= len(val_loader)
                val_loss_values.append(val_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {train_loss_values[-1]:.6f}, Val Loss: {val_loss_values[-1]:.6f}"
                f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}", f"physics_z: {physics_loss_values[-1]:.6f}")


        return train_loss_values, val_loss_values