import time
import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self, units, A1, Lc, kv, P1, P_out, C, dt, x_min, x_max, interpolation):
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.dt = dt
        self.x_min = x_min
        self.x_max = x_max
        self.interpolation = interpolation
        super(MyModel, self).__init__()
        
        # Camada LSTM (não bidirecional)
        self.rnn_layer = nn.LSTM(
            input_size=4,
            hidden_size=units,
            batch_first=True,
            bidirectional=False,  # Unidirecional agora
            bias=True,
        )
        
        # Camadas densas
        self.dense_layers = nn.Sequential(
            nn.Linear(units, 32),  # Não há mais multiplicação por 2
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, inputs):
        # Normalização da entrada
        rnn_input = 2 * (inputs - self.x_min) / (self.x_max - self.x_min) - 1
        
        # Passagem pela camada RNN
        rnn_output, _ = self.rnn_layer(rnn_input)
        
        # Pegando apenas o último passo da sequência
        dense_output = self.dense_layers(rnn_output[:, -1, :])
        
        # Desnormalização da saída
        desnormalizado = ((dense_output + 1) / 2) * (self.x_max[:, :, :2] - self.x_min[:, :, :2]) + self.x_min[:, :, :2]
        return desnormalizado

    
    def train_model(self, model, train_loader, val_loader, lr, epochs, optimizers, patience, factor):
        optimizer = optimizers(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = factor, patience = patience, verbose=True)

        model.train()

        # Listas para armazenar as perdas
        train_loss_values = []
        val_loss_values = []
        phys_loss_values = []
        data_loss_values = []

        for epoch in range(epochs):
            # Treinamento
            model.train()
            total_loss = 0
            total_phys_loss = 0
            total_data_loss = 0

            for inputs, y_true in train_loader:
                optimizer.zero_grad()

                y_pred = model(inputs)
                # Calcular as perdas
                interp = []
                for i in range(inputs.shape[0]):
                    aux = self.interpolation([(inputs[i, -1, -1]).item(), (y_pred[0, i, 0]).item()])
                    interp.append(aux)
                interp = np.array(interp)
                interp = torch.tensor(interp, dtype=torch.float32)

                m_t = (11 * y_pred[:, :, 0] - 18 * inputs[:, -2, 0] + 9 * inputs[:, -3, 0] - 2 * inputs[:, 0, 0]) / (6 * self.dt)
                p_t = (11 * y_pred[:, :, 1] - 18 * inputs[:, -2, 1] + 9 * inputs[:, -2, 1] - 2 * inputs[:, -2, 1]) / (6 * self.dt)

                fLoss_mass = torch.mean(torch.square(m_t - (self.A1 / self.Lc) * ((interp * self.P1) - y_pred[:, :, 1]) * 1e3))
                fLoss_pres = torch.mean(torch.square(
                    p_t - (self.C ** 2) / 2 * (y_pred[:, :, 0] - inputs[:, -1, -2] * self.kv * torch.sqrt(
                        (torch.abs(y_pred[:, :, 1] * 1000 - self.P_out * 1000))))))

                phys_loss = (fLoss_mass + fLoss_pres) * 2e-9
                data_loss = 1e1 * torch.mean((y_true[:, 0, 0] - y_pred[:, :, 0]) ** 2) + 1e2 * torch.mean(
                    (y_true[:, 0, 1] - y_pred[:, :, 1]) ** 2)

                loss = data_loss + phys_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_phys_loss += phys_loss.item()
                total_data_loss += data_loss.item()

            # Atualizar o scheduler
            scheduler.step(total_loss / len(train_loader))

            # Armazenar as médias por época
            train_loss_values.append(total_loss / len(train_loader))
            phys_loss_values.append(total_phys_loss / len(train_loader))
            data_loss_values.append(total_data_loss / len(train_loader))

            # Validação
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_inputs, val_y_true in val_loader:
                    val_y_pred = model(val_inputs)
                    val_data_loss = 1e1 * torch.mean((val_y_true[:, 0, 0] - val_y_pred[:, :, 0]) ** 2) + \
                                    1e2 * torch.mean((val_y_true[:, 0, 1] - val_y_pred[:, :, 1]) ** 2)
                    val_loss += val_data_loss.item()

                val_loss /= len(val_loader)
                val_loss_values.append(val_loss)

            # Log das perdas e LR
            print(f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {train_loss_values[-1]:.6f}, Val Loss: {val_loss_values[-1]:.6f}, "
                f"Phys Loss: {phys_loss_values[-1]:.6f}, Data Loss: {data_loss_values[-1]:.6f}, "
                f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Retornar os valores para posterior plotagem
        return train_loss_values, val_loss_values, phys_loss_values, data_loss_values

            
    def test_model(self, x_test, interval, model):
        model.eval()
        massFlowrate100 = [x_test[0, 0, 0].item(), x_test[0, 1, 0].item(), x_test[0, 2, 0].item()]
        PlenumPressure100 = [x_test[0, 0, 1].item(), x_test[0, 1, 1].item(), x_test[0, 2, 1].item()]

        # Preparar o tensor inicial fora do loop
        input_tensor = torch.zeros((1, 3, 4), dtype=torch.float32)

        # Loop de previsões
        tm1 = time.time()
        for i in range(len(interval)):
            # Atualizar os valores do tensor diretamente
            input_tensor[0, :, 0] = torch.tensor(massFlowrate100[-3:])
            input_tensor[0, :, 1] = torch.tensor(PlenumPressure100[-3:])
            input_tensor[0, :, 2] = x_test[i, :, 2]
            input_tensor[0, :, 3] = x_test[i, :, 3]
            print(input_tensor.shape)
            # Previsão com desativação do gradiente
            with torch.no_grad():
                prediction100 = model(input_tensor)

            # Adicionar previsões diretamente
            massFlowrate100.append(prediction100[0, 0, 0].item())
            PlenumPressure100.append(prediction100[0, 0, 1].item())
        tm2 = time.time()
        timeteste = tm2 - tm1
        model.train()
        return massFlowrate100, PlenumPressure100, timeteste