import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MyModel(nn.Module):
    def __init__(self, units, A1, Lc, kv, P1, P_out, C, dt, x_min, x_max, interpolation):
        super(MyModel, self).__init__()
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

        # Camada LSTM
        self.rnn_layer = nn.LSTM(
            input_size=4,
            hidden_size=units,
            batch_first=True,
            bidirectional=False,
            bias=True,
        )

        # Camadas densas
        self.dense_layers = nn.Sequential(
            nn.Linear(units, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

        # Pesos adaptativos iniciais
        self.log_weight_uq = nn.Parameter(torch.tensor(np.log(10)))  # Peso inicial 10.0
        self.log_weight_f = nn.Parameter(torch.tensor(np.log(10)))  # Peso inicial 10.0

    def forward(self, inputs):
        # Normalização da entrada
        rnn_input = 2 * (inputs - self.x_min) / (self.x_max - self.x_min) - 1

        # Passagem pela camada RNN
        rnn_output, _ = self.rnn_layer(rnn_input)

        # Pegando apenas o último passo da sequência
        dense_output = self.dense_layers(rnn_output[:, -1, :])

        # Desnormalização da saída
        desnormalizado = ((dense_output + 1) / 2) * (
            self.x_max[:, :, :2] - self.x_min[:, :, :2]
        ) + self.x_min[:, :, :2]
        return desnormalizado

    def train_model(self, train_loader, val_loader, lr, epochs, patience, factor):
        optimizer = optim.Adam(
            [
                {"params": self.parameters()},  # Inclui pesos adaptativos
            ],
            lr=lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, verbose=True
        )

        train_loss_values, val_loss_values = [], []
        phys_loss_values, data_loss_values = [], []
        weights_uq, weights_f = [], []

        for epoch in range(epochs):
            self.train()
            total_loss, total_phys_loss, total_data_loss = 0, 0, 0

            for inputs, y_true in train_loader:
                optimizer.zero_grad()

                y_pred = self(inputs)

                interp = []
                for i in range(inputs.shape[0]):
                    aux = self.interpolation([(inputs[i, -1, -1]).item(), (y_true[i, 0, 0]).item()])
                    interp.append(aux)
                interp = np.array(interp)
                interp = torch.tensor(interp, dtype=torch.float32)

                # Cálculo das perdas físicas
                m_t = (
                    11 * y_pred[:, :, 0]
                    - 18 * inputs[:, -2, 0]
                    + 9 * inputs[:, -3, 0]
                    - 2 * inputs[:, 0, 0]
                ) / (6 * self.dt)
                p_t = (
                    11 * y_pred[:, :, 1]
                    - 18 * inputs[:, -2, 1]
                    + 9 * inputs[:, -3, 1]
                    - 2 * inputs[:, 0, 1]
                ) / (6 * self.dt)
                fLoss_mass = torch.mean(
                    torch.square(
                        m_t
                        - (self.A1 / self.Lc) * ((interp * self.P1) - y_true[:, :, 1])
                        * 1e3
                    )
                )
                press_l = ((self.C**2) / 2) * (
                    y_true[:, :, 0]
                    - inputs[:, -1, -2]
                    * self.kv
                    * (
                        torch.sqrt(y_true[:, :, 1] * 1e3 - self.P_out * 1e3)
                    ).clamp(min=0)
                )
                fLoss_pres = torch.mean(torch.square(p_t - press_l * 1e-11))
                phys_loss = (fLoss_mass + fLoss_pres)*1e-2

                # Cálculo das perdas de dados
                data_loss = (
                    torch.mean(1e1 * (y_true[:, 0, 0] - y_pred[:, :, 0]) ** 2)
                    + 1e2 * torch.mean((y_true[:, 0, 1] - y_pred[:, :, 1]) ** 2)
                )

                # Perda total com pesos adaptativos
                loss = (
                    0.5 * torch.exp(-self.log_weight_f.clamp(min=-1)) * phys_loss
                    + 0.5 * torch.exp(-self.log_weight_uq.clamp(min=-1)) * data_loss
                    + self.log_weight_f.clamp(min=0)
                    + self.log_weight_uq.clamp(min=0)
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_phys_loss += phys_loss.item()
                total_data_loss += data_loss.item()

            # Atualizar scheduler
            scheduler.step(total_loss / len(train_loader))

            # Registrar valores dos pesos
            weights_uq.append(torch.exp(self.log_weight_uq).item())
            weights_f.append(torch.exp(self.log_weight_f).item())

            train_loss_values.append(total_loss / len(train_loader))
            phys_loss_values.append(total_phys_loss / len(train_loader))
            data_loss_values.append(total_data_loss / len(train_loader))

            # Validação
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for val_inputs, val_y_true in val_loader:
                    val_y_pred = self(val_inputs)
                    val_loss += (
                        torch.mean(
                            (val_y_true[:, 0, 0] - val_y_pred[:, :, 0]) ** 2
                        ).item()
                        + torch.mean(
                            (val_y_true[:, 0, 1] - val_y_pred[:, :, 1]) ** 2
                        ).item()
                    )

                val_loss /= len(val_loader)
                val_loss_values.append(val_loss)

            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss_values[-1]:.6f}, "
                f"Val Loss: {val_loss_values[-1]:.6f}, "
                f"Weights: uq={weights_uq[-1]:.6f}, f={weights_f[-1]:.6f}"
            )

        return train_loss_values, val_loss_values, phys_loss_values, data_loss_values, weights_uq, weights_f

    def test_model(self, x_test, interval):
        self.eval()
        massFlowrate100 = [
            x_test[0, 0, 0].item(),
            x_test[0, 1, 0].item(),
            x_test[0, 2, 0].item(),
        ]
        PlenumPressure100 = [
            x_test[0, 0, 1].item(),
            x_test[0, 1, 1].item(),
            x_test[0, 2, 1].item(),
        ]

        input_tensor = torch.zeros((1, 3, 4), dtype=torch.float32)

        tm1 = time.time()
        for i in range(len(interval)):
            input_tensor[0, :, 0] = torch.tensor(massFlowrate100[-3:])
            input_tensor[0, :, 1] = torch.tensor(PlenumPressure100[-3:])
            input_tensor[0, :, 2] = x_test[i, :, 2]
            input_tensor[0, :, 3] = x_test[i, :, 3]

            with torch.no_grad():
                prediction100 = self(input_tensor)

            massFlowrate100.append(prediction100[0, 0, 0].item())
            PlenumPressure100.append(prediction100[0, 0, 1].item())

        tm2 = time.time()
        return massFlowrate100, PlenumPressure100, tm2 - tm1
