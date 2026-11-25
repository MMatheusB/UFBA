import time
import torch
import torch.nn as nn
import numpy as np


class MyModel(nn.Module):

    def __init__(
        self, units, A1, Lc, kv, P1, P_out,
        C, dt, x_min, x_max, interpolation
    ):

        super(MyModel, self).__init__()

        # Parâmetros físicos
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.dt = dt

        self.interpolation = interpolation

        # Parâmetros de normalização
        self.register_buffer(
            "x_min",
            torch.as_tensor(
                x_min, dtype=torch.float32
            ).reshape(1, 1, 4)
        )

        self.register_buffer(
            "x_max",
            torch.as_tensor(
                x_max, dtype=torch.float32
            ).reshape(1, 1, 4)
        )

        # Camada LSTM
        self.rnn_layer = nn.LSTM(
            input_size=4,
            hidden_size=units,
            batch_first=True,
            bidirectional=False,
            bias=True
        )

        # Camadas densas
        self.dense_layers = nn.Sequential(
            nn.Linear(units, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 2)
        )

    # ==================================================
    # FORWARD
    # ==================================================

    def forward(self, inputs):

        # Normalização das entradas entre -1 e 1
        rnn_input = (
            2 * (inputs - self.x_min)
            / (self.x_max - self.x_min)
            - 1
        )

        # Passagem pela LSTM
        rnn_output, _ = self.rnn_layer(rnn_input)

        # Último instante da sequência
        dense_output = self.dense_layers(
            rnn_output[:, -1, :]
        )

        # Dimensão: [batch_size, 1, 2]
        dense_output = dense_output.unsqueeze(1)

        # Desnormalização das saídas
        desnormalizado = (
            (dense_output + 1) / 2
        ) * (
            self.x_max[:, :, :2]
            - self.x_min[:, :, :2]
        ) + self.x_min[:, :, :2]

        return desnormalizado

    # ==================================================
    # FUNÇÃO DE PERDA DOS DADOS
    # ==================================================

    def data_loss_function(self, y_pred, y_true):

        m_pred = y_pred[:, 0, 0]
        p_pred = y_pred[:, 0, 1]

        m_true = y_true[:, 0, 0]
        p_true = y_true[:, 0, 1]

        loss_mass = torch.mean(
            (m_true - m_pred) ** 2
        )

        loss_pres = torch.mean(
            (p_true - p_pred) ** 2
        )

        data_loss = (
            1e1 * loss_mass
            + 1e2 * loss_pres
        )

        return data_loss

    # ==================================================
    # FUNÇÃO DE PERDA FÍSICA
    # ==================================================

    def physics_loss_function(self, inputs, y_pred):

        # Estados previstos pela rede
        m_pred = y_pred[:, 0, 0]
        p_pred = y_pred[:, 0, 1]

        # Variáveis manipuladas
        alpha = inputs[:, -1, 2]
        N = inputs[:, -1, 3]

        # ----------------------------------------------
        # Interpolação do mapa do compressor
        # ----------------------------------------------

        interp = []

        for i in range(inputs.shape[0]):

            aux = self.interpolation([
                N[i].detach().item(),
                m_pred[i].detach().item()
            ])

            interp.append(aux)

        interp = torch.as_tensor(
            np.asarray(interp),
            dtype=y_pred.dtype,
            device=y_pred.device
        ).reshape(-1)

        # ----------------------------------------------
        # Derivadas temporais - BDF3
        # ----------------------------------------------

        # Derivada da vazão mássica
        m_t = (
            11 * m_pred
            - 18 * inputs[:, -1, 0]
            + 9 * inputs[:, -2, 0]
            - 2 * inputs[:, -3, 0]
        ) / (6 * self.dt)

        # Derivada da pressão
        p_t = (
            11 * p_pred
            - 18 * inputs[:, -1, 1]
            + 9 * inputs[:, -2, 1]
            - 2 * inputs[:, -3, 1]
        ) / (6 * self.dt)

        # Conversão de kPa/s para Pa/s
        p_t = p_t * 1e3

        # ----------------------------------------------
        # Equação governante da vazão mássica
        # ----------------------------------------------

        f_mass = (
            self.A1 / self.Lc
        ) * (
            interp * self.P1 - p_pred
        ) * 1e3

        # ----------------------------------------------
        # Equação governante da pressão
        # ----------------------------------------------

        # Diferencial de pressão em Pa
        delta_p = (
            p_pred - self.P_out
        ) * 1e3

        # Proteção contra raiz quadrada negativa
        delta_p = torch.clamp(
            delta_p,
            min=1e-8
        )

        # Vazão mássica na válvula
        m_valve = (
            alpha
            * self.kv
            * torch.sqrt(delta_p)
        )

        # Equação governante da pressão
        f_pres = (
            self.C ** 2 / 2
        ) * (
            m_pred - m_valve
        )

        # ----------------------------------------------
        # Resíduos físicos
        # ----------------------------------------------

        fLoss_mass = torch.mean(
            (m_t - f_mass) ** 2
        )

        fLoss_pres = torch.mean(
            (p_t - f_pres) ** 2
        )

        # Perda física total
        phys_loss = (
            fLoss_mass + fLoss_pres
        ) * 2e-9

        return phys_loss

    # ==================================================
    # TREINAMENTO
    # ==================================================

    def train_model(
        self,
        model,
        train_loader,
        val_loader,
        lr,
        epochs,
        optimizers,
        patience,
        factor
    ):

        optimizer = optimizers(
            model.parameters(),
            lr=lr
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience
        )

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Histórico das perdas
        train_loss_values = []
        val_loss_values = []
        phys_loss_values = []
        data_loss_values = []

        for epoch in range(epochs):

            # ==========================================
            # TREINAMENTO
            # ==========================================

            model.train()

            total_loss = 0
            total_phys_loss = 0
            total_data_loss = 0

            for inputs, y_true in train_loader:

                inputs = inputs.to(
                    device=device,
                    dtype=dtype
                )

                y_true = y_true.to(
                    device=device,
                    dtype=dtype
                )

                optimizer.zero_grad()

                # Previsão da rede neural
                y_pred = model(inputs)

                # Perda dos dados
                data_loss = model.data_loss_function(
                    y_pred,
                    y_true
                )

                # Perda física
                phys_loss = model.physics_loss_function(
                    inputs,
                    y_pred
                )

                # Perda total
                loss = data_loss + phys_loss

                # Retropropagação
                loss.backward()

                # Atualização dos parâmetros
                optimizer.step()

                # Acumular perdas
                total_loss += loss.item()

                total_phys_loss += phys_loss.item()

                total_data_loss += data_loss.item()

            # Médias das perdas por época
            mean_train_loss = (
                total_loss / len(train_loader)
            )

            mean_phys_loss = (
                total_phys_loss / len(train_loader)
            )

            mean_data_loss = (
                total_data_loss / len(train_loader)
            )

            # Atualização da taxa de aprendizado
            scheduler.step(mean_train_loss)

            # Armazenar perdas
            train_loss_values.append(
                mean_train_loss
            )

            phys_loss_values.append(
                mean_phys_loss
            )

            data_loss_values.append(
                mean_data_loss
            )

            # ==========================================
            # VALIDAÇÃO
            # ==========================================

            model.eval()

            val_loss = 0

            with torch.no_grad():

                for val_inputs, val_y_true in val_loader:

                    val_inputs = val_inputs.to(
                        device=device,
                        dtype=dtype
                    )

                    val_y_true = val_y_true.to(
                        device=device,
                        dtype=dtype
                    )

                    # Previsão
                    val_y_pred = model(val_inputs)

                    # Perda de validação
                    val_data_loss = (
                        model.data_loss_function(
                            val_y_pred,
                            val_y_true
                        )
                    )

                    val_loss += val_data_loss.item()

            val_loss /= len(val_loader)

            val_loss_values.append(val_loss)

            # ==========================================
            # LOG DO TREINAMENTO
            # ==========================================

            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {mean_train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Phys Loss: {mean_phys_loss:.6f}, "
                f"Data Loss: {mean_data_loss:.6f}, "
                f"Learning Rate: "
                f"{optimizer.param_groups[0]['lr']:.3e}"
            )

        return (
            train_loss_values,
            val_loss_values,
            phys_loss_values,
            data_loss_values
        )

    # ==================================================
    # TESTE AUTORREGRESSIVO
    # ==================================================

    def test_model(self, x_test, interval, model):

        model.eval()

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Condições iniciais da vazão mássica
        massFlowrate100 = [
            x_test[0, 0, 0].item(),
            x_test[0, 1, 0].item(),
            x_test[0, 2, 0].item()
        ]

        # Condições iniciais da pressão
        PlenumPressure100 = [
            x_test[0, 0, 1].item(),
            x_test[0, 1, 1].item(),
            x_test[0, 2, 1].item()
        ]

        # Tensor de entrada
        input_tensor = torch.zeros(
            (1, 3, 4),
            dtype=dtype,
            device=device
        )

        x_test = torch.as_tensor(
            x_test,
            dtype=dtype,
            device=device
        )

        # Tempo inicial
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        tm1 = time.perf_counter()

        with torch.no_grad():

            for i in range(len(interval)):

                # Histórico da vazão mássica
                input_tensor[0, :, 0] = torch.tensor(
                    massFlowrate100[-3:],
                    dtype=dtype,
                    device=device
                )

                # Histórico da pressão
                input_tensor[0, :, 1] = torch.tensor(
                    PlenumPressure100[-3:],
                    dtype=dtype,
                    device=device
                )

                # Abertura da válvula
                input_tensor[0, :, 2] = (
                    x_test[i, :, 2]
                )

                # Rotação do compressor
                input_tensor[0, :, 3] = (
                    x_test[i, :, 3]
                )

                # Previsão da rede
                prediction100 = model(input_tensor)

                # Armazenamento da vazão prevista
                massFlowrate100.append(
                    prediction100[0, 0, 0].item()
                )

                # Armazenamento da pressão prevista
                PlenumPressure100.append(
                    prediction100[0, 0, 1].item()
                )

        # Tempo final
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        tm2 = time.perf_counter()

        timeteste = tm2 - tm1

        model.train()

        return (
            massFlowrate100,
            PlenumPressure100,
            timeteste
        )