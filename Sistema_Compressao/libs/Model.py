import time
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, units, A1, Lc, kv, P1, P_out, C, dt, x_min, x_max):
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.dt = dt
        self.x_min = x_min
        self.x_max = x_max
        super(MyModel, self).__init__()
        
        # Camada de entrada
        self.input_layer = nn.Linear(3, 3)
        
        # Camadas RNN
        self.rnn_layers = nn.ModuleList([
            nn.LSTM(input_size=3, hidden_size=units, batch_first=True, bidirectional=True, bias= True)
        ])
        
        # Camada densa
        self.dense = nn.Linear(units*2, 2)

        # Inicialização dos pesos com Xavier (Glorot Uniform)
        self._initialize_weights()

    def _initialize_weights(self):
        # Inicializador Xavier (Glorot) nos pesos das camadas RNN
        for rnn_layer in self.rnn_layers:
            for name, param in rnn_layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, inputs):
        # Passagem pelas camadas RNN
        rnn_output = 2 * (inputs - self.x_min) / (self.x_max - self.x_min) - 1
        for rnn_layer in self.rnn_layers:
            rnn_output, _ = rnn_layer(rnn_output)
            rnn_output = torch.tanh(rnn_output)  # Aplicando tanh explicitamente após cada camada RNN
         # Pegando apenas a última saída da sequência e desnormalizando
        dense_output = self.dense(rnn_output[:, -1, :])  # Dimensão [batch_size, hidden_size * num_directions]
        
        desnormalizado = ((dense_output + 1) / 2) * (self.x_max[:, :, :2] - self.x_min[:, :, :2]) + self.x_min[:, :, :2]
          # Pegando apenas a última saída da sequência
        return desnormalizado
    
    def loss_custom(self, y_true, y_pred, inputs):
        # Implementação da função de perda
        m_t = (11* y_pred[:, :, 0] - 18 *inputs[:, -2, 0] + 9 * inputs[:, -3, 0] - 2 * inputs[:, 0, 0])/6*self.dt
        p_t = (11* y_pred[:, :, 1] - 18 *inputs[:, -2, 1] + 9 * inputs[:, -2, 1] - 2 * inputs[:, -2, 1])/6*self.dt
        fLoss_mass = torch.mean(torch.square(m_t - (self.A1/self.Lc)*((1.5 * self.P1) - y_pred[:, :, 1]) * 1e3))
        fLoss_pres = torch.mean(torch.square(p_t - (self.C**2)/2 * (y_pred[:, :, 0] - inputs[:, -1, -1]* self.kv * torch.sqrt((torch.abs(y_pred[:, :, 1] * 1000 - self.P_out * 1000))))))
        phys_loss = fLoss_mass + fLoss_pres
        data_loss =  torch.mean((y_true[:, 0, 0] - y_pred[:, :, 0]) ** 2) + torch.mean((y_true[:, 0, 1] - y_pred[:, :, 1]) ** 2)
        return data_loss +  5e-11*phys_loss

    def train_model(self, model, train_loader, lr, epochs, optimizers):
        optimizer = optimizers(model.parameters(), lr=lr)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for inputs, y_true in train_loader:
                optimizer.zero_grad()
                
                y_pred = model(inputs)
                loss = self.loss_custom(y_true, y_pred, inputs)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader)}")
        
    def test_model(self, x_test, interval, model):
        massFlowrate100 = [x_test[0, 0, 0].item(), x_test[0, 1, 0].item(), x_test[0, 2, 0].item()]
        PlenumPressure100 = [x_test[0, 0, 1].item(), x_test[0, 1, 1].item(), x_test[0, 2, 1].item()]

        # Preparar o tensor inicial fora do loop
        input_tensor = torch.zeros((1, 3, 3), dtype=torch.float32)

        # Loop de previsões
        tm1 = time.time()
        for i in range(len(interval)):
            # Atualizar os valores do tensor diretamente
            input_tensor[0, :, 0] = torch.tensor(massFlowrate100[-3:])
            input_tensor[0, :, 1] = torch.tensor(PlenumPressure100[-3:])
            input_tensor[0, :, 2] = x_test[i, :, 2]

            # Previsão com desativação do gradiente
            with torch.no_grad():
                prediction100 = model(input_tensor)

            # Adicionar previsões diretamente
            massFlowrate100.append(prediction100[0, 0, 0].item())
            PlenumPressure100.append(prediction100[0, 0, 1].item())
        tm2 = time.time()
        timeteste = tm2 - tm1
        return massFlowrate100,PlenumPressure100, timeteste