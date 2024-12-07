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
        
        
        # Camadas RNN
        self.rnn_layer = nn.LSTM(
                input_size = 4,
                hidden_size = units,
                batch_first= True,
                bidirectional= True,
                bias = True,
            )
        
        # Camada densa
        self.dense_layers = nn.Sequential(
            nn.Linear(units*2, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
                )


    def forward(self, inputs):
        # Passagem pelas camadas RNN
        rnn_input = 2 * (inputs - self.x_min) / (self.x_max - self.x_min) - 1
        rnn_output, _ = self.rnn_layer(rnn_input)

        dense_output = self.dense_layers(rnn_output[:, -1, :])  # Dimensão [batch_size, hidden_size * num_directions]
        desnormalizado = ((dense_output + 1) / 2) * (self.x_max[:, :, :2] - self.x_min[:, :, :2]) + self.x_min[:, :, :2]
          #Pegando apenas a última saída da sequência
        return desnormalizado
    
    def loss_custom(self, y_true, y_pred, inputs):
        # Implementação da função de perda
        interp = []
        for i in range (0, inputs.shape[0]):
            aux = self.interpolation([(inputs[i, -1, -1]).item(), (y_pred[0, i, 0]).item()])
            interp.append(aux)
        interp = np.array(interp)
        interp = torch.tensor(interp, dtype=torch.float32)

        m_t = (11* y_pred[:, :, 0] - 18 *inputs[:, -2, 0] + 9 * inputs[:, -3, 0] - 2 * inputs[:, 0, 0])/(6*self.dt)
        p_t = (11* y_pred[:, :, 1] - 18 *inputs[:, -2, 1] + 9 * inputs[:, -2, 1] - 2 * inputs[:, -2, 1])/(6*self.dt)
        
        fLoss_mass = torch.mean(torch.square(m_t - (self.A1/self.Lc)*(( interp * self.P1) - y_pred[:, :, 1]) * 1e3))
        fLoss_pres = torch.mean(torch.square(p_t - (self.C**2)/2 * (y_pred[:, :, 0] - inputs[:, -1, -2]* self.kv * torch.sqrt((torch.abs(y_pred[:, :, 1] * 1000 - self.P_out * 1000))))))
        
        phys_loss = fLoss_mass + fLoss_pres
        data_loss =  1e2* torch.mean((y_true[:, 0, 0] - y_pred[:, :, 0]) ** 2) + 1e1*torch.mean((y_true[:, 0, 1] - y_pred[:, :, 1]) ** 2)
        
        return data_loss +  3e-9 *phys_loss

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