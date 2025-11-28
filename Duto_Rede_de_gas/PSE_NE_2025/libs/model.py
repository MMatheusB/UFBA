import torch
import torch.nn as nn
import torch.optim as optim

class RNNModelWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 x_min, x_max, y_min, y_max, lr=1e-3, device="cpu"):

        super().__init__()
        self.device = device

        # Guardar limites para normalização
        self.x_min = torch.tensor(x_min, dtype=torch.float32).to(device)
        self.x_max = torch.tensor(x_max, dtype=torch.float32).to(device)
        self.y_min = torch.tensor(y_min, dtype=torch.float32).to(device)
        self.y_max = torch.tensor(y_max, dtype=torch.float32).to(device)

        # --------------------- MODELO ---------------------
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bias=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

        # Otimizador
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()


    # ====================================================
    # ------------------ NORMALIZAÇÃO --------------------
    # ====================================================
    def normalize_x(self, x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
    
    def normalize_y(self, y):
        # y: [batch, n_vars], self.y_min/self.y_max: [n_vars]
        return 2 * (y - self.y_min.unsqueeze(0)) / (self.y_max.unsqueeze(0) - self.y_min.unsqueeze(0)) - 1
    


    def denormalize_y(self, y_norm):
        return (y_norm + 1) * 0.5 * (self.y_max - self.y_min) + self.y_min


    # ====================================================
    # ----------------------- FORWARD ---------------------
    # ====================================================
    def forward(self, x):
        out, _ = self.rnn(x)
        # pegar apenas o último passo da sequência
        last_out = out[:, -1, :]
        return self.fc(last_out)


    # ====================================================
    # ---------------------- TREINO -----------------------
    # ====================================================
    def train_model(self, x_train, y_train, epochs=50):
        self.train()
        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        # normalizar
        x_train = self.normalize_x(x_train)
        y_train = self.normalize_y(y_train)

        # ----- pesos das variáveis de saída -----
        weights = torch.tensor([1.0, 5, 2.0], dtype=torch.float32).to(self.device)  # exemplo: ajustar w1,w2,w3

        for ep in range(epochs):
            self.optimizer.zero_grad()
            pred = self.forward(x_train)  # (N, output_dim)

            # erro quadrático por variável
            errors = (pred - y_train) ** 2

            # aplica os pesos
            weighted_errors = errors * weights

            # loss total
            loss = weighted_errors.mean()

            # perdas separadas por variável
            loss_per_var = weighted_errors.mean(dim=0)  # (output_dim,)

            loss.backward()
            self.optimizer.step()

            if (ep + 1) % 10 == 0:
                print(f"Epoch {ep+1}/{epochs} | Loss = {loss.item():.5f} | Per-variable = {loss_per_var.detach().cpu().numpy()}")




    # ====================================================
    # ---------------------- PREDIÇÃO ----------------------
    # ====================================================
    def predict(self, x):
        self.eval()
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x_norm = self.normalize_x(x)

        with torch.no_grad():
            y_norm = self.forward(x_norm)

        y = self.denormalize_y(y_norm)
        return y.cpu().numpy()
