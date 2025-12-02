import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
        self.rnn = nn.LSTM(
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
        # x: [batch, seq_len, n_features]
        return 2 * (x - self.x_min[None, None, :]) / (self.x_max[None, None, :] - self.x_min[None, None, :]) - 1

    def normalize_y(self, y):
        # y: [batch, n_vars]
        return 2 * (y - self.y_min[None, :]) / (self.y_max[None, :] - self.y_min[None, :]) - 1

    def denormalize_y(self, y_norm):
        return (y_norm + 1) * 0.5 * (self.y_max[None, :] - self.y_min[None, :]) + self.y_min[None, :]


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
    def train_model(self, train_loader, epochs=50):
        self.train()

        # Pesos das 3 variáveis de saída
        weights = torch.tensor([15, 12.0, 5.0], dtype=torch.float32).to(self.device)

        for ep in range(epochs):
            total_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()

                pred = self.forward(xb)

                errors = (pred - yb) ** 2
                weighted_errors = errors * weights
                loss = weighted_errors.mean()

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (ep + 1) % 10 == 0:
                print(f"Epoch {ep+1}/{epochs} | Loss = {total_loss/len(train_loader):.6f}")




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
