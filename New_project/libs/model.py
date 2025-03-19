import time
import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self, units, x_max, x_min):
        
        self.x_max = x_max
        self.x_min = x_min
        
        super(MyModel, self).__init__()
        
        self.rnn_layer = nn.LSTM(
            input_size=7,
            hidden_size=units,
            batch_first=True,
            bidirectional=False,
            bias=True,
            num_layers= 2
        )
        
        self.dense_layers = nn.Sequential(
            nn.Linear(units, 64),
            nn.Tanh(),
            nn.Linear(64, 14),
        )

    def forward(self, inputs):
        rnn_output, _ = self.rnn_layer(inputs)
        dense_output = self.dense_layers(rnn_output[:, -1, :])
        return dense_output
    
    def train_model(self, model, train_loader, val_loader, lr, epochs, optimizers, patience, factor):
        optimizer = optimizers(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = factor, patience = patience, verbose=True)

        model.train()

        # Listas para armazenar as perdas
        train_loss_values = []
        val_loss_values = []

        for epoch in range(epochs):
            # Treinamento
            model.train()
            total_loss = 0

            for inputs, y_true in train_loader:
                optimizer.zero_grad()
                y_pred = self(inputs)
                loss = nn.functional.mse_loss(y_pred, y_true)  # MSE único
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping
                optimizer.step()

                total_loss += loss.item()
            
            # Atualizar o scheduler
            scheduler.step(total_loss / len(train_loader))

            train_loss_values.append(total_loss / len(train_loader))
            
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
                f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")


        return train_loss_values, val_loss_values