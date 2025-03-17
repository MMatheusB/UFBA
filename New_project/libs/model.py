import time
import torch
import torch.nn as nn
import numpy as np

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
        
        self.rnn_layer = nn.LSTM(
            input_size=4,
            hidden_size=units,
            batch_first=True,
            bidirectional=False,
            bias=True,
        )
        
        self.dense_layers = nn.Sequential(
            nn.Linear(units, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, inputs):
        rnn_input = 2 * (inputs - self.x_min) / (self.x_max - self.x_min) - 1
        
        rnn_output, _ = self.rnn_layer(rnn_input)
        
        dense_output = self.dense_layers(rnn_output[:, -1, :])
        
        desnormalizado = ((dense_output + 1) / 2) * (self.x_max[:, :, :2] - self.x_min[:, :, :2]) + self.x_min[:, :, :2]
        return desnormalizado