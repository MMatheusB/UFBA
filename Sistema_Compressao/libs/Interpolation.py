import numpy as np
import casadi as ca
import pandas as pd

class Interpolation:
    def __init__(self, file_path, decimal = ','):
        self.file_path = file_path
        self.decimal = decimal
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path, decimal=self.decimal)
        self.N_rot = np.arange(2e4,6e4,1e3) # Vai de 20000hz até 60000hz, Shape: (40,)
        self.Mass = np.arange(3,21.1,0.1) # Vai de 3 até 21, Shape: (181,)
        self.Phi = self.data.values # Valores da tabela, Shape: (40,181)
    
    def interpolate(self):
        # Criar uma malha densa para interpolação
        phi_flat = self.Phi.ravel(order='F')
        lut = ca.interpolant('name','bspline',[self.N_rot, self.Mass],phi_flat)

        return lut
    