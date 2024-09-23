import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Criar os dados
# Vamos usar a sequência [1, 2, 3, 4] para prever o próximo número
data = np.array([1, 2, 3, 4])
X = []
y = []

# Criar as sequências de entrada e saída
for i in range(len(data) - 1):
    X.append(data[i:i+1])  # Usamos 1 valor para prever o próximo
    y.append(data[i+1])    # O próximo valor

X = np.array(X)
y = np.array(y)

# Redimensionar X para 3D [amostras, passos de tempo, características]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Definir o modelo
model = keras.Sequential()
model.add(layers.SimpleRNN(10, activation='relu', input_shape=(X.shape[1], 1)))
model.add(layers.Dense(1))  # Camada de saída

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X, y, epochs=2000, verbose=0)

# Fazer previsões
test_input = np.array([[4]])  # Último número da sequência
test_input = test_input.reshape((1, 1, 1))  # Redimensionar para 3D

predicted = model.predict(test_input)
print(f"Previsão para o próximo número após 4: {predicted[0][0]}")