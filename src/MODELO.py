# Implementación de red neuronal para la estimación de recursos minerales

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
import pandas as pd
import numpy as np

np.random.seed(0)
tf.random.set_seed(12)

# Datos de entrada del modelo

centauro_df = pd.read_csv('../DEEPSKILL_DATA/desurveyed_assay.csv')

# Variables predictoras
predictors = ['xm', 'ym', 'zm', 'azmm', 'dipm']
# Variable objetivo
target_variable = 'Au_ppm'

X = centauro_df[predictors]
y = centauro_df[target_variable]

lmt_up = 19452

X_train = X.loc[ : lmt_up-1]
y_train = y.loc[ : lmt_up-1]

X_test = X.loc[lmt_up : ]
y_test = y.loc[lmt_up : ]

scaler = StandardScaler()

X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Implementamos una red neuronal con "keras" con capas de intermedias de 28 neuronas

model = keras.Sequential([
    keras.layers.Dense(28, activation='tanh', input_shape = (X_train_normalized.shape[1],)),
    keras.layers.Dense(28, activation='tanh'),
    keras.layers.Dense(1)
])

model.compile(optimizer='RMSprop',
              loss='mean_squared_error',
              metrics=['mean_squared_error', 'mean_absolute_error'])

model.fit(X_train_normalized, y_train, epochs=400)

# Imprimimos el resultado obtenido

result = model.evaluate(X_train_normalized, y_train)
# test_loss, test_mse, test_mae, test_r2 = model.evaluate(X_train_normalized, y_train)
print('\nTraining Result:', result)

result = model.evaluate(X_test_normalized, y_test)
# test_loss, test_mse, test_mae, test_r2 = model.evaluate(X_test_normalized, y_test)
print('\nTest Result:', result)
