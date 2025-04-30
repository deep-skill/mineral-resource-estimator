# Implementación simplificada usando scikit-learn para la estimación de recursos minerales

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os
import joblib
import path_utils
np.random.seed(0)

class MineralResourceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, data_path=None):
        """Cargar datos desde un archivo CSV o usar un DataFrame proporcionado"""
        if data_path is None:
            data_path = path_utils.get_build_output_file_path('desurveyed_assay.csv')
            
        print(f"Cargando datos desde: {data_path}")
        self.centauro_df = pd.read_csv(data_path)
        
        # Variables predictoras
        self.predictors = ['xm', 'ym', 'zm', 'azmm', 'dipm']
        # Variable objetivo
        self.target_variable = 'Au_ppm'
        
        # Verificar que las columnas existan
        for col in self.predictors + [self.target_variable]:
            if col not in self.centauro_df.columns:
                print(f"Advertencia: Columna {col} no encontrada en los datos")
                # Intentar con nombres alternativos
                if col == 'xm' and 'X' in self.centauro_df.columns:
                    self.predictors[self.predictors.index('xm')] = 'X'
                    print("Usando 'X' en lugar de 'xm'")
                if col == 'ym' and 'Y' in self.centauro_df.columns:
                    self.predictors[self.predictors.index('ym')] = 'Y'
                    print("Usando 'Y' en lugar de 'ym'")
                if col == 'zm' and 'Z' in self.centauro_df.columns:
                    self.predictors[self.predictors.index('zm')] = 'Z'
                    print("Usando 'Z' en lugar de 'zm'")
                if col == 'azmm' and 'AZM' in self.centauro_df.columns:
                    self.predictors[self.predictors.index('azmm')] = 'AZM'
                    print("Usando 'AZM' en lugar de 'azmm'")
                if col == 'dipm' and 'DIP' in self.centauro_df.columns:
                    self.predictors[self.predictors.index('dipm')] = 'DIP'
                    print("Usando 'DIP' en lugar de 'dipm'")
        
        print(f"Predictores utilizados: {self.predictors}")
        print(f"Variable objetivo: {self.target_variable}")
        
        return self.centauro_df
        
    def prepare_data(self, test_size=0.2, random_state=42, limit_index=None):
        """Preparar los datos para entrenamiento y prueba"""
        X = self.centauro_df[self.predictors]
        y = self.centauro_df[self.target_variable]
        
        # Verificar que hay suficientes datos
        if len(X) < 2:
            print(f"Advertencia: Solo hay {len(X)} muestras disponibles, lo cual es insuficiente para entrenar/probar.")
            print("Usando todas las muestras para entrenamiento.")
            self.X_train = X
            self.y_train = y
            # Crear conjuntos de prueba vacíos pero con las mismas columnas
            self.X_test = X.iloc[0:0].copy()
            self.y_test = y.iloc[0:0].copy()
        elif limit_index is not None:
            # Verificar que el índice límite es válido
            if limit_index >= len(X):
                print(f"Advertencia: Índice límite {limit_index} excede el tamaño del dataset ({len(X)})")
                limit_index = int(len(X) * 0.8)  # Usar 80% para entrenamiento por defecto
                print(f"Usando índice límite ajustado: {limit_index}")
            
            # División específica por índice
            self.X_train = X.iloc[:limit_index]
            self.y_train = y.iloc[:limit_index]
            
            # Verificar si hay datos de prueba
            if limit_index < len(X):
                self.X_test = X.iloc[limit_index:]
                self.y_test = y.iloc[limit_index:]
            else:
                # Crear conjuntos de prueba vacíos pero con las mismas columnas
                self.X_test = X.iloc[0:0].copy()
                self.y_test = y.iloc[0:0].copy()
                print("Advertencia: No hay datos para prueba, usando solo datos de entrenamiento.")
        else:
            # División aleatoria
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Normalización
        self.X_train_normalized = self.scaler.fit_transform(self.X_train)
        
        # Verificar si hay datos de prueba antes de transformar
        if len(self.X_test) > 0:
            self.X_test_normalized = self.scaler.transform(self.X_test)
        else:
            # Crear un array vacío con la forma correcta
            self.X_test_normalized = np.empty((0, len(self.predictors)))
        
        print(f"Datos de entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"Datos de prueba: {self.X_test.shape[0]} muestras")
        
        return self.X_train_normalized, self.y_train, self.X_test_normalized, self.y_test
    
    def build_model(self, n_estimators=100, max_depth=None):
        """Construir el modelo RandomForest"""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        return self.model
    
    def train(self, epochs=None, batch_size=None, verbose=1):
        """Entrenar el modelo"""
        if self.model is None:
            self.build_model()
            
        if verbose:
            print("Entrenando modelo RandomForest...")
            
        self.model.fit(self.X_train_normalized, self.y_train)
        
        # Simulamos un historial para mantener compatibilidad con la interfaz original
        history = {
            'loss': [0],
            'val_loss': [0]
        }
        
        return history
    
    def evaluate(self):
        """Evaluar el modelo en datos de entrenamiento y prueba"""
        if self.model is None:
            print("Error: El modelo no ha sido entrenado todavía.")
            return None
        
        # Predicciones en conjunto de entrenamiento
        y_train_pred = self.model.predict(self.X_train_normalized)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        
        print('\nResultados entrenamiento:')
        print(f'  - MSE: {train_mse:.6f}')
        print(f'  - MAE: {train_mae:.6f}')
        
        # Verificar si hay datos de prueba antes de evaluar
        if len(self.X_test) > 0:
            # Predicciones en conjunto de prueba
            y_test_pred = self.model.predict(self.X_test_normalized)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            print('\nResultados prueba:')
            print(f'  - MSE: {test_mse:.6f}')
            print(f'  - MAE: {test_mae:.6f}')
        else:
            print('\nNo hay datos de prueba para evaluar.')
            test_mse = test_mae = float('nan')
        
        # Crear resultados similares a TensorFlow para compatibilidad
        train_result = [0, train_mse, train_mae]  # [loss, mse, mae]
        test_result = [0, test_mse, test_mae]
        
        return train_result, test_result
    
    def predict(self, X_new):
        """Realizar predicciones con el modelo"""
        if self.model is None:
            print("Error: El modelo no ha sido entrenado todavía.")
            return None
            
        # Si X_new es un DataFrame, extraer solo las columnas predictoras
        if isinstance(X_new, pd.DataFrame):
            # Asegurarse de que estén presentes todas las columnas predictoras
            for col in self.predictors:
                if col not in X_new.columns:
                    print(f"Error: Columna predictora '{col}' no está en los datos.")
                    return None
            X_new = X_new[self.predictors]
            
        # Verificar si hay datos para predecir
        if len(X_new) == 0:
            print("Advertencia: No hay datos para predecir.")
            return np.array([]).reshape(-1, 1)
            
        # Normalizar los datos de entrada
        X_new_normalized = self.scaler.transform(X_new)
        
        # Realizar predicción
        predictions = self.model.predict(X_new_normalized)
        
        # Formatear como array 2D para mantener compatibilidad con la interfaz original
        return predictions.reshape(-1, 1)
    
    def save_model(self, model_path='output/model'):
        """Guardar el modelo entrenado"""
        if self.model is None:
            print("Error: No hay modelo que guardar.")
            return False
            
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Guardar modelo
        joblib.dump(self.model, f"{model_path}.joblib")
        print(f"Modelo guardado en {model_path}.joblib")
        
        # Guardar scaler
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        print(f"Scaler guardado en {model_path}_scaler.pkl")
        
        # Guardar un archivo h5 vacío para mantener compatibilidad
        with open(f"{model_path}.h5", 'w') as f:
            f.write("Placeholder for compatibility")
        
        return True
    
    def load_model(self, model_path='output/model'):
        """Cargar un modelo previamente entrenado"""
        joblib_path = f"{model_path}.joblib"
        h5_path = f"{model_path}.h5"
        
        if os.path.exists(joblib_path):
            # Cargar modelo
            self.model = joblib.load(joblib_path)
            print(f"Modelo cargado desde {joblib_path}")
        elif os.path.exists(h5_path):
            # Informar al usuario y entrenar un nuevo modelo
            print(f"Aviso: Se encontró un archivo {h5_path} pero no es compatible con el nuevo modelo.")
            print(f"Se entrenará un nuevo modelo RandomForest.")
            return False
        else:
            print(f"Error: No se encontró el modelo en {model_path}.joblib o {model_path}.h5")
            return False
            
        # Cargar scaler
        if os.path.exists(f"{model_path}_scaler.pkl"):
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            print(f"Scaler cargado desde {model_path}_scaler.pkl")
        
        return True

# Si se ejecuta como script independiente
if __name__ == "__main__":
    # Ejemplo de uso
    model = MineralResourceModel()
    model.load_data()
    model.prepare_data(limit_index=19452)  # Usar la misma división que tenías originalmente
    model.build_model()
    model.train()
    model.evaluate()
    model.save_model()
