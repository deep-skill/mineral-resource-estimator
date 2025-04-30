#!/usr/bin/env python3
"""
Mineral Resource Estimator - Main Orchestration Script
Este script orquesta el proceso completo de estimación de recursos minerales:
1. Preprocesamiento de datos de perforación
2. Entrenamiento/carga del modelo
3. Predicción y visualización de resultados
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import path_utils

def preprocess_data():
    """Ejecutar el preprocesamiento de datos"""
    print("\n=== Iniciando preprocesamiento de datos ===")
    
    try:
        # Importamos el módulo de preprocesamiento
        import preprocessing
        
        # Verificar que el archivo de salida existe
        output_file = path_utils.get_build_output_file_path('desurveyed_assay.csv')
        if os.path.exists(output_file):
            print(f"Archivo de datos procesados generado: {output_file}")
            print(f"  - Tamaño: {os.path.getsize(output_file) / 1024:.2f} KB")
            print(f"  - Fecha modificación: {datetime.fromtimestamp(os.path.getmtime(output_file)).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"Advertencia: No se encontró el archivo de salida esperado: {output_file}")
        
        print("Preprocesamiento completado exitosamente.")
        return True
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        return False

def train_model(force_retrain=False):
    """Entrenar o cargar el modelo predictivo"""
    print("\n=== Iniciando entrenamiento/carga del modelo ===")
    
    try:
        # Importamos la clase del modelo
        from model import MineralResourceModel
        
        # Creamos instancia del modelo
        model = MineralResourceModel()
        
        # Comprobar si existe un modelo entrenado
        model_path = path_utils.get_output_file_path('model')
        if os.path.exists(f"{model_path}.h5") and not force_retrain:
            print(f"Se encontró un modelo entrenado previamente.")
            print(f"Cargando modelo desde: {model_path}.h5")
            model.load_model(model_path)
        else:
            print("Iniciando entrenamiento de nuevo modelo:")
            
            # Cargar los datos
            data = model.load_data()
            
            # Preparar los datos
            model.prepare_data(limit_index=19452)  # Usar la misma división que el script original
            
            # Construir el modelo
            model.build_model()
            
            # Entrenar el modelo (reducido a 10 épocas para pruebas rápidas, ajustar según sea necesario)
            print("Iniciando entrenamiento (esto puede tomar tiempo)...")
            model.train(epochs=400, verbose=1)
            
            # Evaluar el modelo
            model.evaluate()
            
            # Guardar el modelo
            model.save_model()
        
        # Guardar referencia global al modelo para usar en otras funciones
        global trained_model
        trained_model = model
        
        print("Modelo cargado exitosamente.")
        return True
    except Exception as e:
        print(f"Error durante la carga/entrenamiento del modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def make_predictions():
    """Realizar predicciones con el modelo entrenado"""
    print("\n=== Realizando predicciones ===")
    
    try:
        # Verificar que el modelo está cargado
        if 'trained_model' not in globals():
            print("Error: El modelo no ha sido entrenado o cargado.")
            return False
        
        # Comprobar si existen datos de prueba
        if hasattr(trained_model, 'X_test') and hasattr(trained_model, 'y_test') and len(trained_model.X_test) > 0:
            # Usar los datos de prueba existentes
            test_data = trained_model.X_test.copy()
            
            # Realizar predicciones
            predictions = trained_model.predict(test_data)
            
            # Crear DataFrame de resultados
            results_df = pd.DataFrame({
                'Real': trained_model.y_test.reset_index(drop=True),
                'Predicción': predictions.flatten()
            })
            
            # Guardar resultados
            results_path = path_utils.get_output_file_path('predictions.csv')
            results_df.to_csv(results_path, index=False)
            print(f"Predicciones guardadas en: {results_path}")
            
            # Generar visualización básica
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.scatter(results_df['Real'], results_df['Predicción'], alpha=0.5)
                plt.plot([0, results_df['Real'].max()], [0, results_df['Real'].max()], 'r--')
                plt.xlabel('Valores Reales')
                plt.ylabel('Predicciones')
                plt.title('Comparación entre Valores Reales y Predicciones')
                plt.grid(True)
                plot_path = path_utils.get_plot_file_path('predictions_scatter.png')
                plt.savefig(plot_path)
                plt.close()
                print(f"Gráfico de dispersión guardado en: {plot_path}")
            except Exception as e:
                print(f"Advertencia: No se pudo generar el gráfico de dispersión: {e}")
        else:
            print("No hay datos de prueba disponibles en el modelo cargado.")
            print("Para realizar predicciones, necesita cargar un conjunto de datos de prueba o entrenar un nuevo modelo.")
            
            # Opcionalmente, se podría cargar un conjunto de datos aquí para hacer predicciones
            # Por ejemplo:
            # data_path = path_utils.get_data_file_path('new_samples.csv')
            # if os.path.exists(data_path):
            #     test_data = pd.read_csv(data_path)
            #     predictions = trained_model.predict(test_data)
            #     ...
            
            return False
        
        print("Predicciones realizadas exitosamente.")
        return True
    except Exception as e:
        print(f"Error durante las predicciones: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_report():
    """Generar reporte de resultados"""
    print("\n=== Generando reporte de resultados ===")
    
    try:
        # Verificar que el modelo está cargado
        model_loaded = 'trained_model' in globals()
        
        # Verificar que existen archivos de predicciones
        predictions_path = path_utils.get_output_file_path('predictions.csv')
        predictions_exist = os.path.exists(predictions_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = path_utils.get_output_file_path(f"resource_estimation_report_{timestamp}.txt")
        
        with open(report_path, "w") as f:
            f.write(f"REPORTE DE ESTIMACIÓN DE RECURSOS MINERALES\n")
            f.write(f"===========================================\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"RESUMEN DEL PROCESO\n")
            f.write(f"------------------\n")
            f.write(f"1. Preprocesamiento: Completado\n")
            
            if model_loaded:
                f.write(f"2. Modelo: Cargado exitosamente\n")
                
                # Agregar métricas del modelo si están disponibles
                try:
                    train_results, test_results = trained_model.evaluate()
                    f.write(f"   - MSE entrenamiento: {train_results[1]:.6f}\n")
                    f.write(f"   - MAE entrenamiento: {train_results[2]:.6f}\n")
                    f.write(f"   - MSE prueba: {test_results[1]:.6f}\n")
                    f.write(f"   - MAE prueba: {test_results[2]:.6f}\n")
                except:
                    f.write(f"   - No se pudieron obtener métricas detalladas\n")
            else:
                f.write(f"2. Modelo: No disponible\n")
            
            if predictions_exist:
                f.write(f"3. Predicciones: Completadas\n")
                f.write(f"   - Resultados disponibles en: output/predictions.csv\n")
                
                # Agregar estadísticas de las predicciones
                try:
                    predictions_df = pd.read_csv(predictions_path)
                    error = predictions_df['Real'] - predictions_df['Predicción']
                    mae = abs(error).mean()
                    mse = (error ** 2).mean()
                    rmse = np.sqrt(mse)
                    
                    f.write(f"   - Error absoluto medio (MAE): {mae:.6f}\n")
                    f.write(f"   - Error cuadrático medio (MSE): {mse:.6f}\n")
                    f.write(f"   - Raíz del error cuadrático medio (RMSE): {rmse:.6f}\n")
                except:
                    f.write(f"   - No se pudieron calcular estadísticas detalladas\n")
            else:
                f.write(f"3. Predicciones: No disponibles\n")
            
            f.write(f"\nCONCLUSIÓN\n")
            f.write(f"----------\n")
            if model_loaded and predictions_exist:
                f.write(f"El proceso se ha completado exitosamente. El modelo ha sido entrenado\n")
                f.write(f"y se han realizado predicciones sobre los datos de prueba.\n")
            else:
                f.write(f"El proceso ha finalizado con ciertas limitaciones. Revisar los detalles anteriores.\n")
        
        print(f"Reporte generado en: {report_path}")
        return True
    except Exception as e:
        print(f"Error generando el reporte: {e}")
        return False

def main():
    """Función principal que orquesta todo el proceso"""
    parser = argparse.ArgumentParser(description='Mineral Resource Estimator')
    parser.add_argument('--skip-preprocess', action='store_true', 
                        help='Omitir el paso de preprocesamiento')
    parser.add_argument('--skip-model', action='store_true',
                        help='Omitir el paso de entrenamiento/carga del modelo')
    parser.add_argument('--skip-predictions', action='store_true',
                        help='Omitir el paso de predicciones')
    parser.add_argument('--skip-report', action='store_true',
                        help='Omitir la generación del reporte')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Forzar reentrenamiento del modelo aunque exista uno guardado')
    
    args = parser.parse_args()
    
    print("=== Iniciando Estimador de Recursos Minerales ===")
    print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configurar directorios
    path_utils.ensure_directories_exist()
    
    # Ejecutar los pasos del proceso
    if not args.skip_preprocess:
        if not preprocess_data():
            print("Error en preprocesamiento. Abortando.")
            return 1
    else:
        print("Paso de preprocesamiento omitido.")
    
    if not args.skip_model:
        if not train_model(force_retrain=args.force_retrain):
            print("Error en entrenamiento/carga del modelo. Abortando.")
            return 2
    else:
        print("Paso de entrenamiento/carga del modelo omitido.")
    
    if not args.skip_predictions:
        if not make_predictions():
            print("Error en predicciones. Abortando.")
            return 3
    else:
        print("Paso de predicciones omitido.")
    
    if not args.skip_report:
        if not generate_report():
            print("Error generando el reporte. Abortando.")
            return 4
    else:
        print("Generación de reporte omitida.")
    
    print("\n=== Proceso completado exitosamente ===")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 