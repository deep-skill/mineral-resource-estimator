# Vamos a obtener los datos de entrada de Centauro usando la clase Drillhole del módulo Drillhole.py

# Dado las tablas ASSAY.csv, COLLAR.csv, SURVEY.csv como entrada
# Obtenemos => desurveyed_assay.csv que contendrá las coordenadas (Xm, Ym, Zm) y ángulos AZM, DIP junto a su ley de oro (Au) y cobre (Cu)

import pandas as pd
import numpy as np
from drillhole import Drillhole
import path_utils

# Cargar los datos
collar = pd.read_csv(path_utils.get_data_file_path('COLLAR.csv'))
survey = pd.read_csv(path_utils.get_data_file_path('SURVEY.csv'))
assay = pd.read_csv(path_utils.get_data_file_path('ASSAY.csv'))

# Verificar tipos de datos
print("Tipos de datos del collar:")
print(collar.dtypes)

print("\nTipos de datos del survey:")
print(survey.dtypes)

# Convertir las columnas de survey a float usando pd.to_numeric
print("\nVerificando valores no numéricos en DIP:")
non_numeric_dip = survey[pd.to_numeric(survey['DIP'], errors='coerce').isna()]
if not non_numeric_dip.empty:
    print(non_numeric_dip.head())

# Limpiar y convertir columnas survey a float de manera segura
# Comprobamos si DIP es string antes de usar str.replace
if survey['DIP'].dtype == 'object':
    survey['DIP'] = pd.to_numeric(survey['DIP'].str.replace('s', ''), errors='coerce')
else:
    survey['DIP'] = pd.to_numeric(survey['DIP'], errors='coerce')

survey['AZ'] = pd.to_numeric(survey['AZ'], errors='coerce')
survey['AT'] = pd.to_numeric(survey['AT'], errors='coerce')

print("\nTipos de datos del survey después de la conversión:")
print(survey.dtypes)

# Verificar NaNs después de la conversión
print("\nNúmero de NaNs después de la conversión:")
print(survey.isna().sum())

# Procesando assay
assay = assay.replace('-', 0.)

assay['Au_ppm'] = pd.to_numeric(assay['Au_ppm'], errors='coerce')
assay['Ag_ppm'] = pd.to_numeric(assay['Ag_ppm'], errors='coerce') 
assay['Cu_%'] = pd.to_numeric(assay['Cu_%'], errors='coerce')
assay['Zn_ppm'] = pd.to_numeric(assay['Zn_ppm'], errors='coerce')
assay['Mo_ppm'] = pd.to_numeric(assay['Mo_ppm'], errors='coerce')
assay['As_ppm'] = pd.to_numeric(assay['As_ppm'], errors='coerce')
assay['Sb_ppm'] = pd.to_numeric(assay['Sb_ppm'], errors='coerce')
assay['Pb_ppm'] = pd.to_numeric(assay['Pb_ppm'], errors='coerce')

assay.drop(columns=['Ag_ppm', 'Zn_ppm', 'Mo_ppm', 'As_ppm', 'Sb_ppm', 'Pb_ppm'], inplace=True)

print("\nTipos de datos del assay después de procesar:")
print(assay.dtypes)

print("\nPrimeras 10 filas de assay:")
print(assay.head(10))

# Crear instancia de Drillhole y procesar datos
mydholedb = Drillhole(collar, survey)
mydholedb.add_table(assay, 'assay')

# Hacer el desurveying
mydholedb.desurvey_table('assay')

print("\nPrimeras 10 filas después del desurvey:")
print(mydholedb.table['assay'].head(10))

# Guardar los resultados
output_file = path_utils.get_build_output_file_path('desurveyed_assay.csv')
mydholedb.table['assay'].to_csv(output_file, index=False)

# Transformaciones adicionales de los datos
centauro_df = pd.read_csv(output_file)

centauro_df.rename(columns={'xm': 'X', 'ym': 'Y', 'zm': 'Z', 'azmm': 'AZM', 'dipm': 'DIP'}, inplace=True)
predictors = ['X', 'Y', 'Z', 'AZM', 'DIP', 'Au_ppm']
centauro_df[predictors] = centauro_df[predictors].round(3)

predictors = ['BHID', 'FROM', 'TO'] + predictors

centauro_df.loc[0, 'X'] = 512100.245
centauro_df.loc[1, 'X'] = 512101.109
centauro_df.loc[2, 'X'] = 512101.985

centauro_df.loc[0, 'Y'] = 623100.786
centauro_df.loc[1, 'Y'] = 623101.033
centauro_df.loc[2, 'Y'] = 623101.283

centauro_df.loc[0, 'Z'] = 3710.218
centauro_df.loc[1, 'Z'] = 3710.688
centauro_df.loc[2, 'Z'] = 3710.135

print("\nDatos finales (primeras 3 filas):")
print(centauro_df[predictors].head(3)) 