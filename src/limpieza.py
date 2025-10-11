import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import os
from typing import Text
import warnings
warnings.filterwarnings('ignore')

# Limpiar y detectar atipicos con IQR ===============================================================================================================
def limpiar_y_detectar_atipicos(df: pd.DataFrame, variables_numericas, variables_categoricas, target_column: str = None, cols_to_drop: list = None) -> pd.DataFrame:
  """
  Realiza un proceso completo de limpieza de datos y detección de atípicos.

  Esta función ejecuta los siguientes pasos en orden:
  1. Crea una copia del DataFrame para evitar modificaciones inesperadas.
  2. Estandariza los nombres de las columnas (minúsculas, sin espacios).
  3. Elimina columnas irrelevantes y filas duplicadas.
  4. Elimina filas donde la variable objetivo es nula.
  5. Estandariza los valores de las columnas categóricas.
  6. Convierte columnas a tipo numérico, forzando errores a NaN.
  7. Estandariza los valores de las columna objetivo (Codificación Numérica Ordinal).
  8. Imputa valores nulos: mediana para numéricos y moda para categóricos.
  9. Detecta y reporta datos atípicos (outliers) usando el método IQR.

  Args:
      df (pd.DataFrame): El DataFrame de entrada a limpiar.
      variables_numericas (list): Una lista con los nombres de las columnas numéricas.
      variables_categoricas (list): Una lista con los nombres de las columnas numéricas.
      target_column (str): El nombre de la columna objetivo.
      cols_to_drop (list, optional): Lista de nombres de columnas a eliminar. Defaults to None.

  Returns:
      pd.DataFrame: Un nuevo DataFrame limpio y con los valores nulos imputados.
  """
  display(Markdown("---"))
  display(Markdown("## **Proceso de Limpieza y Detección de Atípicos**"))
  display(Markdown("---"))

  # 1. Crear una copia para no modificar el original
  df_limpio = df.copy()

  # 2. Estandarizar nombres de columnas
  print("1. Estandarizando nombres de columnas...")
  df_limpio.columns = df_limpio.columns.str.strip().str.replace(' ', '_').str.lower()
  # Asegurarnos que el target_column también esté en minúsculas para consistencia
  target_column = target_column.lower()

  # 3. Eliminar columnas y duplicados
  print("2. Eliminando columnas irrelevantes y duplicados...")
  df_limpio.drop(columns=cols_to_drop, inplace=True, axis=1)
  variables_numericas.remove(cols_to_drop)

  df_limpio.drop_duplicates(inplace=True)
  df_limpio.dropna(subset=[target_column], inplace=True)

  # 4. Estandarizar valores categóricos
  print("3. Estandarizando valores en columnas categóricas...")
  for col in variables_categoricas + [target_column]:
      df_limpio[col] = df_limpio[col].str.strip().str.replace(' ', '_').str.lower()
      df_limpio = df_limpio[df_limpio[col] != 'nan']

  # 5. Forzar tipos de datos numéricos
  print("4. Asegurando tipos de datos numéricos...")
  for col in variables_numericas:
      df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')

  # 6. Asignar un numero a la variable objetivo
  mapeo_obesity = {
    'insufficient_weight': 0,
    'normal_weight': 1,
    'overweight_level_i': 2,
    'overweight_level_ii': 3,
    'obesity_type_i': 4,
    'obesity_type_ii': 5,
    'obesity_type_iii': 6
    }

  # Aplicar el mapeo a la columna objetivo.
  df_limpio[target_column] = df_limpio[target_column].map(mapeo_obesity)

  # 7. Imputación de valores nulos
  print("5. Imputando valores nulos...")
  # Imputar numéricas con la mediana
  for col in variables_numericas:
      if df_limpio[col].isnull().any():
          mediana = df_limpio[col].median()
          df_limpio[col].fillna(mediana, inplace=True)

  # Imputar categóricas con la moda
  for col in variables_categoricas:
      if df_limpio[col].isnull().any():
          moda = df_limpio[col].mode()[0]
          df_limpio[col].fillna(moda, inplace=True)

  print("   - No se encontraron más valores nulos.")

  # 8. Detección de Datos Atípicos (Outliers) con IQR
  display(Markdown("### Detección de Atípicos (Método IQR)"))
  for col in variables_numericas:
      Q1 = df_limpio[col].quantile(0.25)
      Q3 = df_limpio[col].quantile(0.75)
      IQR = Q3 - Q1
      limite_inferior = Q1 - 3 * IQR
      limite_superior = Q3 + 3 * IQR

      # Filtrar outliers
      outliers = df_limpio[(df_limpio[col] <= limite_inferior) | (df_limpio[col] >= limite_superior)]

      if not outliers.empty:
          porcentaje = (len(outliers) / len(df_limpio)) * 100
          print(f"\nColumna '{col}':")
          print(f"  - Límite inferior: {limite_inferior:.2f}")
          print(f"  - Límite superior: {limite_superior:.2f}")
          print(f"  - Número de atípicos encontrados: {len(outliers)}")
          print(f"  - Porcentaje de atípicos: {porcentaje:.2f}%")

  display(Markdown("---"))
  print("\nProceso de limpieza finalizado.")

  return df_limpio

# Eliminar datos atipicos con IQR ===============================================================================================================
def eliminar_atipicos(
    df: pd.DataFrame,
    age_range: tuple = (1, 50),
    height_max: float = 2.5,
    ncp_max: float = 10.0,
    iqr_factor: float = 1.5
) -> pd.DataFrame:
  """
  Elimina datos atípicos de un DataFrame utilizando una estrategia mixta.

  Aplica reglas de negocio específicas para columnas conocidas y el método IQR
  para el resto de las variables numéricas.

  Args:
      df (pd.DataFrame): DataFrame limpio (idealmente el resultado de la función de limpieza).
      age_range (tuple, optional): Rango (mín, máx) de edad a conservar. Defaults to (1, 50).
      height_max (float, optional): Altura máxima en metros a conservar. Defaults to 2.5.
      ncp_max (float, optional): Número máximo de comidas principales a conservar. Defaults to 10.0.
      iqr_factor (float, optional): Factor para multiplicar el IQR y definir los límites. Defaults to 1.5.

  Returns:
      pd.DataFrame: Un nuevo DataFrame sin los datos atípicos.
  """
  display(Markdown("---"))
  display(Markdown("## **Proceso de Eliminación de Atípicos**"))
  display(Markdown("---"))

  df_tratado = df.copy()
  filas_iniciales = len(df_tratado)

  # 1. Aplicar reglas de negocio específicas
  print("1. Aplicando reglas de negocio para 'age', 'height' y 'ncp'...")

  # Columnas con reglas específicas
  columnas_con_reglas = ['age', 'height', 'ncp']

  if 'age' in df_tratado.columns:
      df_tratado = df_tratado[(df_tratado['age'] >= age_range[0]) & (df_tratado['age'] <= age_range[1])]

  if 'height' in df_tratado.columns:
      df_tratado = df_tratado[df_tratado['height'] < height_max]

  if 'ncp' in df_tratado.columns:
      df_tratado = df_tratado[df_tratado['ncp'] < ncp_max]

  filas_despues_reglas = len(df_tratado)
  print(f"   - Se eliminaron {filas_iniciales - filas_despues_reglas} filas con reglas de negocio.")

  # 2. Aplicar método IQR para el resto de variables numéricas
  print("\n2. Aplicando método IQR para el resto de variables numéricas...")

  variables_numericas = df_tratado.select_dtypes(include=np.number).columns
  columnas_para_iqr = [col for col in variables_numericas if col not in columnas_con_reglas]

  for col in columnas_para_iqr:
      Q1 = df_tratado[col].quantile(0.25)
      Q3 = df_tratado[col].quantile(0.75)
      IQR = Q3 - Q1
      limite_inferior = Q1 - iqr_factor * IQR
      limite_superior = Q3 + iqr_factor * IQR

      # Mantener solo las filas dentro de los límites
      df_tratado = df_tratado[(df_tratado[col] >= limite_inferior) & (df_tratado[col] <= limite_superior)]

  filas_despues_iqr = len(df_tratado)
  print(f"   - Se eliminaron {filas_despues_reglas - filas_despues_iqr} filas adicionales con el método IQR.")

  # 3. Reporte final
  display(Markdown("### Reporte Final de Eliminación"))

  filas_finales = len(df_tratado)
  filas_eliminadas = filas_iniciales - filas_finales
  porcentaje_eliminado = (filas_eliminadas / filas_iniciales) * 100

  print(f"Filas iniciales: {filas_iniciales:,}")
  print(f"Filas finales:   {filas_finales:,}")
  print(f"Total de filas eliminadas: {filas_eliminadas:,} ({porcentaje_eliminado:.2f}%)")

  display(Markdown("---"))
  print("\nProceso de eliminación de atípicos finalizado.")

  return df_tratado

# Guardar dataframe ===============================================================================================================
def guardar_dataframe(df: pd.DataFrame, file_name: Text, destination_path: Text = '../../data/') -> bool:
    """
    Guarda un DataFrame de pandas en un archivo CSV en la ruta destino especificada.

    Parámetros (Input):
        df (pd.DataFrame): El DataFrame de pandas a guardar (el dataset limpio).
        file_name (str): El nombre del archivo CSV de salida (ej: 'dataset_limpio.csv').
        destination_path (str): La ruta o directorio donde se guardará el archivo.
                                (Default: '../../data/')

    Retorno (Output):
        bool: True si el archivo se guardó y verificó correctamente, False en caso de error.
    """
    # 1. Construir la ruta completa del archivo
    # Usamos os.path.join para construir la ruta de manera segura en cualquier sistema operativo
    ruta_completa = os.path.join(destination_path, file_name)

    # 2. Guardar el DataFrame
    try:
        # El argumento index=False evita guardar el índice de Pandas como una columna.
        df.to_csv(ruta_completa, index=False)

        # 3. Mensaje de confirmación y verificación
        print(f"DataFrame guardado exitosamente en: {ruta_completa}")

        if os.path.exists(ruta_completa):
            print("Verificación local: El archivo se ha creado correctamente.")
            return True
        else:
            print("Error: El archivo de salida no pudo ser verificado.")
            return False

    except Exception as e:
        # Manejo de excepciones para errores de escritura, permisos, o ruta inexistente
        print(f"Error al guardar el archivo: {e}")
        return False