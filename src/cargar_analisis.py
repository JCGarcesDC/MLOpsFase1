# Manipulación de datos
import pandas as pd
from typing import Text, Optional
from IPython.display import display, Markdown

# Cargar los datos ===============================================================================================================
def cargar_dataframe(path_data: Text) -> Optional[pd.DataFrame]:
    """
    Carga un archivo CSV desde la ruta especificada y lo convierte en un DataFrame de pandas.

    Esta función utiliza un bloque try-except para manejar el error común de
    'FileNotFoundError' si la ruta del archivo es incorrecta o no existe.

    Parámetros (Input):
        path_data (str): La ruta (path) completa al archivo CSV que se desea cargar.
                         Ejemplo: '../../data/obesity_estimation_modified.csv'

    Retorno (Output):
        pd.DataFrame or None: El DataFrame de pandas si el archivo se carga correctamente,
                              o None si ocurre un 'FileNotFoundError'.
    """
    try:
        # Leer el archivo CSV en un DataFrame de pandas
        df = pd.read_csv(path_data)
        print(f"Archivo CSV cargado exitosamente desde: {path_data}")
        return df

    except FileNotFoundError:
        # Manejo del error si la ruta es incorrecta o el archivo no existe
        print(f"Error: No se encontró el archivo en la ruta: {path_data}")
        print("Por favor, verifica la ruta o si el archivo existe.")
        return None # Retorna None si la carga falla

# Generar variables numericas y categoricas ===============================================================================================================
def crear_listas_variables():
    """
    Crea y devuelve tres objetos (listas y una cadena de texto) que contienen
    los nombres de las variables clasificadas por su tipo y la variable objetivo
    para un conjunto de datos específico.

    Parámetros (Input):
        Esta función no requiere ningún parámetro de entrada.

    Retorno (Output):
        tuple: Una tupla que contiene los nombres de las tres variables creadas
               en el siguiente orden:
               1. variables_numericas (list): Nombres de las variables cuantitativas.
               2. variables_categoricas (list): Nombres de las variables cualitativas/categóricas.
               3. variable_objetivo (str): Nombre de la variable dependiente o objetivo.
    """
    # Crear una lista con las variables númericas (cuantitativas)
    variables_numericas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'mixed_type_col']

    # Crear una lista con las variables cualitativas (categóricas)
    variables_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

    # Definir la variable objetivo (dependiente)
    variable_objetivo = 'NObeyesdad'

    return variables_numericas, variables_categoricas, variable_objetivo

# Función para analisis Inicial ===============================================================================================================
def resumen_eda(df: pd.DataFrame, variables_numericas, variables_categoricas, target_column: str = None):

  """
    Realiza un análisis exploratorio de datos inicial y completo sobre un DataFrame.

    Esta función imprime un resumen que incluye:
    1. Dimensiones del DataFrame.
    2. Tipos de datos y uso de memoria.
    3. Una muestra aleatoria de los datos.
    4. Conteo de valores nulos y filas duplicadas.
    5. Estadísticas descriptivas para variables numéricas y categóricas por separado.
    6. Distribución de la variable objetivo (si se especifica).

    Args:
        df (pd.DataFrame): El DataFrame que se va a analizar.
        variables_numericas (list): Una lista con los nombres de las columnas numéricas.
        variables_categoricas (list): Una lista con los nombres de las columnas numéricas.
        target_column (str, optional): El nombre de la columna objetivo.
                                       Si se proporciona, se mostrará su distribución.
                                       Defaults to None.
    """

  # Imprime un título principal para el reporte
  display(Markdown("---"))
  display(Markdown("## **Análisis Exploratorio del Dataset**"))
  display(Markdown("---"))

  # 1. Dimensiones del DataFrame
  display(Markdown("### 1. Dimensiones del Dataset"))
  print(f"Número de Filas:    {df.shape[0]:,}")
  print(f"Número de Columnas: {df.shape[1]}")
  print("\n")

  # 2. Tipos de datos y memoria
  display(Markdown("### 2. Tipos de Datos y Uso de Memoria"))
  df.info()
  print("\n")

  # 3. Muestra aleatoria de los datos
  display(Markdown("### 3. Muestra Aleatoria de Datos"))
  display(df.sample(5))
  print("\n")

  # 4. Calidad de los Datos
  display(Markdown("### 4. Calidad de los Datos"))
  nulos = df.isnull().sum()
  duplicados = df.duplicated().sum()
  print(f"Número total de filas duplicadas: {duplicados}")
  print("Conteo de valores nulos por columna:")
  # Muestra solo las columnas que tienen valores nulos para no saturar la salida
  if nulos.sum() == 0:
      print("No se encontraron valores nulos.")
  else:
      print(nulos[nulos > 0])
  print("\n")

  # 5. Estadísticas Descriptivas
  display(Markdown("### 5. Estadísticas Descriptivas"))

  # Columnas numéricas
  display(Markdown("#### **Variables Numéricas**"))
  display(df[variables_numericas].describe().T)

  # Columnas categóricas
  display(Markdown("#### **Variables Categóricas**"))
  display(df[variables_categoricas].describe(include=['object', 'category']).T)
  print("\n")

  # 6. Análisis de la Variable Objetivo
  if target_column:
      if target_column in df.columns:
          display(Markdown(f"### 6. Distribución de la Variable Objetivo: '{target_column}'"))
          distribucion = pd.DataFrame({
              'Frecuencia': df[target_column].value_counts(),
              'Porcentaje (%)': df[target_column].value_counts(normalize=True).mul(100).round(2)
          })
          display(distribucion)
      else:
          print(f"Advertencia: La columna objetivo '{target_column}' no se encontró en el DataFrame.")