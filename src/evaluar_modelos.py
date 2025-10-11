import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Tuple, Any, List, Dict, Text
import joblib
import os

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
# Métricas de rendimiento
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

def cargar_artefactos_ml(
    base_path: Text = '../../artefactos/',
    results_name: Text = 'mejores_modelos.csv',
    pipelines_name: Text = 'pipelines_optimizados.joblib'
) -> Tuple[pd.DataFrame, dict]:
    """
    Carga los resultados de la optimización y los modelos (pipelines) previamente guardados.

    Args:
        base_path (str): Directorio base donde se encuentran los archivos.
        results_name (str): Nombre del archivo CSV para los resultados.
        pipelines_name (str): Nombre del archivo joblib para los pipelines.

    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame de resultados y diccionario de pipelines.
    """
    ruta_resultados = os.path.join(base_path, results_name)
    ruta_pipelines = os.path.join(base_path, pipelines_name)
    
    df_cargado = None
    pipelines_cargados = {}
    
    # Cargar resultados
    try:
        df_cargado = pd.read_csv(ruta_resultados)
        print(f"Resultados cargados desde: {ruta_resultados}")
    except FileNotFoundError:
        print(f"Error: Archivo de resultados no encontrado en {ruta_resultados}")
        
    # Cargar pipelines
    try:
        pipelines_cargados = joblib.load(ruta_pipelines)
        print(f"Pipelines cargados desde: {ruta_pipelines}")
    except FileNotFoundError:
        print(f"Error: Archivo de pipelines no encontrado en {ruta_pipelines}")

    return df_cargado, pipelines_cargados



# Funcion para evaluar modelos 
def evaluar_mejor_modelo(
    df_resultados: pd.DataFrame, 
    pipelines: Dict[str, Pipeline], 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Pipeline:
  """
  Seleccionar el mejor modelo basado en el F1-Score, realiza una evaluación 
  detallada en el conjunto de prueba y grafica su matriz de confusión.

  Args:
      df_resultados (pd.DataFrame): DataFrame con los resultados de los modelos optimizados.
      pipelines (Dict[str, Pipeline]): Diccionario con los pipelines entrenados y optimizados.
      X_test (pd.DataFrame): Características del conjunto de prueba.
      y_test (pd.Series): Variable objetivo del conjunto de prueba.

  Returns:
      Pipeline: El objeto del pipeline del mejor modelo, listo para ser usado o guardado.
  """
  # 1. Seleccionar el mejor modelo basado en el F1-Score más alto
  mejor_modelo_nombre = df_resultados.loc[df_resultados['F1-Score (Test)'].idxmax()]['Modelo']
  mejor_modelo_pipeline = pipelines[mejor_modelo_nombre]

  print(f"El mejor modelo seleccionado es: {mejor_modelo_nombre}")

  # 2. Realizar predicciones finales con el mejor modelo
  predicciones_finales = mejor_modelo_pipeline.predict(X_test)

  # 3. Imprimir el reporte de clasificación detallado
  print(f"\n--- Reporte de Clasificación Final para: {mejor_modelo_nombre} ---")
  print(classification_report(y_test, predicciones_finales))

  # 4. Visualizar la Matriz de Confusión
  print("\n--- Matriz de Confusión ---")
  cm = confusion_matrix(y_test, predicciones_finales, labels=mejor_modelo_pipeline.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mejor_modelo_pipeline.classes_)

  fig, ax = plt.subplots(figsize=(8, 8))
  disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
  ax.set_title(f"Matriz de Confusión - {mejor_modelo_nombre} (Optimizado)", fontsize=10)
  plt.show()
  
  return mejor_modelo_pipeline