import pandas as pd

def calcular_imc(df: pd.DataFrame) -> pd.DataFrame:
  """
  Calcula el Índice de Masa Corporal (IMC) y lo añade como una nueva columna.

  Args:
      df (pd.DataFrame): El DataFrame de entrada que contiene 'weight' y 'height'.

  Returns:
      pd.DataFrame: El DataFrame con la nueva columna 'imc'.
  """
  # Se crea una copia para no modificar el DataFrame original
  df_copy = df.copy()

  # Verificar que las columnas necesarias existan
  if 'weight' in df_copy.columns and 'height' in df_copy.columns:
      df_copy['imc'] = df_copy['weight'] / (df_copy['height'] ** 2)
      print("Columna 'imc' creada exitosamente.")
  else:
      print("Advertencia: No se encontraron las columnas 'weight' y/o 'height'. La columna 'imc' no fue creada.")

  return df_copy