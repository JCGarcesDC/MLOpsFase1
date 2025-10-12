import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preparar_datos_para_modelado(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
  """
  Prepara los datos para el modelado de Machine Learning.

  Esta función realiza los siguientes pasos:
  1.  Separa las características (X) y la variable objetivo (y).
  2.  Divide los datos en conjuntos de entrenamiento y prueba de forma estratificada.
  3.  Crea un preprocesador (ColumnTransformer) con pipelines para variables
      numéricas (imputación + escalado) y categóricas (imputación + one-hot encoding).
  4.  Ajusta (fits) el preprocesador SÓLO con los datos de entrenamiento para evitar fuga de datos.

  Args:
      df (pd.DataFrame): El DataFrame limpio y con ingeniería de características.
      target_column (str): El nombre de la columna objetivo.
      test_size (float, optional): La proporción del dataset a reservar para la prueba. Defaults to 0.2.
      random_state (int, optional): Semilla para la reproducibilidad de la división. Defaults to 42.

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
      Un tuple que contiene:
          - X_train: Características de entrenamiento.
          - X_test: Características de prueba.
          - y_train: Variable objetivo de entrenamiento.
          - y_test: Variable objetivo de prueba.
          - preprocesador: El objeto ColumnTransformer ya ajustado a los datos de entrenamiento.
  """
  print("Iniciando la preparación de datos para el modelado...")

  # 1. Separar variables predictoras (X) y variable objetivo (y)
  X = df.drop(target_column, axis=1)
  y = df[target_column]

  # 2. Dividir en conjuntos de entrenamiento y prueba de forma estratificada
  X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=test_size,
      random_state=random_state,
      stratify=y
  )
  print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

  # 3. Identificar columnas numéricas y categóricas desde el conjunto de entrenamiento
  num_cols = X_train.select_dtypes(include=np.number).columns
  cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

  # 4. Crear los pipelines de preprocesamiento
  pipeline_numerico = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())
  ])

  pipeline_categorico = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
  ])

  # 5. Unir los pipelines en el ColumnTransformer
  preprocesador = ColumnTransformer(
      transformers=[
          ('num', pipeline_numerico, num_cols),
          ('cat', pipeline_categorico, cat_cols)
      ],
      remainder='passthrough'
  )

  # 6. Ajustar el preprocesador SÓLO con los datos de entrenamiento para evitar data leakage
  preprocesador.fit(X_train)

  print("\nPreprocesador creado y ajustado a los datos de entrenamiento exitosamente.")

  return X_train, X_test, y_train, y_test, preprocesador