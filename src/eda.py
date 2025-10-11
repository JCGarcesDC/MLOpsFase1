import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import chi2_contingency

# EDA variables numericas ===============================================================================================================
def analisis_exploratorio_numerico(df: pd.DataFrame, num_cols: list, target_col: str):
  """
  Realizar un completo análisis exploratorio (EDA) para las variables numéricas de un DataFrame.

  Esta función genera y muestra:
  1. Un resumen estadístico, incluyendo asimetría y curtosis.
  2. Histogramas y diagramas de caja para cada variable (análisis univariado).
  3. Un mapa de calor de correlación entre las variables numéricas.
  4. Un análisis de la relación entre cada variable numérica y la variable objetivo.

  Args:
      df (pd.DataFrame): El DataFrame a analizar.
      num_cols (list): Una lista con los nombres de las columnas numéricas.
      target_col (str): El nombre de la columna objetivo (categórica).
  """
  display(Markdown("---"))
  display(Markdown("## **Análisis Exploratorio de Variables Numéricas**"))
  display(Markdown("---"))

  # --- 1. Resumen Estadístico ---
  display(Markdown("### 1. Resumen Estadístico"))
  resumen = df[num_cols].describe().T
  resumen['skewness'] = df[num_cols].skew()
  resumen['kurtosis'] = df[num_cols].kurt()
  display(resumen)

  # --- 2. Análisis de Distribución (Univariado) ---
  display(Markdown("\n### 2. Distribución de Cada Variable Numérica"))
  n_filas = int(np.ceil(len(num_cols) / 4))
  fig, axes = plt.subplots(n_filas, 4, figsize=(14, n_filas * 3))
  axes = axes.flatten()

  for i, col in enumerate(num_cols):
      sns.histplot(df[col], ax=axes[i], bins=10, color = '#41abc0')
      axes[i].axvline(x = df[col].mean(), color='red', linestyle='-.')
      axes[i].set_title(f'Distribución de {col}', fontsize=10)

  # Ocultar ejes sobrantes si el número de variables es impar
  for j in range(len(num_cols), len(axes)):
      axes[j].set_visible(False)

  plt.tight_layout()
  plt.show()

  # --- 3. Análisis de Correlación entre Variables Numéricas ---
  display(Markdown("\n### 3. Mapa de Calor de Correlación"))
  plt.figure(figsize=(8, 6))
  correlation_matrix = df[num_cols + [target_col]].corr(method='pearson')
  sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
  plt.title('Correlación entre Variables Numéricas', fontsize=14)
  plt.show()

  # --- 4. Relación con la Variable Objetivo ---
  display(Markdown(f"\n### 4. Relación de Variables Numéricas con '{target_col}'"))

  # Tabla resumen con la media por categoría
  display(Markdown("#### **Media de cada variable por categoría de obesidad**"))
  media_por_categoria = df.groupby(target_col)[num_cols].mean().round(2)
  display(media_por_categoria)

  # Gráficos de caja para visualizar la distribución
  display(Markdown("#### **Distribución de cada variable por categoría de obesidad**"))
  n_filas_target = int(np.ceil(len(num_cols) / 3))
  fig, axes = plt.subplots(n_filas_target, 3, figsize=(15, n_filas_target * 4))
  axes = axes.flatten()

  # Ordenar las categorías de la variable objetivo de manera lógica
  order = sorted(df[target_col].unique())

  for i, col in enumerate(num_cols):
      sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], order=order, palette='viridis')
      axes[i].set_title(f'{col} vs. {target_col}', fontsize=12)
      axes[i].tick_params(axis='x', rotation=45)

  for j in range(len(num_cols), len(axes)):
      axes[j].set_visible(False)

  plt.tight_layout()
  plt.show()

# EDA variables categoricas ===============================================================================================================
def analisis_exploratorio_categorico(df: pd.DataFrame, cat_cols: list, target_col: str):
  """
  Realizar un completo análisis exploratorio (EDA) para las variables categóricas.

  Genera y muestra:
  1. Un resumen estadístico de las variables categóricas.
  2. Gráficos de barras para visualizar la distribución de cada variable.
  3. Gráficos de barras apiladas al 100% para analizar la proporción de la variable
      objetivo en cada categoría.
  4. Una prueba de Chi-cuadrado para determinar la significancia estadística de la
      asociación entre cada variable y el objetivo.

  Args:
      df (pd.DataFrame): El DataFrame a analizar.
      cat_cols (list): Lista con los nombres de las columnas categóricas.
      target_col (str): El nombre de la columna objetivo.
  """
  display(Markdown("---"))
  display(Markdown("## **Análisis Exploratorio de Variables Categóricas**"))
  display(Markdown("---"))

  # --- 1. Resumen Estadístico ---
  display(Markdown("### 1. Resumen Estadístico"))
  display(df[cat_cols].describe().T)

  # --- 2. Análisis de Distribución (Univariado) ---
  display(Markdown("\n### 2. Distribución de Cada Variable Categórica"))
  n_filas = int(np.ceil(len(cat_cols) / 4))
  fig, axes = plt.subplots(n_filas, 4, figsize=(16, n_filas * 3))
  axes = axes.flatten()

  for i, col in enumerate(cat_cols):
      sns.countplot(data=df, y=col, order=df[col].value_counts().index, ax=axes[i], color = '#41abc0')
      axes[i].set_title(f'Distribución de {col}', fontsize=10)
      axes[i].set_xlabel('Frecuencia')
      axes[i].set_ylabel('')

  # Ocultar ejes sobrantes
  for j in range(len(cat_cols), len(axes)):
      axes[j].set_visible(False)

  plt.tight_layout()
  plt.show()

  # --- 3. Relación con la Variable Objetivo (Gráficos de Proporción) ---
  display(Markdown(f"\n### 3. Relación con la Variable Objetivo: '{target_col}'"))
  n_filas = int(np.ceil(len(cat_cols) / 4))
  # Crear la figura y los ejes (subplots)
  fig, axes = plt.subplots(n_filas, 4, figsize=(16, n_filas * 4))
  # Aplanar el array de ejes para poder iterar con un solo índice
  axes = axes.flatten()

  for i, col in enumerate(cat_cols):
      # Seleccionar el eje actual donde se va a graficar
      ax = axes[i]

      # Crear tabla de contingencia y normalizar para obtener porcentajes
      contingency_table = pd.crosstab(df[col], df[target_col], normalize='index') * 100

      # Graficar en el eje especificado (ax=ax)
      contingency_table.plot(kind='bar', stacked=True, ax=ax,  colormap='viridis', width=0.8)

      # Configurar títulos y etiquetas usando el objeto 'ax' para este subplot
      ax.set_title(f'Proporción de Obesidad por {col}', fontsize=10)
      ax.set_xlabel('') # El nombre de la columna ya es visible en el título
      ax.set_ylabel('Porcentaje (%)')
      ax.tick_params(axis='x', rotation=0) # Rotar etiquetas si son largas
      ax.legend(title=target_col, fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left') # Ajustar la leyenda para el subplot

  # Ocultar los ejes sobrantes si el número de gráficos es impar
  for j in range(len(cat_cols), len(axes)):
      axes[j].set_visible(False)

  # Ajustar el layout para evitar solapamientos y mostrar la figura completa
  plt.tight_layout()
  plt.show()

  # --- 4. Prueba de Asociación Estadística (Chi-Cuadrado) ---
  display(Markdown("\n### 4. Prueba de Asociación Estadística (Chi-Cuadrado)"))

  chi2_results = []
  for col in cat_cols:
      contingency_table = pd.crosstab(df[col], df[target_col])
      chi2, p_value, _, _ = chi2_contingency(contingency_table)
      chi2_results.append({'Variable': col, 'Chi2 Statistic': chi2, 'P-Value': p_value})

  results_df = pd.DataFrame(chi2_results)
  results_df['Asociación Significativa (p < 0.05)'] = results_df['P-Value'] < 0.05

  display(results_df.sort_values(by='P-Value'))

# EDA variable objetivo ===============================================================================================================
def analisis_variable_objetivo(df: pd.DataFrame, target_col: str, num_cols: list, cat_cols: list):
  """
  Realiza un análisis profundo y detallado de la variable objetivo categórica.

  Este análisis incluye:
  1.  Distribución de clases para identificar desbalances.
  2.  Perfil categórico (moda) para cada clase.

  Args:
      df (pd.DataFrame): El DataFrame a analizar.
      target_col (str): El nombre de la columna objetivo.
      num_cols (list): Lista de columnas numéricas para el perfilado.
      cat_cols (list): Lista de columnas categóricas para el perfilado.
  """
  display(Markdown("---"))
  display(Markdown(f"## **Análisis de la Variable Objetivo: '{target_col}'**"))
  display(Markdown("---"))

  # --- 1. Análisis de Distribución y Desbalance ---
  display(Markdown("### 1. Distribución de las Clases"))

  # Gráfico de barras
  plt.figure(figsize=(10, 5))
  sns.countplot(data=df, y=target_col, order=df[target_col].value_counts().index, palette='viridis')
  plt.title(f'Distribución de Frecuencias de {target_col}', fontsize=14)
  plt.xlabel('Categoría', fontsize=10)
  plt.ylabel('Frecuencia', fontsize=10)
  plt.xticks(rotation=45)
  plt.show()

  # --- 2. Perfil Categórico por Categoría ---
  display(Markdown("\n### 2. Perfil Categórico Más Común (Moda) por Categoría"))
  display(Markdown("Esta tabla muestra el hábito o característica más frecuente para cada nivel de obesidad."))

  # Usamos una función lambda para obtener la moda de cada grupo
  perfil_categorico = df.groupby(target_col)[cat_cols].agg(lambda x: x.mode()[0])
  display(perfil_categorico)