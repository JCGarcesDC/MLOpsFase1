# **Fase 1: Manipulación y preparación de datos**
## **Objetivo de este Notebook**

Estas carpeta documenta el proceso inicial y fundamental de **Análisis Exploratorio de Datos (EDA)** y preparación de datos para el proyecto de estimación de niveles de obesidad. El propósito es comprender a fondo el dataset, asegurar su calidad y transformarlo en un formato adecuado para el entrenamiento de modelos de Machine Learning.

Las tareas clave que se realizarán en este documento son:

1.   **Importar datos y Análisis Inicial**: Inspeccionar la estructura, tipos de datos y estadísticas descriptivas del dataset.
2.   **Limpieza de Datos**: Identificar y manejar valores nulos, duplicados, inconsistencias o posibles outliers.
3.  **Análisis Exploratorio (EDA)**: Utilizar visualizaciones para descubrir patrones, distribuciones y relaciones entre las variables.
4.  **Versionado de Datos con DVC**: Aplicar DVC (Data Version Control) para registrar las versiones del dataset (crudo y limpio), garantizando la trazabilidad y reproducibilidad de nuestro trabajo.

Esta fase es crucial, ya que la calidad de los datos impacta directamente en el rendimiento y la fiabilidad de cualquier modelo que construyamos posteriormente.

## **Descripción del Problema**

El dataset "Estimation of Obesity Levels Based On Eating Habits and Physical Condition" contiene datos de individuos de México, Perú y Colombia, con atributos relacionados con hábitos alimenticios, condición física y estilo de vida. El objetivo es predecir el nivel de obesidad de una persona, clasificado en siete categorías: **Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, y Obesity Type III.**

## **Propuesta de Valor**

Desarrollar un modelo de clasificación que permita identificar el nivel de obesidad de un individuo a partir de sus hábitos y características físicas. Esto puede ser útil para sistemas de recomendación, monitoreo de salud preventiva y campañas de concientización.

## **Herramientas y Documentación**

Para la manipulación y análisis de datos se utilizarán principalmente las librerías de Python **Pandas, Matplotlib y Seaborn**. El versionado de los artefactos de datos se gestionará con **DVC**.

Para la documentación estratégica del proyecto, incluyendo el mapeo de requerimientos, stakeholders y métricas de éxito, se utilizó el framework **ML Canvas.**
