# Configuración del entorno conda

Este archivo explica cómo crear y activar el entorno conda para este proyecto en Windows PowerShell.

1. Abrir PowerShell (Windows PowerShell o Anaconda Prompt).

2. Para crear el entorno desde `environment.yml`:

```powershell
conda env create -f environment.yml
```

Esto creará un entorno llamado `obesitymine` (ver `name:` en `environment.yml`).

3. Activar el entorno:

```powershell
conda activate obesitymine
```

4. Comprobar las dependencias instaladas:

```powershell
python -c "import sys; import numpy; import pandas; print(sys.version); print(numpy.__version__); print(pandas.__version__)"
```

5. Notas:
- Si usas Anaconda/Miniconda, asegúrate de tener `conda` en el PATH o usar Anaconda Prompt.
- Para actualizar dependencias, edita `environment.yml` y ejecuta:

```powershell
conda env update -f environment.yml -n obesitymine
```

6. Alternativa si no tienes conda: crear un virtualenv y usar pip.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

## 2. Instalar el kernel Jupyter para el entorno

Ejecuta este comando una sola vez (con el entorno activado):

```powershell
python -m ipykernel install --user --name obesitymine --display-name "Python (obesitymine)"
```

## 3. Seleccionar el kernel en Jupyter/VS Code

- Al abrir un notebook, selecciona el kernel llamado "Python (obesitymine)" en la barra superior.
- Así todos los notebooks usarán el entorno y dependencias configuradas.

## 4. Ubicación de los datos

- Coloca tus archivos CSV originales en la carpeta `data/raw/`.
- Los scripts de procesamiento deben leer desde `data/raw/` y guardar los datos limpios en `data/processed/`.
