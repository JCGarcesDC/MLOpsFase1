import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
  """Carga de CSV y devolución de DataFrame."""
  path = Path(file_path)
  if not path.exists():
    raise FileNotFoundError(f"No se encontro el archivo: {file_path}")
  return pd.read_csv(path)
