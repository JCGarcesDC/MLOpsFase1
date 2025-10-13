import subprocess, shlex, sys
from pathlib import Path
import os

def run(cmd: str):
    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(cmd)

def save_csv_and_push(df, file_path="../data/obesity_estimation_cleaned.csv", commit_msg="Update cleaned dataset"):
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    
    # 1. Añadir archivos para el commit
    #run(f'dvc add "{p.as_posix()}"')
    #run(f'git add "{p.as_posix()}.dvc"')

    # 2. Hacer commit (si hay cambios)
    try:
        run(f'git commit -m "{commit_msg}"')
    except RuntimeError:
        print("➜ Git: sin cambios que commitear.")
    
    # 3. Subir datos a DVC
    run('dvc push')
    
    # 4. Sincronizar con el remoto ANTES de subir
    run('git pull --rebase origin main')
    
    # 5. Ahora sí, subir los cambios a Git
    run('git push -u origin main')