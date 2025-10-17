import importlib
import sys

packages = [
    'numpy',
    'pandas',
    'sklearn',
    'matplotlib',
    'seaborn',
    'xgboost',
    'joblib',
]

print('Python:', sys.version.replace('\n', ' '))
for p in packages:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', 'unknown')
        print(f'{p}: OK ({v})')
    except Exception as e:
        print(f'{p}: MISSING ({e.__class__.__name__})')
