from setuptools import find_packages, setup

setup(
    name='obesitymine',
    packages=find_packages(),
    version='0.1.0',
    description='MLOps project for obesity estimation using machine learning',
    author='Juan Carlos Garces',
    license='MIT',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'mlflow>=2.7.0',
        'dvc>=2.0.0',
        'xgboost>=1.5.0',
        'lightgbm>=3.3.0',
        'catboost>=1.0.0',
        'optuna>=3.0.0',
        'shap>=0.41.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'plotly>=5.3.0',
    ],
    python_requires='>=3.11',
)
