# ObesityMine53 - MLOps Project

## Project Structure
```
├── config/             # Configuration files
├── data/              # Data files
│   ├── raw/           # Original data
│   ├── processed/     # Cleaned data
│   └── interim/       # Intermediate data
├── models/            # Saved models
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── utils/        # Utility functions
│   ├── train.py      # Training pipeline
│   └── predict.py    # Prediction service
└── tests/            # Unit tests
```

## Setup

### 1. Environment Setup

Using Conda (recommended):
```bash
conda env create -f environment.yml
conda activate obesitymine
```

Or using pip:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### 2. MLflow Setup

The project uses MLflow for experiment tracking. MLflow UI can be started with:
```bash
mlflow ui
```

Visit http://localhost:5000 to access the MLflow UI.

### 3. Databricks Integration

1. Configure Databricks CLI:
```bash
databricks configure --token
```

2. Update workspace URL and cluster ID in `config/config.yaml`

### 4. Development Workflow

1. Update configuration in `config/config.yaml`
2. Develop in notebooks/
3. Move stable code to src/
4. Track experiments with MLflow
5. Use pre-commit hooks for code quality

## Usage

### Training

```bash
python src/train.py
```

### Testing

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black .
# Sort imports
isort .
# Lint
flake8
```

## MLOps Features

- Experiment Tracking: MLflow
- Model Registry: MLflow
- Data Version Control: DVC
- Code Quality: black, flake8, isort
- Testing: pytest
- Containerization: Docker (coming soon)
- CI/CD: GitHub Actions (coming soon)
- Cloud Deployment: Databricks
- Monitoring: MLflow + Custom Metrics

## Contributing

1. Create a new branch
2. Make changes
3. Run tests
4. Submit PR

## License

[Your License Here]