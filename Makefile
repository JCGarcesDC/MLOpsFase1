.PHONY: clean data lint requirements sync_data_down sync_data_up test_environment

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = obesitymine
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
lint:
	flake8 src
	isort --check --diff --profile black src
	black --check --config pyproject.toml src

## Format source code with black
format:
	isort --profile black src
	black --config pyproject.toml src

## Download Data from DVC remote
sync_data_down:
	dvc pull

## Upload Data to DVC remote
sync_data_up:
	dvc push

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda env create -f environment.yml
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m venv venv
	@echo ">>> New virtualenv created. Activate with:\nsource venv/bin/activate"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Run tests
test:
	pytest tests/

## Run all notebooks
run_notebooks:
	jupyter nbconvert --to notebook --execute notebooks/**/*.ipynb

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

#################################################################################
# Self Documenting Commands                                                    #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
