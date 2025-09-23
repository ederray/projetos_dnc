#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Formata o código-fonte com isort e black
.PHONY: format
format:
	isort project_name
	black project_name

## Verifica o código usando flake8, black e isort (use `make format` para corrigir)
.PHONY: check
check:
	@echo "Verificando a formatação e o estilo do código..."
	black --check --diff project_name
	isort --check --diff project_name
	flake8 project_name

## Formata um único arquivo
.PHONY: format-file
format-file:
	@echo "Formatando o arquivo $(FILE)..."
	isort $(FILE)
	black $(FILE)
	@echo "Formatação de $(FILE) concluída."

## Verifica um único arquivo
.PHONY: check-file
check-file:
	@echo "Verificando o arquivo $(FILE)..."
	black --check --diff $(FILE)
	isort --check --diff $(FILE)
	flake8 $(FILE)
	@echo "Verificação de $(FILE) concluída."

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Executar o pré-processamento dos dados.
PYTHON_INTERPRETER = python

.PHONY: train

train:
	@echo "Executando o script de pré-processamento e visualização..."
	set PYTHONPATH=src && python src/models/run_train.py

#################################################################################
# Self Documenting Commands                                                     #
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
