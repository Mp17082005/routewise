.PHONY: install data validate test lint eval demo all clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

data:
	routewise generate-data --seed 42

validate:
	routewise validate

test:
	$(PY) -m pytest --cov=routewise --cov-report=term-missing

lint:
	ruff check src tests app
	ruff format --check src tests app

eval:
	routewise evaluate --out reports
	routewise sensitivity --shipment S004 --criterion cost
	routewise recommend-all --out reports/recommendations.csv

demo:
	streamlit run app/streamlit_app.py

# one command regenerates data, results and plots
all: data validate test eval
