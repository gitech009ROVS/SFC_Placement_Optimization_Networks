.PHONY: install install-dev test lint sim demo clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,plot]"

install-torch:
	pip install -e ".[torch,dev,plot]"

test:
	pytest

lint:
	ruff check src tests

demo:
	python -m sfc_autoscale.cli demo

sim:
	python -m sfc_autoscale.cli run --config configs/default.yaml

compare:
	python -m sfc_autoscale.cli compare --config configs/default.yaml

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache results
	find . -type d -name __pycache__ -exec rm -rf {} +
