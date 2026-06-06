.PHONY: install install-dev train predict test lint clean api notebook

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,api]"

train:
	ola-lstm train --output artifacts

predict:
	ola-lstm predict --artifacts artifacts --hour 8 --day 0 --month 6

test:
	pytest tests/ -v

lint:
	ruff check src tests

clean:
	rm -rf artifacts/ .pytest_cache/ .ruff_cache/ htmlcov/ dist/ build/ *.egg-info

api:
	uvicorn ola_lstm.api:app --reload --host 0.0.0.0 --port 8000

notebook:
	jupyter notebook notebooks/Ola_Bike_Ride_Forecast_LSTM.ipynb
