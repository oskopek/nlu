requirements:
	pip install -r requirements.txt

train:
	python -m sct.train

check:
	mypy --ignore-missing-imports .
	flake8 .

test:
	python -m pytest tests/

.PHONY: requirements train check test