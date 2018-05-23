requirements:
	pip install -r requirements.txt

clean:
	yapf -ir sct/ tests/

train:
	python -m sct.train

check:
	mypy --ignore-missing-imports .
	flake8 .

test:
	python -m pytest tests/

job:
	source ~/git/cil/venv/bin/activate && bsub -W 04:00 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" python -m sct.train $(filter-out $@,$(MAKECMDGOALS))

%:
	@:

status:
	watch -n 1 bbjobs

output:
	watch -n 1 bpeek

.PHONY: requirements train check test job bbjobs output
