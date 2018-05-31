requirements:
	pip install -r requirements.txt

clean:
	yapf -ir sct/ tests/

train:
	python -m sct.train $(filter-out $@,$(MAKECMDGOALS))

check:
	mypy --ignore-missing-imports .
	flake8 .

test:
	python -m pytest tests/

job:
	bsub -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" python -m sct.train $(filter-out $@,$(MAKECMDGOALS))

%:
	@:

status:
	watch -n 1 bbjobs

output:
	bpeek -f

.PHONY: requirements train check test job status output
