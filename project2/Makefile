requirements:
	pip install -r requirements.txt --user
	python -c 'import nltk; nltk.download("punkt")'
	cd data && ./download_data.sh

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
	bsub -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" python -m sct.train

%:
	@:

status:
	watch -n 1 bbjobs

output:
	bpeek -f

run_all:
	./run_all.sh

.PHONY: requirements train check test job status output run_all
