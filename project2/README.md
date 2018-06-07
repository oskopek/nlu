# Project 2

TODO: Enhance the text here, or add to paper, or both.

## Set up

Install Python 3.6+. Install the following packages using `pip`:
```
pip install -r requirements.txt --user
```
or
```
make requirements
```

## Run experiments

Adjust the directories/other flags in the given experiment file, and run:

```
./experiment.sh sct/experiments/<NAME_OF_EXPERIMENT>.py
```

## Clean, check, ...

```
make clean  # format
make check  # check formatting and types
make test   # run tests
make train  # train locally with sct/flags.py as flags
make job    # train on Leonhard with sct/flags.py as flags (must be logged in)
make status # watch job status on Leonhard
make output # watch job output on Leonhard
```
