# Project 2

## Project structure

* `data/` - Data folder. Contains a script necessary for downloading the datasets, and the downloaded data.
* `notebooks/` - Some experimental Jupyter notebooks, not used.
* `outputs/` - Log directory, organized per model & per run.
* `report/` - The LaTeX report and the generated PDF.
* `sct/` - Source folder (stands for Story Cloze Task).
  * `data/` - Data loading, preprocessing, batching, and pre-trained embeddings.
    * `skip_thoughts/` - External module from [TensorFlow Models](https://github.com/tensorflow/models/tree/master/research/skip_thoughts), implemented by Chris Shallue according to *Jamie Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler. Skip-Thought Vectors. In NIPS, 2015*.
  * `experiments/` - Frozen flags for experiments presented in the report. See `experiment.sh` in the project directory.
  * `model/` - Model directory. Note that models heavily use inheritance!
  * `flags.py` - Hyperparameters and settings. To be replaced by flags from `experiments/` for report-related experiments.
  * `train.py` - Main executable file. Reads flags and runs the corresponding training and/or evaluation.
* `tests/` - (Very few) Unit tests.
* `experiments.sh` - Bash utility script to reproduce experiments from the report.
* `Makefile` - Defines "aliases" for various tasks.
* `README.md` - This manual.
* `requirements.txt` - Required Python packages.

## Set up

* Install Python 3.6+.
* Make sure your `SCRATCH` environmental variable is set to a folder
which has enough memory for pre-trained embeddings (~15GB),
and the device on which this project folder is should have enough
space to store training/evaluation/testing data + model checkpoints and logs (~5GB).
* Run the setup:
    ```
    make requirements
    ```

## Run experiments

Adjust the directories/other flags in the given experiment file, and run:

```
./experiment.sh sct/experiments/<NAME_OF_EXPERIMENT>.py
```

## Loading and saving models

During training, models are saved automatically when a new best
evaluation score is obtained. How often evaluation is performed
can be set using the flag `evaluate_every_steps`.

You can load an existing checkpoint by running:
```
./experiment.sh sct/experiments/<NAME_OF_EXPERIMENT>.py --load_checkpoint outputs/MODEL_DIR/checkpoints/model.ckpt-NUMBER
```
The story cloze test 2016 accuracy is printed at the end of standard output, and the predictions for both this and the ETH test set can be found in `outputs/NEW_MODEL_DIR/predictions/`.

## Other make commands

```
make clean  # format source code
make check  # check formatting and types
make job    # train on Leonhard with sct/flags.py as flags (must be logged in)
make output # watch job output on Leonhard
make status # watch job status on Leonhard
make test   # run unit tests (not used extensively)
make train  # train locally with sct/flags.py as flags
make runall # run all experiments in sct/experiments one after each other (currently only roemmele*)
```
