#!/bin/sh -x

wget https://polybox.ethz.ch/index.php/s/l2wM4RIyI3pD7Tl/download -O stories.train.csv
wget https://polybox.ethz.ch/index.php/s/02IVLdBAgVcsJAx/download -O stories.eval.csv
wget https://polybox.ethz.ch/index.php/s/AKbA8g7SeHwjU0R/download -O stories.test.csv
wget https://polybox.ethz.ch/index.php/s/h2gp3FpS3N7Xgiq/download -O stories.spring2016.csv

mkdir $SCRATCH/st
cd $SCRATCH/st
# Download and extract the unidirectional model.
wget "http://download.tensorflow.org/models/skip_thoughts_uni_2017_02_02.tar.gz"
tar -xvf skip_thoughts_uni_2017_02_02.tar.gz
rm skip_thoughts_uni_2017_02_02.tar.gz
mv skip_thoughts_uni_2017_02_02 uni

# Download and extract the bidirectional model.
wget "http://download.tensorflow.org/models/skip_thoughts_bi_2017_02_16.tar.gz"
tar -xvf skip_thoughts_bi_2017_02_16.tar.gz
rm skip_thoughts_bi_2017_02_16.tar.gz
mv skip_thoughts_bi_2017_02_16 bi
