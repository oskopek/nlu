#!/bin/bash
set +e

wget https://polybox.ethz.ch/index.php/s/l2wM4RIyI3pD7Tl/download -O stories.train.csv
wget https://polybox.ethz.ch/index.php/s/02IVLdBAgVcsJAx/download -O stories.eval.csv
wget https://polybox.ethz.ch/index.php/s/6RELTQ3CapEBSKE/download -O stories.test.csv
wget https://polybox.ethz.ch/index.php/s/h2gp3FpS3N7Xgiq/download -O stories.spring2016.csv

mkdir -p $SCRATCH/st
cd $SCRATCH/st
# Download and extract the unidirectional model.
wget "http://download.tensorflow.org/models/skip_thoughts_uni_2017_02_02.tar.gz"
tar -xvf skip_thoughts_uni_2017_02_02.tar.gz
rm -rf skip_thoughts_uni_2017_02_02.tar.gz uni
mv skip_thoughts_uni_2017_02_02 uni

# Download and extract the bidirectional model.
wget "http://download.tensorflow.org/models/skip_thoughts_bi_2017_02_16.tar.gz"
tar -xvf skip_thoughts_bi_2017_02_16.tar.gz
rm -rf skip_thoughts_bi_2017_02_16.tar.gz bi
mv skip_thoughts_bi_2017_02_16 bi
