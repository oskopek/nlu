#!/bin/sh -x
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

wget https://polybox.ethz.ch/index.php/s/l2wM4RIyI3pD7Tl/download -O stories.train.csv
wget https://polybox.ethz.ch/index.php/s/02IVLdBAgVcsJAx/download -O stories.eval.csv
# Download word2vec embeddings from https://code.google.com/archive/p/word2vec/ (https://arxiv.org/pdf/1301.3781.pdf) trained on GoogleNews
gdrive_download 0B7XkCwpI5KDYNlNUTTlSS21pQmM 'w2vGoogleNews.bin.gz'
gunzip 'w2vGoogleNews.bin.gz'
