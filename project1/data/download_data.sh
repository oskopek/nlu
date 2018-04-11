#!/bin/sh -x
wget https://polybox.ethz.ch/index.php/s/qUc2NvUh2eONfEB/download -O data.tar
tar -xvf data.tar
wget https://polybox.ethz.ch/index.php/s/cpicEJeC2G4tq9U/download -O pretrained_embeddings
