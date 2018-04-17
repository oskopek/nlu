#!/bin/sh -x
wget https://polybox.ethz.ch/index.php/s/qUc2NvUh2eONfEB/download -O data.tar
tar -xvf data.tar
wget https://polybox.ethz.ch/index.php/s/cpicEJeC2G4tq9U/download -O pretrained_embedding
wget https://polybox.ethz.ch/index.php/s/HJUnOuIj3K4FEdT/download -O sentences.test
