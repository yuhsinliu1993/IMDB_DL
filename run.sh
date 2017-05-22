#! /bin/bash

python imdb.py -m train -c lstm
python imdb.py -m train -c rnn
python imdb.py -m train -c gru
