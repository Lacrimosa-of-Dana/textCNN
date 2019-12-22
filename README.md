# TextCNN Algorithm

## Dependencies
Tensorflow 2.0 (compat to version 1.x)

python 3.7

## Guide
Dataset in ./resources/all.csv

Word2Vec model in ./models

Dataset can be set in getTrain.py, word2vec can be finished in wordEmbedding.py after prepareEmbed.py (which is to remove useless labels). Use preparation.py to get the index of the vocabulary, and run train.py to train the model.

Run test.py to test the model with test dataset.

