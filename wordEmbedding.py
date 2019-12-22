import os

import gensim
import pandas as pd


def prepare_embed(path):
    df = pd.read_csv(os.path.join(path, 'all.csv'), lineterminator='\n').astype(str)
    df.drop(['id'], axis=1, inplace=True)
    df.drop(['label'], axis=1, inplace=True)
    df = df.to_csv(os.path.join(path, 'embedTrain.csv'), header=0, index=0)


if __name__ == '__main__':
    path = './resources'
    prepare_embed(path)
    sentences = gensim.models.word2vec.LineSentence(os.path.join(path, 'embedTrain.csv'))
    model = gensim.models.Word2Vec(sentences, sg=1, min_count=1)
    model.save('./models/embedModel')
