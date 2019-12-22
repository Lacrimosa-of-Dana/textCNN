import os

import numpy as np
import pandas as pd


def split_dataset(path):
    df = pd.read_csv(os.path.join(path, 'all.csv'), lineterminator='\n')
    df = df.sample(frac=1, random_state=41)
    total = len(df)
    train = int(total * 0.8)
    train_df = df[:train]
    test_df = df[train:]
    train_df.to_csv(os.path.join(path, 'train.csv'))
    test_df.to_csv(os.path.join(path, 'test.csv'))


def word_to_index(path):
    df = pd.read_csv(os.path.join(path, 'all.csv'), lineterminator='\n')
    word_dict = {'unknown': 0}
    data = list(df['review'])
    index = 1
    for s_index, sentence in enumerate(data):
        for w_index, word in enumerate(sentence.split()):
            if word_dict.get(word) is None:
                word_dict[word] = index
                index += 1
    return word_dict


def load_data(path, max_word_per_sentence=295):
    df = pd.read_csv(path, lineterminator='\n').astype(str)
    labels = np.array([[0, 1] if i == 'positive' else [1, 0] for i in list(df['label'])])
    data = list(df['review'])
    word_dict = word_to_index('./resources')
    x = np.zeros([len(data), max_word_per_sentence], dtype=np.int32)
    for s_index, sentence in enumerate(data):
        for w_index, word in enumerate(sentence.split()):
            x[s_index][w_index] = word_dict.get(word)

    return data, labels, word_dict, np.array(x)


def embed_to_tensor(embed_model, word_dict):
    tensor_model = np.zeros([len(word_dict), 100])
    for word, index in word_dict.items():
        try:
            tensor_model[index] = embed_model.wv[word]
        except KeyError as e:
            tensor_model[index] = np.zeros(100)
    return tensor_model


def batch_iter(data, batch_size, epoch_num, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches = int((len(data)-1)/batch_size) + 1
    for epoch in range(epoch_num):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    split_dataset('./resources')