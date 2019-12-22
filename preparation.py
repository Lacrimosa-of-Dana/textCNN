import pandas as pd


def split_dataset(df):
    df = df.sample(frac=1, random_state=41)
    total = len(df)
    train = int(total * 0.8)
    train_df = df[:train]
    test_df = df[train:]
    train_df.to_csv('./resources/train.csv')
    test_df.to_csv('./resources/test.csv')


def word_to_index():
    df = pd.read_csv('./resources/all.csv', lineterminator='\n')
    word_dict = {'unknown': 0}
    data = list(df['review'])
    index = 1
    for s_index, sentence in enumerate(data):
        for w_index, word in enumerate(sentence.split()):
            if word_dict.get(word) is None:
                word_dict[word] = index
                index += 1
    return word_dict
