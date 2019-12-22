import pandas as pd

df = pd.read_csv('./resources/all.csv', lineterminator='\n').astype(str)
df.drop(['id'], axis=1, inplace=True)
df.drop(['label'], axis=1, inplace=True)
df = df.to_csv('./resources/embedTrain.csv', header=0, index=0)
