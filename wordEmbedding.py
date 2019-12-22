import gensim

path = './resources/embedTrain.csv'
sentences = gensim.models.word2vec.LineSentence(path)
model = gensim.models.Word2Vec(sentences, sg=1, min_count=1)
model.save('./models/embedModel')
