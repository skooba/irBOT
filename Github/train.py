from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np

#do for processed data
training_data = np.load('training_data.npy')
training_values = [" ".join(training_data[i].values()) for i in range(training_data.size)]
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags = [str(i)]) for i,_d in enumerate(training_values)]
print(tagged_data)


#tain the data
