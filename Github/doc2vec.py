
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def chunks(l, n):
    for i in range(0, len(l), n):
        if i == 0:
            vector = [l[i:i + n]]
        else:
            vector += [l[i:i + n]]
    return vector

#number of words to split document into
n = 10

dataframe = pd.read_csv('training_data_w_labels.csv')

training_data = np.load('training_data.npy')
training_values = [" ".join(training_data[i].values()) for i in range(training_data.size)]
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags = [str(i)]) for i,_d in enumerate(training_values)]

bow = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(tagged_data)):
    #split into 10 words
    bow[i] = chunks(tagged_data[i][0],n)

sections = ['VERTEX CARE AND HANDLING', 'GENERAL', 'NOTES ON SAFETY', 'TECHNICAL DATA', 'TRANSPORT/DELIVERY/STORAGE', 'INSTALLATION AND REMOVAL', 'COMMISIONING', 'UTILIZING THE HART COMMUNICATION', 'MAINTENANCE','TROUBLESHOOTING','DYNISCO CONTACT INFOMRATION','ACCESSORIES','APPROVALS/CERTIFICATES', 'OUTLINE DRAWINGS']

df = pd.DataFrame([])
for i in range(0,len(training_data)):
    dfnew = pd.DataFrame({'sections':sections[i],'words':bow[i]})
    #print(dfnew)
    df = df.append(dfnew, ignore_index = True)
print(df)

def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(v)
        #labeled.append(TaggedDocument(v.split(), [label]))
    return labeled

X_train, X_test, y_train, y_test = train_test_split(df.words, df.sections, random_state=0, train_size=0.7)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
#print(y_train)
#print(y_test)
all_data = X_train + X_test
#print(all_data)


model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(all_data)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def get_vectors(model, corpus_size, vectors_size, vectors_type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors

train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')


logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(train_vectors_dbow, y_train)
logreg = logreg.fit(train_vectors_dbow, y_train)
y_pred = logreg.predict(test_vectors_dbow)
print('accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred,target_names=sections))
