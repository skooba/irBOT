from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
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
# word stemmer
stemmer = LancasterStemmer()
def preprocess(dataframe,c):
    addtl_stopwords = []
    stop = stopwords.words('english')+addtl_stopwords
    dataframe[c] = map(lambda x: ' '.join([stemmer.stem(word) for word in x.split() if word not in stop]), dataframe[c])
    #dataframe[c] = map(lambda x: ' '.join([word for word in x.split() if word not in stop]), dataframe[c])
    return dataframe[c]

def chunks(l, n):
    for i in range(0, len(l), n):
        if i == 0:
            vector = [l[i:i + n]]
        else:
            vector += [l[i:i + n]]
    return vector
accVector = [0]*10
neg = 20
for ac in range(10):
    
	#number of words to split document into
	n = 4

	dataframe = pd.read_csv('training_data_w_labelsg.csv',  error_bad_lines=False)
	dataframe['question'] = map(lambda x: x.lower(), dataframe['question'])
	dataframe['question'] = map(lambda x: x.rstrip('? \''), dataframe['question'])
	#print(preprocess(dataframe,'question'))
	training_data = np.load('training_data.npy')
	training_values = [" ".join(training_data[i].values()) for i in range(training_data.size)]
	tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags = [str(i)]) for i,_d in enumerate(training_values)]
        #print(tagged_data)
	bow = [0]*14
	for i in range(len(tagged_data)):
	    #split into n words
	    bow[i] = chunks(tagged_data[i][0],n)
        #print(bow)
	#sections = dataframe.section.unique()
        sections = [list(training_data[i].values())[1] for i in range(len(training_data))]
	df = pd.DataFrame([])
	for i in range(0,len(training_data)):
	    dfnew = pd.DataFrame({'sections':sections[i],'words':bow[i]})
	    df = df.append(dfnew, ignore_index = True)
	df['words'] = [' '.join(map(str,l)) for l in df['words']]
	#print(df)

	def label_sentences(corpus, label_type):
	    labeled = []
	    for i, v in enumerate(corpus):
		label = label_type + '_' + str(i)
		labeled.append(TaggedDocument(v.split(), [label]))
	    return labeled

	#X_train = df.words
	X_train = preprocess(df,'words')
	y_train =df.sections
	#X_test = dataframe.question
	X_test = preprocess(dataframe,'question')
	y_test = dataframe.section 
	X_test = label_sentences(X_test, 'Test')
	X_train = label_sentences(X_train, 'Train')
        #print X_test
        #print X_train
        #print y_train.unique()
        
        #print y_train.unique()
        
	all_data = X_train + X_test
        vs = 350
        alp = 0.085
        alpmin = 0.085
	model_dbow = Doc2Vec(dm=1, vector_size=vs, negative=neg, min_count=2, alpha=alp, min_alpha=alpmin)
	model_dbow.build_vocab([x for x in tqdm(all_data)])

	for epoch in range(30):
	    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
	    model_dbow.alpha -= 0.001
	    model_dbow.min_alpha = model_dbow.alpha

	def get_vectors(model, corpus_size, vectors_size, vectors_type):
	    vectors = np.zeros((corpus_size, vectors_size))
	    for i in range(0, corpus_size):
		prefix = vectors_type + '_' + str(i)
		vectors[i] = model.docvecs[prefix]
	    return vectors

	train_vectors_dbow = get_vectors(model_dbow, len(X_train), vs, 'Train')
	test_vectors_dbow = get_vectors(model_dbow, len(X_test), vs, 'Test')


	logreg = LogisticRegression(n_jobs=1, C=1e5) #max_iter = 1000
	logreg.fit(train_vectors_dbow, y_train)
	logreg = logreg.fit(train_vectors_dbow, y_train)
	y_pred = logreg.predict(test_vectors_dbow)
	if ac == 0:
		predictions = pd.DataFrame(y_test)     
	predictions[ac+1] = y_pred 
	print(predictions)
	print('accuracy %s' % accuracy_score(y_pred, y_test))
	print(classification_report(y_test, y_pred))
        accVector[ac] = accuracy_score(y_pred, y_test)
print((sum(accVector)/10))
predictions.to_csv(str('10_predictions'),index = False)