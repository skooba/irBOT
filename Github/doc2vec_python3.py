from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#PARAMETERS
#number to times to run through the algorthim
iterations = 10
#number of words to split document into
n = 4
#doc2vec function parameters
neg = 20
vs = 350
alp = 0.085
alpmin = 0.085
#ability to add additional stopwords to skip over
addtl_stopwords = []

stemmer = LancasterStemmer()
#function used to preprocess a column of a dataframe with string entries
#dataframe vairable is dataframe name and c is column name to preprocess
def preprocess(dataframe,c):
    #omit stopwords including user defined additional stop words
    stop = stopwords.words('english')+addtl_stopwords
    #split words in string, omit stopwords and stem remaining words then join
    #to a single string agian
    dataframe[c] = list(map(lambda x: ' '.join(
    [stemmer.stem(word) for word in x.split() if word not in stop]
    ), dataframe[c]))
    #return the modified dataframe
    return dataframe[c]

#function that breaks the PDF data into sections of n words
#l is the section to break into chunks
def chunks(l, n):
    #loops over all words in the section
    for i in range(0, len(l), n):
        #if this is the first chunk of the section
        if i == 0:
            #begin a list of chunks and make the first chunk the first n words
            #of l
            vector = [l[i:i + n]]
        #otherwise, if this is not the first iteration in the function
        else:
            #concatenate the chunk to create a list of chunks
            vector += [l[i:i + n]]
    #returns a list of chunks with each chunk a list of n words
    #number of chunks = roundup(number of words in section/n)
    return vector

#function that outputs a list of tagged entries from a dataframe column
#corpus is a column of string entries and label_type is the name of the labels
#tag type
def label_sentences(corpus, label_type):
    #initialize the list of tagged entries
    labeled = []
    #loop over the rows of each entry and keeps track of the entry and number
    for i, v in enumerate(corpus):
        #creates label for each entry
        label = label_type + '_' + str(i)
        #tag each entry and append to the end of tagged entries list
        labeled.append(TaggedDocument(v.split(), [label]))
    #return the list of tagged entries
    return labeled

#creates word leveling embedings using doc2vec method
def get_vectors(model, corpus_size, vectors_size, vectors_type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors

#initialize a vector for acccuracy which will replace a 0 for every iteration
accVector = [0]*iterations
#save the accuracies over iterations
for ac in range(10):
    #read labeled question file to classify into a dataframe
    #csv file has columns labeled 'question' and 'section'
    dataframe = pd.read_csv(
    'testing_data_w_labels.csv',  error_bad_lines=False
    )
    #lower case the question column lowercase and strip ' and ?
    dataframe['question'] = list(
    map(lambda x: x.lower().rstrip('? \''), dataframe['question'])
    )
    #load the training data output from TCmix.py
    #data loads as a list of words for each section
    training_data = np.load('training_data.npy')
    #change dictionary datastructure to list
    training_values = [" ".join(
    training_data[i].values()
    ) for i in range(training_data.size)]
    #tag the document - each section is tagged with a number that indicates
    #the order it appears in the PDF
    tagged_data = [TaggedDocument(
    words=word_tokenize(_d.lower()), tags = [str(i)]
    ) for i,_d in enumerate(training_values)]
    #initialize bag of words vector with one entry per section
    bow = [0]*14
    #loop over every section
    for i in range(len(tagged_data)):
        #break section into chunks
        bow[i] = chunks(tagged_data[i][0],n)
        #create list of sections from the training data
        sections = [list(
        training_data[i].values()
        )[0] for i in range(len(training_data))]
    #create a pandas dataframe structure for PDF training data
    df = pd.DataFrame([])
    #for each section in the PDF
    for i in range(0,len(training_data)):
        #create a temporary dataframe each row containing a new chunk of words
        #and the corresponding section
        dfnew = pd.DataFrame({'sections':sections[i],'words':bow[i]})
        #append the temporary dataframe to the PDF training dataframe
        df = df.append(dfnew, ignore_index = True)
    #maps each of chunk of words to a single string
    df['words'] = [' '.join(map(str,l)) for l in df['words']]
    #training dataset is the preprocessed dataframe chunks
    #note preprocessing data after breaking into chunks yeilds best results
    X_train = preprocess(df,'words')
    #labels for training are the data chunks
    y_train =df.sections
    #testing dataset is the proprocessed questions from the .csv file
    X_test = preprocess(dataframe,'question')
    #testing labels from the .csv file
    y_test = dataframe.section
    #tag/label X_test and X_train rows
    X_test = label_sentences(X_test, 'Test')
    X_train = label_sentences(X_train, 'Train')
    #combine training and testing data without labels
    all_data = X_train + X_test
    #initialize and train the model
    model_dbow = Doc2Vec(
    dm=1, vector_size=vs, negative=neg, min_count=2, alpha=alp, min_alpha=alpmin
    )
    #builds the vocabulary list from both training dataset and testing questions
    model_dbow.build_vocab([x for x in all_data])
    #train a neural network using gradient descent
    for epoch in range(30):
        #shuffle up the testing and training data and run trough neural net
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
        #decrease stepsize of learning rate at each iteration
        model_dbow.alpha -= 0.001
        #decrease learning rate is new minimum value
        model_dbow.min_alpha = model_dbow.alpha
    #create word level embedding vector for training dataset
    train_vectors_dbow = get_vectors(model_dbow, len(X_train), vs, 'Train')
    #create word level embedding vector for testing dataset
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), vs, 'Test')
    #one versus the rest logisitc regression classification_report
    #classes are weighted by the length of section
    logreg = LogisticRegression(n_jobs=1,class_weight='balanced', C=1e5) #max_iter = 1000
    #fit a regression line through the model using y_train
    logreg = logreg.fit(train_vectors_dbow, y_train)
    y_pred = logreg.predict(test_vectors_dbow)
    if ac == 0:
        predictions = pd.DataFrame(y_test)
    predictions[ac+1] = y_pred
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
    accVector[ac] = accuracy_score(y_pred, y_test)
print((sum(accVector)/10))
predictions.to_csv(str('10_predictions'),index = False)
