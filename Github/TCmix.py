import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from tabula import wrapper
import pandas as pd
import numpy as np
import os
import csv
import re
from collections import OrderedDict

stemmer = LancasterStemmer()
#add words that are not to be considered (not including stopwords) as list of
#strings
addtl_stopwords = []
#*******************consider changing numbers to words i.ie 11 becomes 'one one'
def preprocess_pdfs():
    pdf_folder = 'data/pdf'
    csv_folder = 'data/csv'
    #call tabula from java jar file
    base_command = 'java -jar tabula-1.0.2-jar-with-dependencies.jar \
    -n -p all -a 100,0,730,612 -f TSV -o {} {}'
    for filename in os.listdir(pdf_folder):
        pdf_path = os.path.join(pdf_folder, filename)
        csv_path = os.path.join(csv_folder, filename.replace('.pdf', '.csv'))
        #read pdf to text
        command = base_command.format(csv_path, pdf_path)
        os.system(command)
        read_file = []
        with open (csv_path, 'r') as infile:
            reader = csv.reader(infile)
            count = 0
            start_contents = []
            first_section = []
            #if running in Python 2, a standard dictionary will loose order
            dictionary = OrderedDict()
            for row in reader:
                count += 1
                #PDF must have a Table of Contents
                if 'TABLE OF CONTENTS' in row[0].upper():
                    #line after Table of Contents is the beginning of TOC
                    start_contents = count+1
                if count == start_contents:
                    #first line of TOC is first section title
                    first_section = re.sub('[^a-zA-Z]+','', row[0])
                if str(first_section).upper() == re.sub(
                '[^a-zA-Z]+','',row[0]
                ).upper():
                    #last line of TOC is the line before the first section
                    #starts
                    end_contents = count - 1
                read_file.append('\n'.join(row))
                #between the first and last line of table of contents exists
                #section titles and page numbers, loop through these
        for section in read_file[start_contents-1:end_contents]:
            #get rid of page numbers to leave only section titles
            text = re.sub('[\W]+','', section).rstrip('0123456789').upper()
            count = end_contents-1
            #look through each line in the PDF starting after the TOC
            for line in read_file[end_contents:]:
                count += 1
                #if a line exists with only the section title then it is the
                #beginning of the section
                if text == re.sub('[\W]','', line).upper():
                    #dictionary takes from {section names: section line starts}
                    dictionary[
                    re.sub(r'\t|[0-9]\s*','',read_file[count])
                    ] = count
        #form a list of the line numbers where each section starts in the PDF
        section_starts = list(dictionary.values())
        count = 0
        #loop through the section names
        for section in list(dictionary.keys()):
            #if this isn't the final section in the pdf
            if count < len(section_starts)-2:
                #update the dictionary value to be the contents between the
                #section title of the current loop and the next section title
                update = {section:re.sub(r'[^A-Z0-9 ]','',' '.join(
                read_file[section_starts[count]+1:section_starts[count+1]]
                ).upper())}
            #if this is the final section of the PDF
            else:
                #update the final dictionary value to be the contents after the
                #last section title all the way to the end of the PDF
                update = {section:re.sub(
                r'[^A-Z0-9 ]','',' '.join(read_file[section_starts[count]:]
                ).upper())}
            dictionary.update(update)
            count += 1
    #save keys and values as a list
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    training_data = []
    for i in range(len(keys)):
        #create a data structure for all the contents in the PDF
        training_data.append({"class":keys[i], 'section':values[i]})
    #save the data structure so we can call this into the Doc2Vec algorthim
    np.save('training_data', training_data)
    # capture unique stemmed words in the training corpus
    corpus_words = {}
    class_words = {}
    # turn a list into a set (of unique items) and then a list again
    #(this removes duplicates)
    classes = list(set([a['class'] for a in training_data]))
    for c in classes:
        # prepare a list of words within each class
        class_words[c] = []
    # loop through each sentence in our training data
    # save out text documents for different steps of preprocessing
    for data in training_data:
        # tokenize each sentence into words
        for word in nltk.word_tokenize(data['section']):
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
            # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])
    # we now have each stemmed word and the number of occurances of the word in
    #our training corpus (AKA the word's commonality)
    # we also we have all words in each class
    return class_words, corpus_words

# calculate a score for a given class taking into account word commonality
def calculate_class_score_commonality(
class_words, corpus_words, sentence, class_name, show_details
):
    score = 0.0
    stop = stopwords.words('english') + addtl_stopwords
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        if word not in stop:
            # check to see if the stem of the word is in any of our classes
            if stemmer.stem(word.lower()) in class_words[class_name]:
                count = class_words[class_name].count(
                stemmer.stem(word.lower())
                )
                # treat each word with relative weight
                score += (count / float(corpus_words[
                stemmer.stem(word.lower())
                ]))
                if show_details:
                    print (
                    "   match: %s (%s)" % (stemmer.stem(word.lower()
                    ), count / float(corpus_words[stemmer.stem(word.lower())])))
    return score

# return the class with highest score for sentence
def classify(sentence,class_words,corpus_words):
    sentence = sentence.lower()
    sentence = sentence.rstrip('?')
    high_class = None
    high_score = 0
    # loop through our classes
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score_commonality(
        class_words, corpus_words, sentence, c, show_details=False
        )
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score
    return high_class, high_score

#call this function to output a single senctence
def get_target(sentence):
    print(classify(sentence))

#call this function to read a list of questions in csv format and save a new
#csv with questions and classified targets
def get_targets(csv):
    class_words, corpus_words = preprocess_pdfs()
    dataframe = pd.read_csv(csv)
    dataframe.columns = ['question','section']
    dataframe['classify'] = pd.Series('str', range(len(dataframe.index)))
    #dataframe['weight'] = pd.Series(np.nan, range(len(dataframe.index)))
    for i in range(len(dataframe.index)):
        question = str(dataframe.iloc[i]['question'])
        [clas, weight] = classify(question,class_words,corpus_words)
        dataframe.at[i,'classify'] = clas
        #dataframe.at[i, 'weight'] = weight
    dataframe.to_csv(str('training_data_predictions.csv'),index=False)
    return dataframe

predicted_dataframe = get_targets('training_data_w_labels.csv')
#training_data_w_labels is labeled data, sum over the number of predictions
#that match the labels
correct = sum(predicted_dataframe['classify']==predicted_dataframe['section'])
#find how many questions are in training_data_w_labels
[total,_] = predicted_dataframe.shape
#calculate the accuracy
accuracy = correct/total
print(accuracy)
