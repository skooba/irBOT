import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from tabula import wrapper
import os
import csv
import re
from collections import OrderedDict
# word stemmer
stemmer = LancasterStemmer()



def preprocess_pdfs():
    pdf_folder = 'data/pdf'
    csv_folder = 'data/csv'

    base_command = 'java -jar tabula-1.0.2-jar-with-dependencies.jar -n -p all -a 100,0,730,612 -f TSV -o {} {}'

    for filename in os.listdir(pdf_folder):
        pdf_path = os.path.join(pdf_folder, filename)
        csv_path = os.path.join(csv_folder, filename.replace('.pdf', '.csv'))
        command = base_command.format(csv_path, pdf_path)
        os.system(command)
        read_file = []
        with open (csv_path, 'r') as infile:
            reader = csv.reader(infile)
            count = 0
            start_contents = []
            first_section = []
            dictionary = OrderedDict()
            for row in reader:
                count += 1
                if 'TABLE OF CONTENTS' in row[0].upper():
                    start_contents = count+1
                if count == start_contents:
                    first_section = re.sub('[^a-zA-Z]+','', row[0])
                if str(first_section).upper() == re.sub('[^a-zA-Z]+','', row[0]).upper():
                    end_contents = count - 1
                read_file.append('\n'.join(row))
        for section in read_file[start_contents-1:end_contents]:
            text = re.sub('[\W]+','', section).rstrip('0123456789').upper()
            count = end_contents-1
            for line in read_file[end_contents:]:
                count += 1
                if text == re.sub('[\W]','', line).upper():
                    dictionary[re.sub(r'\t|[0-9]\s*','',read_file[count])] = count
        section_starts = list(dictionary.values())
        count = 0
        for section in list(dictionary.keys()):
            if count < len(section_starts)-2:
                update = {section:re.sub(r'[^A-Z0-9 ]','',' '.join(read_file[section_starts[count]+1:section_starts[count+1]]).upper())}
            else:
                update = {section:re.sub(r'[^A-Z0-9 ]','',' '.join(read_file[section_starts[count]:]).upper())}
            dictionary.update(update)
            count += 1
        return dictionary



dictionary = preprocess_pdfs()
#print('Dict looks like', dictionary.shape)
keys = list(dictionary.keys())
values = list(dictionary.values())

training_data = []
for i in range(len(keys)):
    training_data.append({"class":keys[i], 'section':values[i]})

print ("%s sentences of training data" % len(training_data))

# capture unique stemmed words in the training corpus
corpus_words = {}
class_words = {}
addtl_stopwords = ['much','sensor']
stop = stopwords.words('english') + addtl_stopwords
# turn a list into a set (of unique items) and then a list again (this removes duplicates)
classes = list(set([a['class'] for a in training_data])) #Loops through the training data storing the classes and keeping the classes by using set(Eliminates duplicates)
#print('TD IS', training_data)

for c in classes:
    # prepare a list of words within each class
    class_words[c] = []

# loop through each sentence in our training data
for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['section']):
        # ignore a some things
        if word not in ["?", "'s", "0", "1", "2", "3", "4", "5", "6", "7", "8","9"]:
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])

# we now have each stemmed word and the number of occurances of the word in our training corpus (the word's commonality)
print ("Corpus words and counts: %s \n" % corpus_words)
# also we have all words in each class
print ("Class words: %s" % class_words)

# we can now calculate a score for a new sentence
sentence = "how much torque should I apply?"
sentence = sentence.lower()

# calculate a score for a given class taking into account word commonality
def calculate_class_score_commonality(sentence, class_name, show_details=True):
    score = 0.0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        if word not in stop:
        # check to see if the stem of the word is in any of our classes
            if stemmer.stem(word.lower()) in class_words[class_name]:
                count = class_words[class_name].count(stemmer.stem(word.lower()))
                #print('count is', word)
                #print('classwords is', class_words[class_name])
                # treat each word with relative weight
                score += (count / float(corpus_words[stemmer.stem(word.lower())]))
                print('stem is', stemmer.stem(word.lower()), class_name,  count)
                print('corpus whatever', corpus_words[stemmer.stem(word.lower())])

                if show_details:
                    print ("   match: %s (%s)" % (stemmer.stem(word.lower()), count / float(corpus_words[stemmer.stem(word.lower())])))
    return score

# now we can find the class with the highest score
#print('class words', class_words.keys())
for c in class_words.keys():
    #print('c is',c)
    print ("Class: %s  Score: %s \n" % (c, calculate_class_score_commonality(sentence, c)))
#print('classwords is', class_words['UTILIZING THE HART COMMUNICATIONS'])
# return the class with highest score for sentence
def classify(sentence):
    high_class = None
    high_score = 0
    # loop through our classes
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score_commonality(sentence, c, show_details=False)
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score

    return high_class, high_score




#print(classify("what is the output of my vertex sensor?"))

