
#step: 1 import everything needed for the program

#used to read in files from the email directory
import os
import io 


#numpy and pandas are used to store data in memory
import numpy as np
from pandas import DataFrame

#Count vectorizer gets the amount of occurrances of words in the email
from sklearn.feature_extraction.text import CountVectorizer 

#sklearn also has contains Naive Bayes functionality already.
from sklearn.naive_bayes import MultinomialNB

from pylab import *

#STEP 2:  Read in the files from the file system

def readFiles(path):
    for root, directories, files in os.walk(path):
        for filename in files:
            path = os.path.join(root, filename)
            inBody = False
            lines = []
            with io.open(path, 'r', encoding='latin1') as f:         
                for line in f:
                    if inBody:
                        lines.append(line)
                    elif line == '\n':
                        inBody = True
           
            message = '\n'.join(lines)
            yield path, message

def getDataFrame(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification })
        index.append(filename)

    return DataFrame(rows, index=index)

cwd = os.getcwd()

#get data from the emails subdirectory in the project
data = DataFrame({'message': [], 'class': []})
data = data.append(getDataFrame(os.path.join(cwd, 'emails/spam'), 'spam'))
data = data.append(getDataFrame(os.path.join(cwd, 'emails/ham'), 'ham'))

#verify that data has been read in
print(data.head())

#split the data into trainig & testing data
#usually 80 20 is good split. The majority of the 
#data is used for training.
train_data = data.sample(frac =0.8)
test_data = data.sample(frac = 0.2)

#get a count of each instance of the words in all the emails
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(train_data['message'].values)

#create the Multinomial Naive Bayes classifier
#and train it on the training data.
classifier = MultinomialNB()
targets = train_data['class'].values
classifier.fit(counts, targets)

#predict the values of the test data. 
#Can see how effienct the model is based
#off wheter the email is spam or ham
example_counts = vectorizer.transform(test_data['message'].values)
predictions = classifier.predict(example_counts)
print(predictions)
