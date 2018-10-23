#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:55:33 2018

@author: victor
"""

import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords


"""
The IMDb Movie Review Dataset
-----------------------------
"""
# if you want to download the original file:
#df = pd.read_csv('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/50k_imdb_movie_reviews.csv')
# otherwise load local file
df = pd.read_csv('shuffled_movie_data.csv')
df.tail()


"""
Preprocessing Text Data
-----------------------
Now, let us define a simple tokenizer that splits the text into individual word 
tokens. Furthermore, we will use some simple regular expression to remove HTML 
markup and all non-letter characters but "emoticons," convert the text to lower 
case, remove stopwords, and apply the Porter stemming algorithm to convert the 
words into their root form.
"""
stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text

tokenizer('This :) is a <a> test! :-)</br>')

    
"""
Learning (SciKit)
-----------------
First, we define a generator that returns the document body and the 
corresponding class label:
"""
def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label    

"""
To conform that the stream_docs function fetches the documents as intended, 
let us execute the following code snippet before we implement the get_minibatch 
function:
"""

next(stream_docs(path='shuffled_movie_data.csv'))

"""
After we confirmed that our stream_docs functions works, we will now implement 
a get_minibatch function to fetch a specified number (size) of documents:
"""

def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y

"""
Next, we will make use of the "hashing trick" through scikit-learns 
HashingVectorizer to create a bag-of-words model of our documents. Details of 
the bag-of-words model for document classification can be found at Naive Bayes 
and Text Classification I - Introduction and Theory.
"""    

from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

# Excercise 1: define new features according to 
#              https://web.stanford.edu/~jurafsky/slp3/5.pdf


"""
Using the SGDClassifier from scikit-learn, we will can instanciate a logistic 
regression classifier that learns from the documents incrementally using 
stochastic gradient descent.
"""

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='shuffled_movie_data.csv')

# Excercise 2: implement a Logistic Regression classifier, using regularization, 
# according to https://web.stanford.edu/~jurafsky/slp3/5.pdf

import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    #pbar.update()

"""
Depending on your machine, it will take about 2-3 minutes to stream the 
documents and learn the weights for the logistic regression model to classify 
"new" movie reviews. Executing the preceding code, we used the first 45,000 
movie reviews to train the classifier, which means that we have 5,000 reviews 
left for testing:
"""

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

"""
I think that the predictive performance, an accuracy of ~87%, is quite 
"reasonable" given that we "only" used the default parameters and didn't do any 
hyperparameter optimization.

After we estimated the model perfomance, let us use those last 5,000 test 
samples to update our model.
"""

clf = clf.partial_fit(X_test, y_test)

"""
Model Persistence
-----------------
In the previous section, we successfully trained a model to predict the 
sentiment of a movie review. Unfortunately, if we'd close this IPython notebook 
at this point, we'd have to go through the whole learning process again and 
again if we'd want to make a prediction on "new data."

So, to reuse this model, we could use the pickle module to "serialize a Python 
object structure". Or even better, we could use the joblib library, which 
handles large NumPy arrays more efficiently.

To install: conda install -c anaconda joblib
"""

import joblib
import os
if not os.path.exists('./pkl_objects'):
    os.mkdir('./pkl_objects')
    
joblib.dump(vect, './vectorizer.pkl')
joblib.dump(clf, './clf.pkl')


"""
Using the code above, we "pickled" the HashingVectorizer and the SGDClassifier 
so that we can re-use those objects later. However, pickle and joblib have a 
known issue with pickling objects or functions from a __main__ block and we'd 
get an AttributeError: Can't get attribute [x] on <module '__main__'> if we'd 
unpickle it later. Thus, to pickle the tokenizer function, we can write it to a
file and import it to get the namespace "right".
"""

#writefile tokenizer.py
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text

from tokenizer import tokenizer
joblib.dump(tokenizer, './tokenizer.pkl')

"""
Now, let us restart this IPython notebook and check if the we can load our 
serialized objects:
"""

#import joblib
#tokenizer = joblib.load('tokenizer.pkl')
#vect = joblib.load('./vectorizer.pkl')
#clf = joblib.load('./clf.pkl')

"""
After loading the tokenizer, HashingVectorizer, and the tranined logistic 
regression model, we can use it to make predictions on new data, which can be 
useful, for example, if we'd want to embed our classifier into a web 
application -- a topic for another IPython notebook.
"""

#example = ['I did not like this movie']
#X = vect.transform(example)
#clf.predict(X)

#example = ['I loved this movie']
#X = vect.transform(example)
#clf.predict(X)
