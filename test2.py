#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:46:27 2018

@author: victor
"""
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import warnings
warnings.filterwarnings("ignore")

print(chr(27) + "[2J")

stop = stopwords.words('english')
porter = PorterStemmer()


"""
Fun. stream documents
---------------------
"""
def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label   
##  ---------------------------------------------------------------------------    


"""
Fun. tokenizer
--------------
"""
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text
##  ---------------------------------------------------------------------------    


"""
Get batch
---------
"""
def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y
##  ---------------------------------------------------------------------------    


"""
Stochastic Gradient Descent
---------------------------
"""
def StochasticGradientDescent(x_train,y_train,w):
    import random
    from scipy.stats import logistic
    
    # Parameters
    eta       = 0.001
    err       = 1000
    errNorm   = 1000
    threshold = 0.001
    
    n_samples  = x_train.shape[0]
    
    """
    # Train Loop
    while (errNorm>threshold):
        
        exErr = err
        err   =   0
        
        # Random selection
        n = round(random.uniform(0, n_samples))
        xs = x_train[n]
        ys = y_train[n]
        
        # Loss-function
        L = logistic.cdf(w[:-1]*xs + w[-1]) - ys
        
        # Gradient
        g = L*xs
        
        # Update
        w = w - eta*g
        
        # Prediction
        y_pred = w*xs
        
        # Error
        err = np.sum(np.abs(y_train - y_pred))
        
        # Update error
        errNorm = np.abs(exErr - err)/np.abs(err)
    """
    
    err = 0
    
    for n in range(n_samples):
        
        exErr = err
        xs = x_train[n]
        ys = y_train[n]
        
        xs = np.append(xs.toarray(),[[1]], axis=1)
        
        # Prediction
        if n>0:
            y_pred = np.dot(xs,w.T)
        
        # Loss-function
        L = logistic.cdf( np.dot(xs,w.T) ) - ys
        
        # Gradient
        g = L*xs
        
        # Update
        w = w - eta*g
        
        # Error
        if n>0:
            err = err + np.abs(ys - y_pred)
    
    # Update error
    errNorm = np.abs(exErr - err)/np.abs(err)
    
    print("-- Error: ",errNorm)
    return w
##  ---------------------------------------------------------------------------    


"""
Features
--------
"""
from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)


"""
Classifier
----------
"""
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='shuffled_movie_data.csv')


"""
Train
-----

import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    # Getting
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    
    # Feature
    X_train = vect.transform(X_train)
    
    # Run train
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
"""
# Getting
#print('a')
#x_train, y_train = get_minibatch(doc_stream, size=45000)
#print('b')

# Feature
#x_train = vect.transform(x_train)
#print('c')

import pyprind
pbar = pyprind.ProgBar(45)

# Train
w = np.zeros( 2**21 + 1 )
for _ in range(45):
    # Getting
    x_train, y_train = get_minibatch(doc_stream, size=1000)
    
    # Feature
    x_train = vect.transform(x_train)
    
    # Run train
    w = StochasticGradientDescent(x_train, y_train,w)
    pbar.update()


"""
Test
----
"""
from scipy.stats import logistic
x_test, y_test = get_minibatch(doc_stream, size=5000)
x_test = vect.transform(x_test)

acc = 0;
y = list()
for i in range(5000):
    y_pred = logistic.cdf( np.dot(np.append(x_test[i][:].toarray(),[[1]], axis=1),w.T) )
    y.append(y_pred[0][0])
    
    if( ( y_pred>0 and y_test==1 ) or 
        ( y_pred<0 and y_test==0 ) ):
        acc = acc + 1
    

#print('Accuracy: %.4f' % clf.score(X_test, y_test))

