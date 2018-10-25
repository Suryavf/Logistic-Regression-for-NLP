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

# Return a lower case proccesed text
def processtext(texto):
    import re
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\n)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    texto = REPLACE_NO_SPACE.sub('', texto.lower())
    texto = REPLACE_WITH_SPACE.sub(' ', texto)
    return texto

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
Preprocessing
-------------
"""
def dataAnalysis(doc):
    
    import pandas as pd
    
    wordsP = list()
    countP = list()
    
    wordsN = list()
    countN = list()
    
    import pyprind
    pbar = pyprind.ProgBar(100)
    print('Data Analysis -----------------------\nRead text')
    for _ in range(100):
        # Getting
        x, y = get_minibatch(doc, size=500)
        
        for xs,ys in zip(x,y):
            
            # Positive
            if ys==1:
                for w in xs[1:-1].split():
                    if w in wordsP:
                        idx = wordsP.index(w)
                        countP[idx] = countP[idx] + 1
                    else:
                        wordsP.append(w)
                        countP.append(1)
                
            else:
                for w in xs[1:-1].split():
                    if w in wordsN:
                        idx = wordsN.index(w)
                        countN[idx] = countN[idx] + 1
                    else:
                        wordsN.append(w)
                        countN.append(1)
                
                
        # Bar
        pbar.update()
        
        
    print('\nSorting')
    positive = sorted(zip(countP,wordsP),reverse=True)
    negative = sorted(zip(countN,wordsN),reverse=True)
    
    positive = pd.DataFrame({'Word' : [w for _,w in positive],
                             'Count': [c for c,_ in positive]})
    negative = pd.DataFrame({'Word' : [w for _,w in negative],
                             'Count': [c for c,_ in negative]})
    
    return positive,negative
##  ---------------------------------------------------------------------------    


"""
Stochastic Gradient Descent
---------------------------
"""
def StochasticGradientDescent(x_train,y_train):
    import random
    from scipy.stats import logistic
    
    # Parameters
    eta       = 0.001
    err       = 1000
    errNorm   = 1000
    threshold = 0.00001
    
    n_samples  = len(x_train   )
    n_features = len(x_train[0])
    
    w = np.zeros(n_features + 1)
    
    # Train Loop
    while (errNorm>threshold):
        exErr = err
        
        # Random selection
        n = round(random.uniform(0, n_samples-1))
        try:
            xs = np.array( x_train[n] + [1] )
        except:
            print('n: ',n)
            print('n_samples: ',n_samples)
            
        ys = y_train[n]
        
        # Hypotesis
        h = logistic.cdf( np.dot(xs,w) ) 
        
        # Gradient
        g = (h - ys)*xs
        
        # Update
        w = w - eta*g
        
        # Prediction
        y_pred = w*xs
        
        # Error
        err = np.sum(np.abs(ys - y_pred))
        
        # Update error
        errNorm = np.abs(exErr - err)/np.abs(err)
        
    return w

def applyModel(x,w):      
    
    y_pred = list()
    for xs in x:
        ys =  logistic.cdf( np.dot( np.array(xs + [1]),w ) ) 
        ys = int( ys > 0.5 )
        
        y_pred.append(ys)
    
    return y_pred
##  ---------------------------------------------------------------------------    


"""
Features
--------
 - x1: Positive lexicon 
 - x2: Negative lexicon
 - x3: Exist "No"?
 - x4: Exist pronouns?
 - x5: Exist "!"?
 - x6: log(count words)
"""
from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)


""" x1: Positive lexicon """
def positiveLexicon(text):
    
    # Good words list
    with open('positive-words.txt', 'r') as f:
        goodWords = f.read().split('\n')[:-2]
    
    good = 0
    for w in text.split():
        if w in goodWords:
            good = good + 1
    
    return good


""" x2: Negative lexicon """
def negativeLexicon(text):
    
    # Good words list
    with open('negative-words.txt', 'r') as f:
        badWords = f.read().split('\n')[:-2]
    
    bad = 0
    for w in text.split():
        if w in badWords:
            bad = bad + 1
    
    return bad


""" x3: Does include "no"? """
def doesIncludeNo(text):
    
    nos = ['No','no','Not','not']
    isthereNo = 0
    for w in text[1:-1].split():
        if w in nos:
            isthereNo = 1
            break
    
    return isthereNo


""" x4: Does include Pronouns (1st and 2nd)? """
def doesIncludePronouns(text):
    
    pronouns = stopwords.words('english')[17:]
    isthere = 0
    for w in text[1:-1].split():
        if w in pronouns:
            isthere = 1
            break
    
    return isthere


""" x5: Does include "!"? """
def doesIncludeExclamationMark(text):
    return int( '!' in text[1:-1] )


""" x6: log(count words) """
def logCountWords(text):
    return np.log( len(text[1:-1].split()) )
##  ---------------------------------------------------------------------------    


"""
Generate features
-----------------
"""

import pyprind
pbar = pyprind.ProgBar(50)
doc_stream = stream_docs(path='shuffled_movie_data.csv')

print('\nGenerate Features')
x = list()
y = list()
for _ in range(50):
    # Getting
    x_raw, y_raw = get_minibatch(doc_stream, size=1000)
    
    # Update features
    features = [ [ positiveLexicon           (processtext(text)),
                   negativeLexicon           (processtext(text)),
                   doesIncludeNo             (processtext(text)), 
                   doesIncludePronouns       (processtext(text)),
                   doesIncludeExclamationMark(processtext(text)),
                   logCountWords             (processtext(text))] for text in x_raw ] 
    x = x + features
    
    # Update out
    y = y + y_raw
    
    # Bar
    pbar.update()


"""
Data analysis
-------------
"""

doc_stream = stream_docs(path='shuffled_movie_data.csv')
positive,negative = dataAnalysis(doc_stream)

# Save
positive.to_csv('positive.csv',index=False)
negative.to_csv('negative.csv',index=False)

import pandas as pd
n = 1000
select_positive = positive.loc[:n,:]
select_negative = negative.loc[:n,:]

select_words_positive = select_positive['Word'].values.tolist()
select_words_negative = select_negative['Word'].values.tolist()


# -----------------------------------------------------------------------------
PosNeg  = list()
coefPos = list()

Pos_Neg = list()
countPN = list()
for ip in range( len(select_positive) ):
    
    w = select_positive.loc[ip,'Word']
    
    if w in select_words_negative:
        count_pos = select_positive.loc[ip,'Count']
        count_neg = select_negative.loc[select_negative['Word'] == w,'Count'].values[0]
    
        PosNeg.append(w)
        coefPos.append( count_pos/count_neg )
    else:
        Pos_Neg.append(w)
        countPN.append( select_positive.loc[ip,'Count'] )
        
interPosNeg = pd.DataFrame({'Word' : PosNeg,'Coefficient': coefPos})
excluPosNeg = pd.DataFrame({'Word' : Pos_Neg,'Count': countPN})
    
interPosNeg = interPosNeg.sort_values('Coefficient',ascending = False).reset_index()
excluPosNeg = excluPosNeg.sort_values('Count',ascending = False).reset_index()

        
# -----------------------------------------------------------------------------
NegPos  = list()
coefNeg = list()

Neg_Pos = list()
countNP = list()
for ip in range( len(select_negative) ):
    
    w = select_negative.loc[ip,'Word']
    
    if w in select_words_positive:
        count_pos = select_negative.loc[ip,'Count']
        count_neg = select_positive.loc[select_positive['Word'] == w,'Count'].values[0]
    
        NegPos.append(w)
        coefNeg.append( count_pos/count_neg )
    else:
        Neg_Pos.append(w)
        countNP.append(select_negative.loc[ip,'Count'])

interNegPos = pd.DataFrame({'Word' : NegPos,'Coefficient': coefNeg})
excluNegPos = pd.DataFrame({'Word' : Neg_Pos,'Count': countNP})

interNegPos = interNegPos.sort_values('Coefficient',ascending = False).reset_index()
excluNegPos = excluNegPos.sort_values('Count',ascending = False).reset_index()




"""
Train LR
--------
"""
from scipy.stats import logistic
from sklearn.model_selection import KFold

print('\nTrain Logistic Regression')
kf = KFold(n_splits=8)  
pbar = pyprind.ProgBar(8)

accuracy = list()
for train, test in kf.split(x):
    
    # Select
    x_train = [ x[i] for i in train ]
    y_train = [ y[i] for i in train ]
    
    x_test = [ x[i] for i in test ]
    y_test = [ y[i] for i in test ]
    
    # Run train
    w = StochasticGradientDescent(x_train, y_train)
    
    # Run test
    y_pred = applyModel(x_test,w)
    
    # Accuracy
    acc = 0
    for real,pred in zip(y_pred,y_test):
        acc = acc + int( real == pred )
    
    accuracy.append( acc*100/len(test) )
    
    # Bar
    pbar.update()

