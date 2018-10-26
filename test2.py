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
def preprocessing(texto):
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
            xs = preprocessing(xs)
            
            # Positive
            if ys==1:
                for w in xs.split():
                    
                    if w in wordsP:
                        idx = wordsP.index(w)
                        countP[idx] = countP[idx] + 1
                    else:
                        wordsP.append(w)
                        countP.append(1)
                
            else:
                for w in xs.split():
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
def StochasticGradientDescent(x_train,y_train,features):
    import random
    from scipy.stats import logistic
    
    # Parameters
    eta       = 0.001
    err       = 1000
    errNorm   = 1000
    threshold = 0.00001
    
    n_samples  = len(x_train )
    n_features = len(features) ## ==================================================================
    
    w = np.zeros(n_features + 1)
    
    # Train Loop
    while (errNorm>threshold):
        exErr = err
        
        # Random selection
        n = round(random.uniform(0, n_samples-1))
        try:
            xs = np.array( [x_train[n][i] for i in features] + [1] ) ## ==================================================================
            
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

def applyModel(x,w,features):      
    
    y_pred = list()
    for xs in x:
        ys =  logistic.cdf( np.dot( np.array([xs[i] for i in features]  + [1]),w ) )  ## ==================================================================
        
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
    
    # Bad words list
    with open('negative-words.txt', 'r') as f:
        badWords = f.read().split('\n')[:-2]
    
    bad = 0
    for w in text.split():
        if w in badWords:
            bad = bad + 1
    
    return bad


""" x3: Does include "no"? """
def doesIncludeNo(text):
    
    nos = ['no','not']
    isthereNo = 0
    for w in text.split():
        if w in nos:
            isthereNo = 1
            break
    
    return isthereNo


""" x4: Does include Pronouns (1st and 2nd)? """
def doesIncludePronouns(text):
    
    pronouns = stopwords.words('english')[:17]
    isthere = 0
    for w in text.split():
        if w in pronouns:
            isthere = 1
            break
    
    return isthere


""" x5: Does include "!"? """
def doesIncludeExclamationMark(text):
    return int( '!' in text )


""" x6: log(count words) """
def logCountWords(text):
    return np.log( len(text.split()) )


""" x7: Does include More Positives? """
def morePositives(text):
    import pandas as pd
    df = pd.read_csv('interPosNeg.csv')
    positives = df.loc[:10,'Word'].values.tolist()
    
    df = pd.read_csv('excluPosNeg.csv')
    positives = positives + df.loc[:10,'Word'].values.tolist()
    
    isthere = 0
    for w in text.split():
        if w in positives:
            isthere = 1
            break
    
    return isthere


""" x8: Does include More Negatives? """
def moreNegatives(text):
    import pandas as pd
    df = pd.read_csv('interNegPos.csv')
    negatives = df.loc[:30,'Word'].values.tolist()
    
    df = pd.read_csv('excluNegPos.csv')
    negatives = negatives + df.loc[:2,'Word'].values.tolist()
    
    isthere = 0
    for w in text.split():
        if w in negatives:
            isthere = 1
            break
    
    return isthere


""" x9: How much include More Positives? """
def howMuchPositives(text,interPosNeg,excluPosNeg):
    positives = interPosNeg.loc[:30,'Word'].values.tolist()
    positives = positives + excluPosNeg.loc[:2,'Word'].values.tolist()
    
    count = 0
    for w in text.split():
        if w in positives:
            count = count + 1

    return count


""" x10: How much include More Negatives? """
def howMuchNegatives(text,interNegPos,excluNegPos):
    negatives = interNegPos.loc[:30,'Word'].values.tolist()
    negatives = negatives + excluNegPos.loc[:2,'Word'].values.tolist()
    
    count = 0
    for w in text.split():
        if w in negatives:
            count = count + 1
    
    return count


""" x11: How much include More Positives? (ratio) """
def howMuchPositivesRatio(text,interPosNeg,excluPosNeg):
    positives = interPosNeg.loc[:30,'Word'].values.tolist()
    positives = positives + excluPosNeg.loc[:2,'Word'].values.tolist()
    
    countPost = 0
    countWord = 0
    for w in text.split():
        countWord = countWord + 1
        if w in positives:
            countPost = countPost + 1
    
    return countPost/countWord


""" x12: How much include More Negatives? (ratio) """
def howMuchNegativesRatio(text,interNegPos,excluNegPos):
    negatives = interNegPos.loc[:30,'Word'].values.tolist()
    negatives = negatives + excluNegPos.loc[:2,'Word'].values.tolist()
    
    countNega = 0
    countWord = 0
    for w in text.split():
        countWord = countWord + 1
        if w in negatives:
            countNega = countNega + 1
    
    return countNega/countWord

##  ---------------------------------------------------------------------------    


"""
Data analysis
-------------
"""

"""
doc_stream = stream_docs(path='shuffled_movie_data.csv')
positive,negative = dataAnalysis(doc_stream)

# Save
positive.to_csv('positive.csv',index=False)
negative.to_csv('negative.csv',index=False)

"""
import pandas as pd

positive = pd.read_csv('positive.csv')
negative = pd.read_csv('negative.csv')

if 'index' in positive:
    positive.drop('index', axis=1, inplace=True)
if 'index' in negative:
    negative.drop('index', axis=1, inplace=True)

n = 2000
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
    
interPosNeg = interPosNeg.sort_values('Coefficient',ascending = False).reset_index(drop=True)
excluPosNeg = excluPosNeg.sort_values('Count',ascending = False).reset_index(drop=True)

interPosNeg.to_csv('interPosNeg.csv',index=False)
excluPosNeg.to_csv('excluPosNeg.csv',index=False)
        
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

interNegPos = interNegPos.sort_values('Coefficient',ascending = False).reset_index(drop=True)
excluNegPos = excluNegPos.sort_values('Count',ascending = False).reset_index(drop=True)

interNegPos.to_csv('interNegPos.csv',index=False)
excluNegPos.to_csv('excluNegPos.csv',index=False)



"""
Generate features
-----------------
"""
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
    features = [ [ positiveLexicon           (preprocessing(text)),
                   negativeLexicon           (preprocessing(text)),
                   doesIncludeNo             (preprocessing(text)), 
                   doesIncludePronouns       (preprocessing(text)),
                   doesIncludeExclamationMark(preprocessing(text)),
                   logCountWords             (preprocessing(text)),
                   morePositives             (preprocessing(text)),
                   moreNegatives             (preprocessing(text)),
                   howMuchPositives          (preprocessing(text),
                                                      interPosNeg,
                                                      excluPosNeg),
                   howMuchNegatives          (preprocessing(text),
                                                      interNegPos,
                                                      excluNegPos),
                   howMuchPositivesRatio     (preprocessing(text),
                                                      interPosNeg,
                                                      excluPosNeg),
                   howMuchNegativesRatio     (preprocessing(text),
                                                      interNegPos,
                                                      excluNegPos)] for text in x_raw ] 
    x = x + features
    
    # Update out
    y = y + y_raw
    
    # Bar
    pbar.update()
"""


"""
Train LR
--------
"""
from scipy.stats import logistic
from sklearn.model_selection import KFold

print('\nTrain Logistic Regression')
kf = KFold(n_splits=8)  
pbar = pyprind.ProgBar(8)

prediction = list()

features = [0,1,2,3,4,5,6,7,8,9,10,11]

for train, test in kf.split(x):
    
    # Select
    x_train = [ x[i] for i in train ]
    y_train = [ y[i] for i in train ]
    
    x_test = [ x[i] for i in test ]
    y_test = [ y[i] for i in test ]
    
    # Run train
    w = StochasticGradientDescent(x_train, y_train,features)
    
    # Run test
    y_pred = applyModel(x_test,w,features)
    
    prediction.append({'Real'      : y_test,
                       'Prediction': y_pred})
    
    # Bar
    pbar.update()

"""
Result Analysis
---------------
"""
from sklearn.metrics import roc_curve

print('\nResult Analysis')
accuracy   = list()
for p in prediction:
    fpr, tpr, thresholds =roc_curve(p['Real'], p['Prediction'])
    
    fpr_tpr = [ np.abs(a-b) for a,b in zip(fpr,tpr) ]
    threshold = thresholds[ fpr_tpr.index(max(fpr_tpr)) ]
    
    y_pred = [ int(score>threshold) for score in p['Prediction']]
    acc = 0
    for real,pred in zip(y_pred,p['Real']):
        acc = acc + int( real == pred )
    
    accuracy.append( acc*100/len(y_pred) )