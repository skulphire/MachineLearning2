from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
import numpy as np
import random
import pickle

lemmatizer = WordNetLemmatizer()
#hm_lines = 10000000

POS = []
NEG = []

def readfiles():
    with open('pos-neg-sentdex/pos.txt','r') as f:
        contents = f.readlines()
        for line in contents:
            POS.append(line)
    with open('pos-neg-sentdex/neg.txt','r') as f:
        contents = f.readlines()
        for line in contents:
            NEG.append(line)
    #print(len(POS))
    #print(len(NEG))

def createlexicon():
    lexicon = []
    for line in POS:
        allwords = word_tokenize(line.lower())
        lexicon += list(allwords)
    for line in NEG:
        allwords = word_tokenize(line.lower())
        lexicon += list(allwords)
    #print(lexicon)
    lexicon = [lemmatizer.lemmatize(x)for x in lexicon]
    wordcounts = Counter(lexicon)
    lexicon2 = []
    for word in wordcounts:
        if 1000 > wordcounts[word] > 25:
            lexicon2.append(word)
    print(len(lexicon2))
    return lexicon2

def classify(lexicon, classification):
    data = NEG
    if classification[0] == 1:
        data = POS
    featureset = []

    for line in data:
        currentwords = word_tokenize(line.lower())
        currentwords = [lemmatizer.lemmatize(x)for x in currentwords]
        features = np.zeros(len(lexicon))
        for word in currentwords:
            if word.lower() in lexicon:
                index = lexicon.index(word.lower())
                features[index] += 1
        features = list(features)
        featureset.append([features,classification])
        #print(featureset)
    return featureset

def createsets(lexicon, test_size=0.1):
    features = []
    features += classify(lexicon,[1,0])
    features += classify(lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size*len(features))
    #trainX = list(features[:,0][:-testing_size])
    #trainY = list(features[:,1][:-testing_size])
    #testX = list(features[:,0][-testing_size:])
    #testY = list(features[:,1][-testing_size:])
    trainSet = list(features[:-testing_size])
    testSet = list(features[-testing_size:])

    return trainSet,testSet

def createdataset():
    readfiles()
    lexicon = createlexicon()
    trainSet,testSet = createsets(lexicon)
    #print(trainX[0])
    with open ('set.pickle','wb') as f:
        pickle.dump([trainSet,testSet],f)
    

if __name__ == '__main__':
    readfiles()
    lexicon = createlexicon()
    trainSet,testSet = createsets(lexicon)
    print(trainSet[0])
    #print(len(trainX[0]))

    with open ('set.pickle','wb') as f:
       pickle.dump([trainSet,testSet],f)