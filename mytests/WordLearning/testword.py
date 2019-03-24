from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
import numpy as np
import random

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
        features = torch.zeros(len(lexicon))
        for word in currentwords:
            if word.lower() in lexicon:
                index = lexicon.index(word.lower())
                features[index] += 1
        featureset.append([features,classification])
        print(featureset)
    return featureset

def createsets(lexicon, test_size=0.1):
    features = []
    features += classify(lexicon,[1,0])
    features += classify(lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size*len(features))
    trainX = torch.tensor(torch.from_numpy(features[:,0][:-testing_size]))
    trainY = torch.tensor(torch.from_numpy(features[:,1][:-testing_size]))
    testX = torch.tensor(torch.from_numpy(features[:,0][-testing_size:]))
    testY = torch.tensor(torch.from_numpy(features[:,1][-testing_size:]))

    return trainX,trainY,testX,testY

if __name__ == '__main__':
    readfiles()
    lexicon = createlexicon()
    trainX,trainY,testX,testY = createsets(lexicon)
    print(trainX[0])