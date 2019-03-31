from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
import numpy as np
import random
import pickle

lemmatizer = WordNetLemmatizer()

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

def classify(lexicon):
    featureset = []
    for line in NEG:
        currentwords = word_tokenize(line.lower())
        currentwords = [lemmatizer.lemmatize(x) for x in currentwords]
        features = np.zeros(len(lexicon))
        for word in currentwords:
            if word.lower() in lexicon:
                index = lexicon.index(word.lower())
                features[index] += 1
        featureset.append([features,[0,1]])

    for line in POS:
        currentwords = word_tokenize(line.lower())
        currentwords = [lemmatizer.lemmatize(x) for x in currentwords]
        features = np.zeros(len(lexicon))
        for word in currentwords:
            if word.lower() in lexicon:
                index = lexicon.index(word.lower())
                features[index] += 1
        featureset.append([features.tolist,[1,0]])
    return featureset

if __name__ == '__main__':
    readfiles()
    lexicon = createlexicon()
    set = classify(lexicon)
    print(set[0])