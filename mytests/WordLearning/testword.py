from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
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
    print(len(POS))
    print(len(NEG))
    lexicon = []
    for line in POS:
        allwords = word_tokenize(line)
        lexicon += list(allwords)
    for line in NEG:
        allwords = word_tokenize(line)
        lexicon += list(allwords)
    #print(lexicon)
    for x in lexicon:
        lexicon = [lemmatizer.lemmatize(x)]
    wordcounts = Counter(lexicon)
    lexicon2 = []
    for word in wordcounts:
        if 1000 > wordcounts[word] > 50:
            lexicon2.append(word)
    print(len(lexicon2))
    return lexicon2


if __name__ == '__main__':
    readfiles()
    createlexicon()