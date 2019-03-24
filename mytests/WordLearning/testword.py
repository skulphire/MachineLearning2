from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
import random

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

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
    print(len(POS))
    print(len(NEG))

if __name__ == '__main__':
    readfiles()