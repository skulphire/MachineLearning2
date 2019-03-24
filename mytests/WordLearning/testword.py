from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
import random

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def readfiles():
