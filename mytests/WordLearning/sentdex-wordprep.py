from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import random
import pickle
import torch

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos,neg):
    lexicon = []
    with open(pos,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(neg,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample,lexicon,classification):

    featureset = []

    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features,classification])
            print(featureset)

    return featureset


def create_sets(pos,neg,test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos,lexicon,[1,0])
    features += sample_handling(neg,lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)
    #print(features[0])
    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][:-testing_size])
    print(train_x[0])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x,train_y,test_x,test_y

#if __name__ == '__main__':
 #    train_x,train_y,test_x,test_y = create_sets('/MachineLearning2/pos-neg-sentdex/pos.txt','/MachineLearning2/pos-neg-sentdex/neg.txt')

     #with open ('set.pickle','wb') as f:
     #    pickle.dump([train_x,train_y,test_x,test_y],f)