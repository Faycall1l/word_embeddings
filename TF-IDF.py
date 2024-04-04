# You can calculate tfidf by directly using TfidfVectorizer provided by scikit-learn
# Sklearn adds 1 in numerator and denominator in the textbook idf formula. Why? To avoid division by zero.

import math
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def fit(dataset):
    unq_wrds = set()
    if isinstance(dataset, list):
        for row in dataset:
            for wrd in row.split(" "):
                if len(wrd) < 2:
                    continue
                unq_wrds.add(wrd)
        unq_wrds = sorted(list(unq_wrds))
        vocab = {wrd: idx for idx, wrd in enumerate(unq_wrds)}
    return vocab

def transform(dataset, vocab):
    num_docs = len(dataset)
    sparse_matrix = csr_matrix((num_docs, len(vocab)), dtype=float)
    idf_vals = {}
    for row_idx, row in enumerate(dataset):
        word_freq = Counter(row.split(' '))
        for word in row.split(' '):
            if word in vocab:
                tf = word_freq[word] / len(row.split(' '))
                idf = 1 + math.log((1 + num_docs) / (1 + sum(word in doc for doc in dataset)))
                tfidf = tf * idf
                sparse_matrix[row_idx, vocab[word]] = tfidf
                idf_vals[word] = idf
    output = normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False)
    return output, idf_vals

corpus = [
    "the first document",
    "the second document",
    "the third one",
    "the forth text",
]

vocab = fit(corpus)
tfidf_corpus, idfs_ = transform(corpus, vocab)

print("Custom idfs:\n", list(idfs_.values()))
print("TF-IDF values for the first document calculated by custom implementation:\n", tfidf_corpus[0].toarray())


#################################################
# This is th sk-learn implementation 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
skl_output = vectorizer.transform(corpus)

print('---------------------------------')
print('sk-learn values : \n')
print(vectorizer.idf_)  
print(skl_output[0])   


