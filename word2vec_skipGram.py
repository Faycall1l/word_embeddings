# The gist of it is that we can express words as vectors that encode their semantics in a meaningful way.
import re
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

## this modified text from wikipedia will be used 
text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''

#splitting the text
def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text)

def mapping(tokens):
    word_to_id = {}
    id_to_word = {}
    
    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token
    
    return word_to_id, id_to_word


def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i),
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))
    return np.asarray(X), np.asarray(y)

# the weight matrix effectively acts as a lookup table
# that moves one-hot encoded vectors to dense vectors in a different dimension
# then we pass the output into a softmax layer that will convert it into probability vectors whose elements sum up to one.
# we use Shannon entropy H(p,q)=−p⋅log(q)
# p is one-hot-encoded vector, and q is the prediction

# The overall implementaion :
## A1=XW1
## A2=A1W2
## Z=softmax(A2)

def init_network(vocab_size, n_embedding):
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model

def forward(model, X, return_cache = True): # if we simply want the final prediction vectors only, not the cache, we set return_cache to False
    cache = {}

    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])

    if not return_cache:
        return cache["z"]
    return cache

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

def entropy(z, y):
    return(-np.sum(np.log(z) * y))

## ∂L/∂A2=Z−y given our previous Z
def backward(model, X, y, alpha):
    cache = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1

    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2

    return entropy(cache["z"], y)

## getting the embedding of a specific word
def get_embedding(model, word):
    try:
        idx = word_to_id[word]
    except KeyError:
        print("`word` not in corpus")
    one_hot = one_hot_encode(idx, len(word_to_id))
    return forward(model, one_hot)["a1"]







tokens = tokenize(text)
# lookup tables for the text
word_to_id, id_to_word = mapping(tokens)
# generate some training data with a window size of two
X, y = generate_training_data(tokens, word_to_id, 2)

print(X.shape)
print(y.shape)

model = init_network(len(word_to_id), 10)

print((X @ model["w1"]).shape)
print((X @ model["w1"] @ model["w2"]).shape) # back to the original dimentionality

# Testing 
n_iter = 50
learning_rate = 0.05

history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]

plt.plot(range(len(history)), history, color="skyblue")
plt.show()

## sanity check we can perform is to see which token our model predicts given the word “learning.”
# If the model was trained properly, the most likely word should understandably be “machine.”
get_embedding(model, "learning")
learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
result = forward(model, [learning], return_cache=False)[0]

for word in (id_to_word[id] for id in np.argsort(result)[::-1]):
    print(word)
