import numpy as np

samples = {"The cat died six times", "I crashed the car"}
# we store the words with their corresponding index
# every word will occur exactly once (first one), and symbols and special characters will be counted as words
token_index = {word: index + 1 for index, word in enumerate(set(word for sample in samples for word in sample.split()))}
print(token_index)

# Set max_length to 6
max_length = 6
# Create a tensor of dimension 3 named results whose every elements are initialized to 0
results  = np.zeros(shape = (len(samples),max_length,max(token_index.values()) + 1)) 

# Now create a one-hot vector corresponding to the word
# iterate over enumerate(samples) enumerate object
for i, sample in enumerate(samples): 
  
# Convert enumerate object to list and iterate over resultant list 
  for j, considered_word in list(enumerate(sample.split())):
    
    # set the value of index variable equal to the value of considered_word in token_index
    index = token_index.get(considered_word)
    
    # In the previous zero tensor: results, set the value of elements with their positional index as [i, j, index] = 1.
    results[i, j, index] = 1. 


print(results)