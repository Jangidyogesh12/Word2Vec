import nltk
import sqlite3
from nltk.tokenize import word_tokenize
from collections import deque
import numpy as np
import torch

'''
 
   This is a tokeniser function which reads the text file and tokenises it
 
'''
def tokeniser(path):
    # Open the file in read-write mode ('+r') with UTF-8 encoding
    with open(path, '+r', encoding='utf-8') as file:
        # Read the entire content of the file
        data = file.read()
        
        # Convert data to lowercase and replace 'old_text' with 'new_text'
        data = data.lower()
        new_data = data.replace('old_text', 'new_text')
        
        # Move the file pointer to the beginning, write new_data, and truncate the rest
        file.seek(0)
        file.write(new_data)
        file.truncate()
        
        # Split the modified data into tokens
        tokens = word_tokenize(new_data)
    
    # Return the list of tokens
    return tokens



'''

    This function terate through the tokenized corpus and for each word in the corpus (the target word), 

    create training examples using the words within the context window

'''
def generate_training_samples(path):
    tokens = tokeniser(path)
    context_window_size = 2

    context_words = deque([0] * 2 * context_window_size, maxlen=2 * context_window_size)  # Initialize a deque to store context words
    target = []
    training_samples = []
    word_to_id, _ = unique_word_vector(path)
    
    # Define a special token for out-of-vocabulary words
    unknown_token = len(word_to_id)  # Assign a unique ID for the OOV token
    
    for i, target_word in enumerate(tokens):
        # Clear the context_words deque for each new target word
        context_words = deque([0] * 2 * context_window_size, maxlen=2 * context_window_size)
        
        # Define the context window boundaries
        start = max(0, i - context_window_size)
        end = min(len(tokens), i + context_window_size + 1)
        
        for j in range(start, end):
            if j != i:
                word = tokens[j]
                word_id = word_to_id.get(word, unknown_token)  # Handle out-of-vocabulary words
                context_words.append(word_id)
        
        target_word_id = word_to_id.get(target_word, unknown_token)  # Handle out-of-vocabulary words
        target.append(target_word_id)
        
        # Create a training sample (context words and target word)
        training_samples.append(torch.tensor(list(context_words)))
    
    return training_samples, target, len(target)




''' 

   This function removes all the duplicates in the text document, arrange them in the sorted manner 

   and later creted a dictionary with word as key and its index as int value

   
'''
def unique_word_vector(path):
    tokenised_text_input = tokeniser(path)
    s = set()
    for i in tokenised_text_input:
        s.add(i)
    
    # sorting the data
    sorted_list = sorted(s)

    #Creating a dictionary
    word_to_id = {}
    id_to_word = {}
    for i, word in enumerate(sorted_list):
        word_to_id[word] = i+1
        id_to_word[i] = word
    
    return word_to_id, id_to_word

