# -*- coding: utf-8 -*-
"""
Created on Wed June 6 16:22:39 2023
Title: NLP project on SDG
This time we will implement LDA
@author: Nazlican
"""
#I will try to follow the 

import pandas as pd
import numpy as np
import string # Used to remove stopwords
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy import spatial


#Dataset
dataset = pd.read_csv('LDA/tokenized_data_without_punctuation_stopwords_lemma.csv')
print(dataset.shape)
dataset.head()

import gensim.downloader as api
import json
info = api.info()
for model_name, model_data in sorted(info['models'].items()):
    print(
        '%s (%d records): %s' % (
            model_name,
            model_data.get('num_records', -1),
            model_data['description'][:40] + '...',
        )
    )
    
glove_wiki_50_info = api.info('glove-wiki-gigaword-50')
print(json.dumps(glove_wiki_50_info, indent=4))
glove_vectors = api.load('glove-wiki-gigaword-50')

result = glove_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
print(f"Considering the word relationship (king, man) \n The model deems the pair (woman, ?) to be answered as ? = {result[0][0]} with score {result[0][1]} ")

print(glove_vectors.most_similar("emotional"))
print(glove_vectors.most_similar("rational"))

distance = glove_vectors.distance("coffee", "tea")
print(f"The distance between the words 'coffee' and 'tea' is: {distance}")

distance = glove_vectors.distance("coffee", "coffee")
print("The distance between a word and itself (using w2v) is always: ", distance)

vector = glove_vectors['computer']
print("Shape of the vector is: ", vector.shape)
print("With values: ", vector)

##Word Embedding
dataset.columns

word_vectors_all_sentences = []
dict_sentence_embeddings = {}  # dict with mapping of {id: sentence_embedding}
dict_paper_names = {}

for index, row in dataset.iterrows():
    document = row["text"]
    sentences = document.lower().split(".")
    
    for sentence in sentences:
        parsed_sentence = sentence.split()
        print(sentence)
        print(parsed_sentence)
        
        word_vectors = []
        
        for word in parsed_sentence:
            try:
                word_vector = glove_vectors[word]
                word_vectors.append(word_vector)
            except KeyError:
                print(f"Word '{word}' is not in the vocabulary.")
        
        if len(word_vectors) > 0:
            sentence_w2v_embedding = np.average(np.asarray(word_vectors), axis=0)
            print(sentence_w2v_embedding)
            word_vectors_all_sentences.append(sentence_w2v_embedding)
    
    sentence_embedding = np.average(np.asarray(word_vectors_all_sentences), axis=0)
    dict_sentence_embeddings[row["text_id"]] = sentence_embedding
    dict_paper_names[row["text_id"]] = row["text_id"]
    
    print("-------------------")

word_vectors_all_sentences = np.asarray(word_vectors_all_sentences)
print(word_vectors_all_sentences.shape)

print(dict_sentence_embeddings)
print(dict_paper_names)

import pickle

# Save the word_vectors_all_sentences dictionary
with open("word_vectors.pkl", "wb") as file:
    pickle.dump(word_vectors_all_sentences, file)
    
with open("word_vectors_per_text.pkl", "wb") as file:
    pickle.dump(dict_sentence_embeddings, file)
    
#Load it again
with open("word_vectors.pkl", "rb") as handle:
    dict_embeddings = pickle.load(handle)
    
with open("word_vectors_per_text.pkl", "rb") as handle:
    dict_embeddings_per_text = pickle.load(handle)
    

num_rows = len(dict_embeddings_per_text)  # Number of rows in the matrix
num_columns = len(next(iter(dict_embeddings_per_text.values())))  # Number of columns in the matrix

# Initialize an empty matrix
matrix = np.empty((num_rows, num_columns))

# Fill the matrix with values from the dictionary
for row_idx, values in enumerate(dict_embeddings_per_text.values()):
    matrix[row_idx] = values

# Print the matrix
print(matrix)
##For distribution creation it is better to use dict_embeddings dictionary    
dist_to_health = {}
for index, value in enumerate(dict_embeddings):
    key = index  # Use the index as the key
    dist_to_health[key] = spatial.distance.cosine(value, glove_vectors['health'])

      
    
most_similar_health = {}

for k, v in sorted(dist_to_health.items(), key=lambda item: item[1])[:5]:
    rows = dataset.loc[dataset['doi'] == k, 'text']
    if not rows.empty:
        most_similar_health[k] = rows.iloc[0]

most_similar_health