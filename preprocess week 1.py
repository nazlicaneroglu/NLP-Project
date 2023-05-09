# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:22:39 2023
Title: NLP project on SDG
@author: Nazlican
"""
#install packages
#installed nltk and other packages via pip by using console'
import nltk
import pandas as pd
dataset = pd.read_csv('columns_arranged.csv')
print(dataset.shape)
dataset.head()

#We will use package NLTK to do tokenization first
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize

example = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""
sent_tokenize(example) #This splits documents into sentence
word_tokenize(example) #This splits documents into words, and punctuation. 
#We thought using punkt package would be a better choice because it also saves punctuation seperately. 
#So we will be able to delete them easily whenever we want.

dataset['sentence_split'] = dataset['text'].apply(sent_tokenize) 
dataset['word_split'] = dataset['text'].apply(word_tokenize) 
dataset.to_csv('tokenized_data.csv')

#Stop words
nltk.download() #download all the popular packages
from nltk.corpus import stopwords
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
# Remove stop words from the "word_split" column
dataset['word_split'] = dataset['word_split'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
dataset.to_csv('tokenized_data_without_stopwords.csv')

#Punctuation
import string
punctuations = set(string.punctuation)
dataset['word_split'] = dataset['word_split'].apply(lambda x: [word for word in x if word not in punctuations])

# When I check the word split column, I saw that there are still elements like ▪, “, ’, ‘ or ©. Update the punctuation set and do again.

punctuations = set(string.punctuation + '▪“”’‘©')
dataset['word_split'] = dataset['word_split'].apply(lambda x: [word for word in x if word not in punctuations])
dataset.to_csv('tokenized_data_without_punctuation_stopwords.csv')

#It is also good to have one more column with lemmatization so that we can compare the quality of both choices. I will again use NLTK package.
#We could simply use function lemmatizer.lemmatize() to do that but this function assumes that the word is always a noun. That's why, I wanted to try
# lemmatizer.lemmatize(word, pos) which also takes the type of word into account.

#Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for word, tag in nltk.pos_tag(word_tokenize(text)):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_token = lemmatizer.lemmatize(word, pos)
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens

dataset['lemmatized'] = dataset['word_split'].apply(lambda x: [word.lower() for word in lemmatize_text(' '.join(x))])
dataset.to_csv('tokenized_data_without_punctuation_stopwords_lemma.csv')

#Remove numbers: I am not sure whether I should do that because some years might have importance. Ask this one in the lecture

