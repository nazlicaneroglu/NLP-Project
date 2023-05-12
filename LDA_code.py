# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:22:39 2023
Title: NLP project on SDG
This time we will implement LDA
@author: Nazlican
"""
import pandas as pd
dataset = pd.read_csv('LDA/tokenized_data_without_punctuation_stopwords_lemma.csv')
print(dataset.shape)
dataset.head()


