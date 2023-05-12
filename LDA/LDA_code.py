# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:22:39 2023
Title: NLP project on SDG
This time we will implement LDA
@author: Nazlican
"""
#Packages:
import pandas as pd
import numpy as np
import nltk # Used to 
import gensim.corpora as corpora
import matplotlib.pyplot as plt

from pprint import pprint
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from tqdm import tqdm
from gensim import corpora

#Dataset
dataset = pd.read_csv('LDA/tokenized_data_without_punctuation_stopwords_lemma.csv')
print(dataset.shape)
dataset.head()

token_lists = dataset['lemmatized'].tolist()
token_lists = [d.split() for d in token_lists]

dictionary = corpora.Dictionary(token_lists)

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)  
_ = dictionary[0] # need this line to force-load the data in the kernel
id2word = dictionary.id2token

corpus = [dictionary.doc2bow(doc) for doc in token_lists]

lda_model = LdaModel(corpus, id2word=id2word,num_topics=20, decay = 0.6, minimum_probability=0.001)
for topic in lda_model.print_topics():
    print(topic)
#it seems that we have reasonable words in each topic
n_topics = 15 
n_keywords = 15 
topic_keywords,topic_keyvalues = [],[]
for (topic, values) in lda_model.print_topics(n_topics, n_keywords):
    temp_list_keywords,  temp_list_keyvalues = [],[]
    for value in (str(values).split()):
        if "*" in value:
            value = value.split("\"")
            value_keyword = float(value[0][:-1])
            keyword = value[1]
            temp_list_keywords.append(keyword)
            temp_list_keyvalues.append(value_keyword)
    topic_keywords.append(temp_list_keywords)
    topic_keyvalues.append(temp_list_keyvalues)
    
    
# save the output of the model in a dataframe. You can also store the list of "topic_keyvalues" to save the estimated values.
df_to_save = pd.DataFrame(topic_keywords, index = [f"Topic {topic_number+1}" for topic_number in range(n_topics)],
                  columns = [f"Keyword #{index+1}" for index, _ in enumerate(topic_keywords[0])])

df_to_save.to_csv("results_LDA_model.csv")

#Visualization
# I had an error for pyLDAvis package, i tried the following but couldnot solve
#pip install pyLDAvis
from IPython.display import display
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
pyLDAvis.enable_notebook()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
display(pyLDAvis.display(vis_data))
pyLDAvis.save_html(vis_data, 'lda.html')
#last line of code saves a html files in the directory where i can interactively examine the model. very useful!

#The following code is taken from Week_3_Latent_Dirichlet_Allocation.ipynb that is provided in the canvas.
topics = lda_model[corpus[2]] # 0 denotes the document
plt.bar(list(zip(*topics))[0], list(zip(*topics))[1])

topic_dist = pd.DataFrame(columns = ['topics', 'topic_list', 'top_topic','top_topic_prob'])
topic_dist.topics = lda_model[corpus]
topic_dist.topic_list = topic_dist.topics.apply(lambda y: [x[0] for x in y])
topic_dist.top_topic = topic_dist.topics.apply(lambda y: max(y, key = lambda x: x[1])[0])
topic_dist.top_topic_prob = topic_dist.topics.apply(lambda y: max(y, key = lambda x: x[1])[1])
print(topic_dist.head())

topic_dist.top_topic_prob.plot.hist(grid = True, bins = 20, rwidth = 0.9, color = '#607c8e')
plt.title('Histogram of probabilies of top topic')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

topic_dist.top_topic.plot.hist(grid = True, bins = 20, rwidth = 0.9, color = '#607c8e')
plt.title('Histogram of top topics')
plt.xlabel('Topic')
plt.ylabel('Frequency')
plt.xticks(range(20))
plt.grid(axis='y', alpha=0.75)








