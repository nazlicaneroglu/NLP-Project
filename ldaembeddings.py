import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Load dataset and BERT model
dataset = pd.read_csv('tokenized_data_without_punctuation_stopwords_lemma.csv')
dataset = dataset[:2000]
dataset = dataset['text'].astype(str).tolist()

logging.getLogger("transformers").setLevel(logging.ERROR)
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Get BERT embeddings
embeddings = []
for sentence in dataset:
    encoded_input = tokenizer.encode_plus(sentence, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    with torch.no_grad():
        model_output = model(input_ids)
    sentence_embeddings = model_output.last_hidden_state.squeeze(dim=0)
    embeddings.append(sentence_embeddings.numpy())

# Convert BERT embeddings to document-term matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset)

# Apply LDA on BERT embeddings
lda = LatentDirichletAllocation(n_components=17)  # Adjust the number of topics as needed 10
lda.fit(X)

# Get topic distributions for each document
topic_distributions = lda.transform(X)

# Print the topic distributions for the first few documents
for i, topic_dist in enumerate(topic_distributions[:5]):
    print(f"Document {i+1} - Topic Distribution: {topic_dist}")


# Plot the topic distributions
plt.figure(figsize=(10, 6))
plt.bar(range(len(topic_distributions[0])), topic_distributions[0])
plt.xlabel('Topic')
plt.ylabel('Probability')
plt.title('Topic Distribution - Document 1')
plt.xticks(range(len(topic_distributions[0])))
plt.show()


import numpy as np

# Get the most important words for each topic
feature_names = np.array(vectorizer.get_feature_names_out())
n_top_words = 10  # Number of top words to extract

for topic_idx, topic in enumerate(lda.components_):
    # Sort the indices of the words based on their importance in the topic
    top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
    top_words = feature_names[top_word_indices]
    
    # Print the topic and its most important words
    print(f"Topic {topic_idx+1}:")
    print(", ".join(top_words))
    print()
