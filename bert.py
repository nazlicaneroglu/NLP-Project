import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import string # Used to remove stopwords
import nltk 
#nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy import spatial
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel


#Dataset
dataset = pd.read_csv('tokenized_data_without_punctuation_stopwords_lemma.csv')


# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(dataset)
tokenizer = BertTokenizer.from_pretrained(dataset)




# Tokenize the input text
tokens = tokenizer.encode_plus(
    dataset,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Pass the tokenized input through the BERT model
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits

# Get the predicted label
predicted_label = torch.argmax(logits, dim=1).item()

# Print the predicted label
print(f"Predicted label: {predicted_label}")

