import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging

dataset = pd.read_csv('tokenized_data_without_punctuation_stopwords_lemma.csv')
dataset = dataset[:1000]
dataset = dataset['text'].astype(str).tolist()

# Set log level to ERROR to suppress warning messages
logging.getLogger("transformers").setLevel(logging.ERROR)

model_name = 'bert-base-uncased'  # Pre-trained BERT model
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def get_bert_embeddings(dataset):
    embeddings = []
    for sentence in dataset:
        encoded_input = tokenizer.encode_plus(sentence, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        with torch.no_grad():
            model_output = model(input_ids)
        sentence_embeddings = model_output.last_hidden_state.squeeze(dim=0)  # Extract embeddings
        embeddings.append(sentence_embeddings)
    return embeddings

embeddings = get_bert_embeddings(dataset)
print(embeddings)
