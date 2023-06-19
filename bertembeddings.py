import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging




dataset = pd.read_csv('tokenized_data_without_punctuation_stopwords_lemma.csv')
dataset = dataset['text'].astype(str).tolist()

# Set log level to ERROR to suppress warning messages
logging.getLogger("transformers").setLevel(logging.ERROR)




model_name = 'bert-base-uncased'  # Pre-trained BERT model
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def get_bert_embeddings(dataset):
    encoded_input = tokenizer(dataset, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.squeeze(dim=0)  # Extract embeddings
    return embeddings

embeddings = get_bert_embeddings(dataset)
print(embeddings)
