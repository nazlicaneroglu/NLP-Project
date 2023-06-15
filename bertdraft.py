import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('tokenized_data_without_punctuation_stopwords_lemma.csv')

# Splitting the data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

#train_data = train_data.dropna(subset=['sdg'])
#train_labels = [int(label) for label in train_data['sdg'].tolist()]
#val_labels = [int(label) for label in val_data['sdg'].tolist()]

train_data = train_data.dropna(subset=['sdg'])


default_label = 10  # Choose a default label value

train_labels = []
for label in train_data['sdg'].tolist():
    if pd.isna(label) or not label.isdigit():
        train_labels.append(default_label)
    else:
        train_labels.append(int(label))

val_labels = []
for label in val_data['sdg'].tolist():
    if pd.isna(label) or not label.isdigit():
        val_labels.append(default_label)
    else:
        val_labels.append(int(label))


train_data = train_data[pd.to_numeric(train_data['sdg'], errors='coerce').notnull()]
train_labels = [int(label) for label in train_data['sdg'].tolist()]

val_data = val_data[pd.to_numeric(val_data['sdg'], errors='coerce').notnull()]
val_labels = [int(label) for label in val_data['sdg'].tolist()]




train_labels = [int(label) if not pd.isna(label) else default_label for label in train_data['sdg'].tolist()]
val_labels = [int(label) if not pd.isna(label) else default_label for label in val_data['sdg'].tolist()]


train_data = train_data[pd.to_numeric(train_data['sdg'], errors='coerce').notnull()]
train_labels = [int(label) for label in train_data['sdg'].tolist()]
val_labels = [int(label) for label in val_data['sdg'].tolist()]



#train_data = train_data[train_data['sdg'].astype(str).str.isdigit()]
#train_labels = [int(label) for label in train_data['sdg'].tolist()]
#val_labels = [int(label) for label in val_data['sdg'].tolist()]


import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt


# BERT-specific configuration
pretrained_model = 'bert-base-uncased'
num_classes = len(data['sdg'].unique())  # Number of unique labels
max_length = 128
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

# Data processing
# Data processing
train_encodings = tokenizer(train_data['text'].astype(str).tolist(), truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(val_data['text'].astype(str).tolist(), truncation=True, padding=True, max_length=max_length)

#train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
#val_encodings = tokenizer(val_data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
#train_labels = [int(label) for label in train_data['sdg'].tolist()]
#val_labels   = [int(label) for label in val_data['sdg'].tolist()]

#train_labels = train_data['sdg'].tolist()
#val_labels = val_data['sdg'].tolist()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['sdg'] = torch.tensor(int(self.labels[idx]))  # Convert the label to an integer
        return item

    def __len__(self):
        return len(self.labels)


"""
# Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['sdg'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
"""

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model initialization
model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_classes)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

training_loss = []
validation_loss = []
validation_accuracy = []
training_time = []
validation_time = []

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc='Epoch {}'.format(epoch + 1), leave=False):
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'sdg'}
        labels = batch['sdg'].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        num_batches += 1

    avg_train_loss = total_loss / num_batches
    training_loss.append(avg_train_loss)

    # Print Training Loss
    print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'sdg'}
            labels = batch['sdg'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    validation_loss.append(avg_val_loss)

    # Print Validation Loss
    print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

    # Calculate and Print Validation Accuracy
    accuracy = correct_predictions / total_predictions
    validation_accuracy.append(accuracy)
    print(f"Epoch {epoch+1} - Validation Accuracy: {accuracy:.4f}")

# Plotting the Training and Validation Loss
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Printing the Training and Validation Metrics
print("Training Loss:", training_loss)
print("Validation Loss:", validation_loss)
print("Validation Accuracy:", validation_accuracy)
