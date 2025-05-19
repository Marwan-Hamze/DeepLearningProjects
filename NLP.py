import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

'''
This is a simple Text Classification problem using Bert. 
'''

# Sample dataset
data = {
    'text': ["I love this movie", "I hate this movie", "This was fantastic!", "Not my taste at all", "Absolutely wonderful experience", "Worst movie ever",
             "Meh", "Good", "Great", "Just bad", "Not worth it", "Brilliant"],
    'label': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# # There seems no need to go through Pandas, and instead split the data like this:
# train_texts_data, val_texts_data, train_labels_data, val_labels_data = train_test_split(data['text'], data['label'], test_size=0.2)
# # train_texts_data is the same as list(train_text), which is used for the tokenizer.
# print(train_texts_data)

# Print/Visualize

# print(f"train_texts type: {type(train_texts)}| train_labels type: {type(train_labels)}")
# print(df)
# print(train_texts)
# print(train_texts.values)
# print(list(train_texts))

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=64)

# Print/Visualize
# print(type(train_encodings))
# print(train_encodings)
# print(train_encodings['input_ids'])

# Convert to PyTorch tensors
train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
val_inputs = torch.tensor(val_encodings['input_ids'])
val_masks = torch.tensor(val_encodings['attention_mask'])

# Print/Visualize
# print(train_inputs)

# Grouping the data in Batches
train_data = torch.utils.data.TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

val_data = torch.utils.data.TensorDataset(val_inputs, val_masks, val_labels)
val_loader = DataLoader(val_data, batch_size=4)

# Define the Model and the Optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 20
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        optimizer.zero_grad()
        
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    if epoch == 0:    
        print(f'Epoch {epoch + 1} completed. Loss = {loss}')
    elif (epoch+1)%5 == 0:
        print(f'Epoch {epoch + 1} completed. Loss = {loss}')

# Inference

model.eval()
predictions, true_labels = [], []

with torch.inference_mode():
    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy()) 
        # argmax on logits works without applying softmax: softmax is a monotonic transformation: a>b --> soft(a) > soft(b)
        true_labels.extend(b_labels.cpu().numpy())

print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive']))



