#import pandas and ncessary libraries for reading the train data
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

#read the train data
train = pd.read_csv('data/TrainingData.csv')

#split the data into train and validation
train, val = train_test_split(train, test_size=0.2, random_state=42)

#load the model
model, tokenizer, device = load_model(model_config, return_model=True)

#tokenize the data
train_encodings = tokenizer(train.text.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val.text.tolist(), truncation=True, padding=True)

#convert the data into tensors
train_labels = torch.tensor(train.label.values)
val_labels = torch.tensor(val.label.values)

#convert the data into tensors
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
#load the data into the model
train_dataset = ToxicDataset(train_encodings, train_labels)
val_dataset = ToxicDataset(val_encodings, val_labels)

#train the model
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(




