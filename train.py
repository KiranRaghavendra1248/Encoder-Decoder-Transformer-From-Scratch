# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from spacy.tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
import os
import re
from nltk.corpus import stopwords
import random
from tqdm import tqdm
from utils import *
from models import *

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset
train_dataset,test_dataset,glove_vectors,vocab_len = get_dataset_train_test()

# Instantiate the model
input_dim = vocab_len
output_dim = vocab_len
learning_rate = 0.001
embedding_dim = 300
hidden_dim = 512
n_layers = 2
dropout = 0.2
num_epochs = 25
num_workers = 2

encoder = Encoder(glove_vectors, embedding_dim,
                  hidden_dim, n_layers, dropout)
decoder = Decoder(glove_vectors, output_dim,
                  embedding_dim, hidden_dim,
                  n_layers, dropout)
model = EncDecLSTM(encoder, decoder)
print(model)

# Specify optimizer and loss function
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)
loss_fun = nn.CrossEntropyLoss()

# Create dataloaders
train_loader = DataLoader(train_dataset,
                          batch_size=8,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=num_workers)

test_loader = DataLoader(test_dataset,
                         batch_size=8,
                         shuffle=False,
                         collate_fn=collate_fn,
                         num_workers=num_workers)


train_loop(model, train_loader,
           loss_fun, optimizer, device, num_epochs)

