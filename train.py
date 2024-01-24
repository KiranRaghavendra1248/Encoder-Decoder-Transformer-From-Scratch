# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from spacy.tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords
from utils import *
from models import *

# Create NLTK tokenizer
tokenizer = get_tokenizer()
# Load pretrained GloVe embeddings
global_vectors = GloVe(name='6B', dim=100)

df1 = create_dataframe("/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/business","/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/business")
df2 = create_dataframe("/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/entertainment","/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/entertainment")
df3 = create_dataframe("/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/politics","/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/politics")
df4 = create_dataframe("/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/sport","/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/sport")
df5 = create_dataframe("/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/tech","/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/tech")

# Merge into single df
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# Split into train and test sets
df = df.rename(columns = {"headlines":"source_text","text":"summary_text"})
X,Y = df["source_text"],df["summary_text"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
train_df = pd.DataFrame({'source_text': X_train, 'summary_text': Y_train})
test_df = pd.DataFrame({'source_text': X_test, 'summary_text': Y_test})

# Get contraction mapping
contraction_mapping = get_contraction_mapping()
# Create stop_words set
stop_words = set(stopwords.words('english'))

# Tokenize and lowercase text using spacy
train_df['source_text'] = train_df['source_text'].apply(lambda x: [token.text.lower() for token in tokenizer(text_cleaner(x,contraction_mapping, stop_words))])
train_df['summary_text'] = train_df['summary_text'].apply(lambda x: [token.text.lower() for token in tokenizer(text_cleaner(x,contraction_mapping, stop_words))])

test_df['source_text'] = test_df['source_text'].apply(lambda x: [token.text.lower() for token in tokenizer(text_cleaner(x,contraction_mapping, stop_words))])
test_df['summary_text'] = test_df['summary_text'].apply(lambda x: [token.text.lower() for token in tokenizer(text_cleaner(x,contraction_mapping, stop_words))])

# Add START AND END tokens to summary
train_df['source_text'] = train_df['source_text'].apply(lambda x : ['_START_']+ x + ['_END_'])
train_df['summary_text'] = train_df['summary_text'].apply(lambda x : ['_START_']+ x + ['_END_'])

test_df['source_text'] = test_df['source_text'].apply(lambda x : ['_START_']+ x + ['_END_'])
test_df['summary_text'] = test_df['summary_text'].apply(lambda x : ['_START_']+ x + ['_END_'])

# Build vocabularies - each word has an index, note : words sorted in ascending order
all_tokens = train_df['source_text'].tolist() + train_df['summary_text'].tolist()
source_vocab = {actual_word: idx + 1 for idx, (word_num, actual_word) in enumerate(sorted(enumerate(set(token for tokens in all_tokens for token in tokens)), key=lambda x: x[1]))}
target_vocab = {actual_word: idx + 1 for idx, (word_num, actual_word) in enumerate(sorted(enumerate(set(token for tokens in all_tokens for token in tokens)), key=lambda x: x[1]))}

# Create source vectors to be used by embedding layer
source_vectors = torch.stack([global_vectors.get_vecs_by_tokens(word) for word, idx in sorted(source_vocab.items(), key=lambda x: x[1])])

# Create custom datasets
train_dataset = CustomDataset(train_df['source_text'].tolist(), train_df['summary_text'].tolist(), source_vocab, target_vocab)
test_dataset = CustomDataset(test_df['source_text'].tolist(), test_df['summary_text'].tolist(), source_vocab, target_vocab)

# Params
source_vocab_size = len(source_vocab)
target_vocab_size = len(target_vocab)
embedding_dim = 100
hidden_dim = 512
n_layers = 2
dropout = 0.3
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = Seq2SeqLSTM(source_vocab_size, target_vocab_size, embedding_dim, hidden_dim, n_layers, dropout, source_vectors).to(device)

# Specify optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    for batch in train_loader:
        source_text, target_summary = batch
        output = model(source_text, target_summary)
        output_dim = output.shape[-1]

        loss = criterion(output.view(-1, output_dim), target_summary.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




