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

# Define the seq2seq LSTM model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim, hidden_dim, n_layers, dropout, source_vectors):
        super(Seq2SeqLSTM, self).__init__()

        # Use pretrained GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(source_vectors, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, source_text, target_summary):
        embedded = self.embedding(source_text)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        return output

