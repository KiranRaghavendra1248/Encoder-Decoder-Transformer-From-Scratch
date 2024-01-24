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

# Add data from files into dataframe for easier access
def create_dataframe(source_text_path,target_text_path):
    txt_files_source = [file for file in os.listdir(source_text_path) if file.endswith('.txt')]
    txt_files_target = [file for file in os.listdir(target_text_path) if file.endswith('.txt')]
    df = pd.DataFrame(columns=['headlines','text'])
    for source,target in zip(txt_files_source,txt_files_target):
        assert source==target
        source_file_path = os.path.join(source_text_path, source)
        target_file_path = os.path.join(target_text_path, target)
        # Read the content of the file
        with open(source_file_path,'r',encoding='latin-1') as file:
            source_text = file.read()
        with open(target_file_path,'r',encoding='latin-1') as file:
            target_text = file.read()
        df.loc[len(df.index)] = [source_text,target_text]
    return df

# Create and return tokenizer
def get_tokenizer():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)
    return tokenizer

# Return contraction mapping
def get_contraction_mapping():
    contraction_mapping = {"ain't": "is not",
                           "aren't": "are not",
                           "can't": "cannot",
                           "'cause": "because",
                           "could've": "could have",
                           "couldn't": "could not",

                           "didn't": "did not",
                           "doesn't": "does not",
                           "don't": "do not",
                           "hadn't": "had not",
                           "hasn't": "has not",
                           "haven't": "have not",

                           "he'd": "he would",
                           "he'll": "he will",
                           "he's": "he is",
                           "how'd": "how did",
                           "how'd'y": "how do you",
                           "how'll": "how will",
                           "how's": "how is",

                           "I'd": "I would",
                           "I'd've": "I would have",
                           "I'll": "I will",
                           "I'll've": "I will have",
                           "I'm": "I am",
                           "I've": "I have",
                           "i'd": "i would",

                           "i'd've": "i would have",
                           "i'll": "i will",
                           "i'll've": "i will have",
                           "i'm": "i am",
                           "i've": "i have",
                           "isn't": "is not",
                           "it'd": "it would",

                           "it'd've": "it would have",
                           "it'll": "it will",
                           "it'll've": "it will have",
                           "it's": "it is",
                           "let's": "let us",
                           "ma'am": "madam",

                           "mayn't": "may not",
                           "might've": "might have",
                           "mightn't": "might not",
                           "mightn't've": "might not have",
                           "must've": "must have",

                           "mustn't": "must not",
                           "mustn't've": "must not have",
                           "needn't": "need not",
                           "needn't've": "need not have",
                           "o'clock": "of the clock",

                           "oughtn't": "ought not",
                           "oughtn't've": "ought not have",
                           "shan't": "shall not",
                           "sha'n't": "shall not",
                           "shan't've": "shall not have",

                           "she'd": "she would",
                           "she'd've": "she would have",
                           "she'll": "she will",
                           "she'll've": "she will have",
                           "she's": "she is",

                           "should've": "should have",
                           "shouldn't": "should not",
                           "shouldn't've": "should not have",
                           "so've": "so have",
                           "so's": "so as",

                           "this's": "this is",
                           "that'd": "that would",
                           "that'd've": "that would have",
                           "that's": "that is",
                           "there'd": "there would",

                           "there'd've": "there would have",
                           "there's": "there is",
                           "here's": "here is",
                           "they'd": "they would",
                           "they'd've": "they would have",

                           "they'll": "they will",
                           "they'll've": "they will have",
                           "they're": "they are",
                           "they've": "they have",
                           "to've": "to have",

                           "wasn't": "was not",
                           "we'd": "we would",
                           "we'd've": "we would have",
                           "we'll": "we will",
                           "we'll've": "we will have",
                           "we're": "we are",

                           "we've": "we have",
                           "weren't": "were not",
                           "what'll": "what will",
                           "what'll've": "what will have",
                           "what're": "what are",

                           "what's": "what is",
                           "what've": "what have",
                           "when's": "when is",
                           "when've": "when have",
                           "where'd": "where did",
                           "where's": "where is",

                           "where've": "where have",
                           "who'll": "who will",
                           "who'll've": "who will have",
                           "who's": "who is",
                           "who've": "who have",

                           "why's": "why is",
                           "why've": "why have",
                           "will've": "will have",
                           "won't": "will not",
                           "won't've": "will not have",

                           "would've": "would have",
                           "wouldn't": "would not",
                           "wouldn't've": "would not have",
                           "y'all": "you all",

                           "y'all'd": "you all would",
                           "y'all'd've": "you all would have",
                           "y'all're": "you all are",
                           "y'all've": "you all have",

                           "you'd": "you would",
                           "you'd've": "you would have",
                           "you'll": "you will",
                           "you'll've": "you will have",

                           "you're": "you are",
                           "you've": "you have"}

    return contraction_mapping

def text_cleaner(text,contraction_mapping,stop_words):
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    tokens = [w for w in newString.split() if not w in stop_words]
    return " ".join(tokens)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, source_texts, target_summaries, source_vocab, target_vocab):
        self.source_texts = source_texts
        self.target_summaries = target_summaries
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = [self.source_vocab[word] for word in self.source_texts[idx]]
        target_summary = [self.target_vocab[word] for word in self.target_summaries[idx]]
        return torch.tensor(source_text), torch.tensor(target_summary)

# Define collate function for DataLoader
def collate_fn(batch):
    sources, targets = zip(*batch)
    padded_sources = pad_sequence(sources, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)
    return padded_sources, padded_targets

'''
Note : 
In PyTorch, the `collate_fn` parameter in the `DataLoader` can be either a function or an object of a class. Both approaches are valid, and the choice depends on your preference and the complexity of your collation logic.

1. Function as `collate_fn`:
def my_collate_fn(batch):
    # Your custom collation logic here
    return processed_batch
# Use the function with DataLoader
train_loader = DataLoader(dataset, batch_size=64, collate_fn=my_collate_fn)

2. Class as `collate_fn`:
class MyCollateClass:
    def __call__(self, batch):
        # Your custom collation logic here
        return processed_batch
# Instantiate the class and use it with DataLoader
my_collate_instance = MyCollateClass()
train_loader = DataLoader(dataset, batch_size=64, collate_fn=my_collate_instance)

Using a class allows you to maintain state between batches if needed, as the class instance retains its state between calls. This can be beneficial if your collation logic requires some persistent information.

The key point is that the `collate_fn` parameter should be a callable (a function or an object with a `__call__` method) that takes a list of batch data and returns the processed batch. The processing typically involves padding sequences, converting data types, or any other necessary steps to prepare the batch for the model.
'''