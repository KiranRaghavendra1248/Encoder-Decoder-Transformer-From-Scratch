# Imports
import torch
from torch.utils.data import Dataset
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

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

stop_words = set(stopwords.words('english'))

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

def create_dataframe(source_text_path,
                     target_text_path):
    txt_files_source = [file for file in
                        os.listdir(
                            source_text_path) if
                        file.endswith('.txt')]
    txt_files_target = [file for file in
                        os.listdir(
                            target_text_path) if
                        file.endswith('.txt')]
    df = pd.DataFrame(
        columns=['headlines', 'text'])
    for source, target in zip(txt_files_source,
                              txt_files_target):
        assert source == target
        source_file_path = os.path.join(
            source_text_path, source)
        target_file_path = os.path.join(
            target_text_path, target)
        # Read the content of the file
        with open(source_file_path, 'r',
                  encoding='latin-1') as file:
            source_text = file.read()
        with open(target_file_path, 'r',
                  encoding='latin-1') as file:
            target_text = file.read()
        df.loc[len(df.index)] = [source_text,
                                 target_text]
    return df


# Check accuracy function
def check_accuracy(output, labels):
    _, predpos = output.max(1)
    num_samples = len(labels)
    num_correct = (predpos == labels).sum()
    return (num_correct / num_samples) * 100


# Save checkpoint
def save_checkpoint(state,
                    filename='weights.pth.tar'):
    print('Saving weights-->')
    torch.save(state, filename)


# Load checkpoint
def load_checkpoint(checkpoint, model, optim):
    print('Loading weights-->')
    model.load_state_dict(
        checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, source_texts,
                 target_summaries, source_vocab,
                 target_vocab):
        self.source_texts = source_texts
        self.target_summaries = target_summaries
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = [self.source_vocab[word] for
                       word in
                       self.source_texts[idx]]
        target_summary = [self.target_vocab[word]
                          for word in
                          self.target_summaries[
                              idx]]
        return torch.tensor(
            source_text), torch.tensor(
            target_summary)

def text_cleaner(text):
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '',
                       newString)
    newString = re.sub('"', '', newString)
    newString = ' '.join([contraction_mapping[
                              t] if t in contraction_mapping else t
                          for t in
                          newString.split(" ")])
    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ",
                       newString)
    tokens = [w for w in newString.split() if
              not w in stop_words]
    return " ".join(tokens)
def get_dataset_train_test():
    df1 = create_dataframe(
        "/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/business",
        "/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/business")
    df2 = create_dataframe(
        "/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/entertainment",
        "/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/entertainment")
    df3 = create_dataframe(
        "/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/politics",
        "/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/politics")
    df4 = create_dataframe(
        "/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/sport",
        "/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/sport")
    df5 = create_dataframe(
        "/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/tech",
        "/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/tech")

    df = pd.concat([df1, df2, df3, df4, df5],
                   ignore_index=True)

    # Split into train and test sets
    df = df.rename(
        columns={"headlines": "source_text",
                 "text": "summary_text"})
    X, Y = df["source_text"], df["summary_text"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    train_df = pd.DataFrame(
        {'source_text': X_train,
         'summary_text': Y_train})
    test_df = pd.DataFrame({'source_text': X_test,
                            'summary_text': Y_test})


    # Tokenize and lowercase text using spacy
    train_df['source_text'] = train_df[
        'source_text'].apply(
        lambda x: [token.text.lower() for token in
                   tokenizer(text_cleaner(x))])
    train_df['summary_text'] = train_df[
        'summary_text'].apply(
        lambda x: [token.text.lower() for token in
                   tokenizer(text_cleaner(x))])

    test_df['source_text'] = test_df[
        'source_text'].apply(
        lambda x: [token.text.lower() for token in
                   tokenizer(text_cleaner(x))])
    test_df['summary_text'] = test_df[
        'summary_text'].apply(
        lambda x: [token.text.lower() for token in
                   tokenizer(text_cleaner(x))])

    # Add START AND END tokens to summary
    train_df['source_text'] = train_df[
        'source_text'].apply(
        lambda x: ['_START_'] + x + ['_END_'])
    train_df['summary_text'] = train_df[
        'summary_text'].apply(
        lambda x: ['_START_'] + x + ['_END_'])

    test_df['source_text'] = test_df[
        'source_text'].apply(
        lambda x: ['_START_'] + x + ['_END_'])
    test_df['summary_text'] = test_df[
        'summary_text'].apply(
        lambda x: ['_START_'] + x + ['_END_'])

    all_tokens = train_df[
                     'source_text'].tolist() + \
                 train_df['summary_text'].tolist()
    source_vocab = {actual_word: idx for
                    idx, (word_num, actual_word)
                    in
                    enumerate(
                        sorted(enumerate(set(
                            token for tokens in
                            all_tokens
                            for token in tokens)),
                            key=lambda x: x[
                                1]))}
    target_vocab = {actual_word: idx for
                    idx, (word_num, actual_word)
                    in
                    enumerate(
                        sorted(enumerate(set(
                            token for tokens in
                            all_tokens
                            for token in tokens)),
                            key=lambda x: x[
                                1]))}

    global_vectors = GloVe(name='6B', dim=300)

    source_vectors = torch.stack(
        [global_vectors.get_vecs_by_tokens(word)
         for
         word, idx in sorted(source_vocab.items(),
                             key=lambda x: x[1])])

    # Create custom datasets
    train_dataset = CustomDataset(
        train_df['source_text'].tolist(),
        train_df['summary_text'].tolist(),
        source_vocab, target_vocab)
    test_dataset = CustomDataset(
        test_df['source_text'].tolist(),
        test_df['summary_text'].tolist(),
        source_vocab, target_vocab)

    return train_dataset, test_dataset, source_vectors, len(source_vocab)

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

# Define collate function for DataLoader
def collate_fn(batch):
    sources, targets = zip(*batch)
    padded_sources = pad_sequence(sources,
                                  batch_first=True)
    padded_targets = pad_sequence(targets,
                                  batch_first=True)
    return padded_sources, padded_targets

def train_loop(model, dataloader,
               loss_fun, optimizer, device, num_epochs):
    model.train()
    model.to(device)
    min_loss = None
    for epoch in range(num_epochs):
        losses = []
        accuracies = []
        loop = tqdm(enumerate(dataloader),
                    total=len(dataloader),
                    leave=True)
        for batch, (x, y) in loop:
            # put on cuda
            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(x, y)

            # calculate loss & accuracy
            loss = loss_fun(y_pred, y.reshape(-1))
            losses.append(loss.detach().item())

            accuracy = check_accuracy(y_pred,
                                      y.reshape(
                                          -1))
            accuracies.append(
                accuracy.detach().item())

            # zero out prior gradients
            optimizer.zero_grad()

            # backprop
            loss.backward()

            # update weights
            optimizer.step()

            # Update TQDM progress bar
            loop.set_description(
                f"Epoch [{epoch}/{num_epochs}] ")
            loop.set_postfix(
                loss=loss.detach().item(),
                accuracy=accuracy.detach().item())

        moving_loss = sum(losses) / len(losses)
        moving_accuracy = sum(accuracies) / len(
            accuracies)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
        # Save check point
        if min_loss == None:
            min_loss = moving_loss
            save_checkpoint(checkpoint)
        elif moving_loss < min_loss:
            min_loss = moving_loss
            save_checkpoint(checkpoint)
        print(
            'Epoch {0} : Loss = {1} , Accuracy={2}'.format(
                epoch, moving_loss,
                moving_accuracy))
