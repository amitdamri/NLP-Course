import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
import re
from nltk.stem.snowball import EnglishStemmer

import pandas as pd
import torch

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

device = torch.device('cpu')

stemmer = EnglishStemmer()

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(lower=True, include_lengths=True, batch_first=True)


def create_train_val_test_files(x_train, x_test, y_train, y_test):
    """
    Create train, val and test files from the train and test files
    :param x_train: dataframe of x features of the train data
    :param x_test: dataframe of x features of the test data
    :param y_train: series of y values of the train data
    :param y_test: series of y values of test data
    :return:
    """

    # data splitting
    train_val_ratio = 0.8

    # Train-validation split
    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train)

    # Concatenate splits of different labels
    df_train = pd.DataFrame(x_train_split)
    df_train['clean_text'] = df_train['clean_text'].apply(lambda x: " ".join(x.split(' ')[:10]))
    df_train['label'] = y_train_split

    df_val = pd.DataFrame(x_val)
    df_val['clean_text'] = df_val['clean_text'].apply(lambda x: " ".join(x.split(' ')[:10]))
    df_val['label'] = y_val

    df_test = x_test.copy()
    df_test['clean_text'] = df_test['clean_text'].apply(lambda x: " ".join(x.split(' ')[:10]))
    df_test['label'] = y_test

    df_train[['label', 'clean_text']].to_csv('train.csv', index=False)
    df_val[['label', 'clean_text']].to_csv('val.csv', index=False)
    df_test[['label', 'clean_text']].to_csv('test.csv', index=False)


class LSTM(nn.Module):
    """
    LSTM model class
    """

    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 100)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=100,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(dimension, 1)

    def forward(self, text, text_len):
        """
        implementation of the forward pass in the model
        :param text: text of a tweet
        :param text_len: length of the text
        :return: the output of the forward pass
        """
        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        text_fea = self.drop(out_forward)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


def save_checkpoint(save_path, model, optimizer, valid_loss):
    """
    saving model checkpoint
    :param save_path: the path to save the model to
    :param model: the model to save
    :param optimizer: the optimizer that was used
    :param valid_loss: the loss of the validation set in the model 
    """
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    """
    load a model that was saved in a checkpoint
    :param load_path: the path to load the model from
    :param model: the model object to load the model into
    :param optimizer: the optimizer object that will be used in the model 
    :return: the validation loss of the loaded model
    """
    
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    """
    save metrics into file
    :param save_path: the file path to save the metrics to
    :param train_loss_list: list of train losses to save
    :param valid_loss_list: list of validation losses to save
    :param global_steps_list:  global steps list to save
    """
    
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    """
    load metrics from a file
    :param load_path: the path the file is in
    :return: train loss list, validation loss list and global steps list
    """
    
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def train_model(model,
                optimizer,
                criterion,  # =nn.BCELoss(),
                train_loader,
                valid_loader,
                num_epochs,
                eval_every,
                file_path,
                best_valid_loss=float("Inf")):
    """
    train an LSTM model
    :param model: the LSTM model to train
    :param optimizer: the optimizer to train the model with
    :param criterion: the loss function
    :param train_loader: an iterator on the train set
    :param valid_loader: an iterator no the validation set
    :param num_epochs: number of epochs to train on
    :param eval_every: every number of epochs to evaluate the training process
    :param file_path: the file path to save the model to in relevant checkpoints
    :param best_valid_loss: the best validation loss if we got in the past and we want the model to save
                            a new model in a checkpoint only if we the model received a better validation loss
    :return: 
    """
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, (text, text_len)), _ in train_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for (labels, (text, text_len)), _ in valid_loader:
                        labels = labels.to(device)
                        text = text.to(device)
                        text_len = text_len.to(device)
                        output = model(text, text_len)

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + 'model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


def evaluate(model, test_loader, threshold=0.5):
    """
    evaluate a model
    :param model: the model to evaluate
    :param test_loader: an iterator of test samples to evaluate
    :param threshold: the threshold that above it the label is 1 and below it the label would be 0
    :return: a string of classification report on the evaluated model
    """
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, (text, text_len)), _ in test_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    results = classification_report(y_true, y_pred, labels=[1, 0], digits=4, output_dict = True)
    print(results)

    return results


def fit(x_train, x_test, y_train, y_test):
    """
    start an LSTM model training process
    :param x_train: a dataframe of the x features in the train set
    :param x_test: a dataframe of the x features in the test set
    :param y_train: the labels of the train set
    :param y_test: the labels of the test set
    :return: the trained best model and the accuracy on of the best model on the test set
    """
    # Create train, val and test sets
    create_train_val_test_files(x_train, x_test, y_train, y_test)

    # Load dataset
    # Fields
    fields = [('label', label_field), ('text', text_field)]

    # TabularDataset load
    train, valid, test = TabularDataset.splits(path="", train='train.csv', validation='val.csv',
                                               test='test.csv',
                                               format='CSV', fields=fields, skip_header=True)

    # Iterators
    train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text),
                                device=device, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.text),
                                device=device, sort=True, sort_within_batch=True)
    test_iter = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.text),
                               device=device, sort=True, sort_within_batch=True)

    # Vocabulary
    text_field.build_vocab(train, min_freq=1)

    # Build model
    model = LSTM(dimension=64).to(device)
    optimizer = optim.Adam(model.parameters())  # ), lr=0.001)

    # pretrained embedding
    embedding_glove = GloVe(name='6B', dim=100)
    pretrained_embedding = embedding_glove.get_vecs_by_tokens(text_field.vocab.itos)
    model.embedding.weight.data = pretrained_embedding

    # Train model
    train_model(model=model, optimizer=optimizer, criterion=nn.BCELoss(), train_loader=train_iter,
                valid_loader=valid_iter,
                num_epochs=3, eval_every=len(train_iter), file_path="", best_valid_loss=float("Inf"))

    # test model
    best_model = LSTM(dimension=64).to(device)

    # optimizer = optim.Adam(best_model.parameters(), lr=0.0001, weight_decay=3)
    optimizer = optim.Adam(best_model.parameters(), weight_decay=3)

    load_checkpoint('model.pt', best_model, optimizer)
    test_results = evaluate(best_model, test_iter)

    return best_model, test_results['accuracy']
