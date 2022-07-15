import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self):
        self.itos = {0: '[PAD]', 1: '[START]', 2: '[STOP]', 3: '[CPY]', 4: '[UNK]'}

        self.stoi = {k:j for j,k in self.itos.items()}

    def __len__(self):
        return len(self.itos)
        # return len(self.vocab)

    def build_vocabulary(self, train_df, column_name):
        indx = 5

        self.vocab = set()
        for line in train_df[column_name]:
            for token in line:
                self.vocab.add(token)


        words = sorted(list(self.vocab))

        for word in words: 
            if (word not in self.stoi.keys()):
                self.stoi[word] = indx 
                indx += 1

        self.itos = {k:j for j,k in self.stoi.items()}


    def numericalize(self, text):
        numericalized_text = []

        for token in text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else: #out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi['[UNK]'])

        return numericalized_text

    def get_max_len(self, train_df, column_name):
        maxlist = max(train_df[column_name], key=len)
        return len(maxlist)


class TrainDataset(Dataset):

    def __init__(self, df, src_col, targ_col):
        self.df = df 
        self.src_col = src_col
        self.targ_col = targ_col

        self.source_vocab = Vocabulary()
        self.source_vocab.build_vocabulary(self.df, src_col)

        self.target_vocab = Vocabulary()
        self.target_vocab.build_vocabulary(self.df, targ_col)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        src_text = self.df.iloc[index][self.src_col]
        target_text = self.df.iloc[index][self.targ_col]

        numeric_src = self.source_vocab.numericalize(src_text)
        numeric_targ = self.target_vocab.numericalize(target_text)

        return torch.Tensor(numeric_src), torch.Tensor(numeric_targ)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[0] for item in batch] 
        #pad source sentences to max source length in the batch
        source = pad_sequence(source, batch_first=False, padding_value = self.pad_idx) 
        
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch] 
        #pad target sentences to max target length in the batch
        target = pad_sequence(target, batch_first=False, padding_value = self.pad_idx)

        # TODO Consider adding pack_pad_sequence to save computations

        return source, target



def get_train_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True): #increase num_workers according to CPU
    #get pad_idx for collate fn
    pad_idx = dataset.source_vocab.stoi['[PAD]']
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx)) #MyCollate class runs __call__ method by default
    return loader



