import pandas as pd
import torch
from typing import Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence

# Vocab indices of constants
PAD_INDEX = 0
START_INDEX = 1
STOP_INDEX = 2
CPY_INDEX = 3
UNKNOWN_INDEX = 4

# Constants for vocabulary
PAD_TOKEN = '[PAD]'
START_TOKEN = '[START]'
STOP_TOKEN = '[STOP]'
CPY_TOKEN = '[CPY]'
UNKNOWN_TOKEN = '[UNK]'

RESERVED_TOKENS = {PAD_TOKEN, START_TOKEN, STOP_TOKEN, CPY_TOKEN, UNKNOWN_TOKEN}
NON_CPY_TOKENS = {PAD_TOKEN, START_TOKEN, STOP_TOKEN, UNKNOWN_TOKEN}
class Vocabulary:
    """
        Representation of a vocabulary/language as several dictionaries
        
        
        Attributes:
            name (str): name of the vocabulary/language (e.g. 'english', 'pseudo', 'code' etc)
            itos (dict): dictionary of index to token/string (e.g. {0: '[PAD]', 1: '[START]', 2: '[STOP]', 3: '[CPY]', 4: '[UNK]', 5: 'int'})
            stoi (dict): dictionary of token/string to indices (e.g. {'[PAD]': 0, '[START]': 1, '[STOP]': 2, '[CPY]': 3, '[UNK]': 4, 'int': 5})
            vocab (set): set of all tokens in the vocabulary (e.g. {'[PAD]', '[START]', '[STOP]', '[CPY]', '[UNK]', 'int'})
            n_words (int): number of tokens in the vocabulary (e.g. 6)
        """
    def __init__(self, name: str):
        """
        Args:
            name: name of the language
        """
        self.name = name
        self.itos = {
            PAD_INDEX: PAD_TOKEN, 
            START_INDEX: START_TOKEN, 
            STOP_INDEX: STOP_TOKEN, 
            CPY_INDEX: CPY_TOKEN,
            UNKNOWN_INDEX: UNKNOWN_TOKEN
        }
        self.stoi = {k:j for j,k in self.itos.items()}
        self.vocab = set()
        self.n_words = len(self.itos)

    def __len__(self):
        return len(self.itos)
        

    def build_vocabulary(self, train_df: pd.DataFrame, column_name: str):
        """
        Build a vocabulary from a dataframe column containing lists of tokens
        """
        # Empty vocab set
        self.vocab = set()

        # Loop through every row in the dataframe
        # TODO: parallelize this
        for line in train_df[column_name]:
            # Augment the vocab
            for token in line:
                self.vocab.add(token)

        # Sort the vocabulary and store in a list
        # This is being done so that the indices are always
        # the same for the same vocabulary (e.g. if the vocab is
        # {'a', 'b', 'c'} then the indices are always {0, 1, 2})

        words = sorted(list(self.vocab))

        for word in words: 
            # If word not in vocabulary (should not be unless it is a reserved token)
            if word not in self.stoi:
                self.stoi[word] = self.n_words
                self.itos[self.n_words] = word 
                self.n_words += 1
            elif word in RESERVED_TOKENS:
                pass
            else:
                raise ValueError(f'Word "{word}" already in vocabulary')
        


    def numericalize(self, text: list) -> list:
        """
        Convert a list of strings to a list of vocab indices. Treats unknown words as '[UNK]'
        """
        numericalized_text = []

        for token in text:
            if token in self.stoi:
                numericalized_text.append(self.stoi[token])
            else: 
                # out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi[UNKNOWN_TOKEN])

        return numericalized_text

    def get_max_len(self, train_df: pd.DataFrame, column_name: str):
        """
        Get the max length (array) of a tokenized sentence in the train dataframe
        """
        maxlist = max(train_df[column_name], key=len)
        return len(maxlist)


class TrainDataset(Dataset):
    """
    Dataset class for training

    Attributes:
        df (DataFrame): dataframe containing the X and y columns
        src_col (str): source column name
        targ_col (str): target column name
        source_vocab (Vocabulary): vocabulary of source column
        target_vocal (Vocabulary): vocabulary of target column
    """
    def __init__(self, df: pd.DataFrame, src_col: str, targ_col: str, use_tokenizer: PreTrainedTokenizer=None):
        """
        Args:
            df (DataFrame): dataframe containing the X and y columns
            src_col (str): source column name
            targ_col (str): target column name
        """
        self.df = df[[src_col, targ_col]]
        self.src_col = src_col
        self.targ_col = targ_col
        self.use_tokenizer = use_tokenizer
        self.source_vocab = None
        self.target_vocab = None

        if self.use_tokenizer is None:

            self.source_vocab = Vocabulary('source_vocab')
            self.source_vocab.build_vocabulary(self.df, src_col)

            self.target_vocab = Vocabulary('target_vocab')
            self.target_vocab.build_vocabulary(self.df, targ_col)

        self._len = df.shape[0]

    def get_pad_index(self) -> int:
        """
        Returns index of the [PAD] token
        """
        if self.use_tokenizer is None:
            return self.source_vocab.stoi[PAD_TOKEN]

        else:
            return self.use_tokenizer(PAD_TOKEN).input_ids[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a tuple of two tensors each storing the 
        indices of the words in the lists
        """
        src_text = self.df.iloc[index][self.src_col]
        target_text = self.df.iloc[index][self.targ_col]

        if self.use_tokenizer is None:
            numeric_src = self.source_vocab.numericalize(src_text)
            numeric_targ = self.target_vocab.numericalize(target_text)
        else:
            numeric_src = [self.use_tokenizer.encode(s)[0] for s in src_text]
            numeric_targ = [self.use_tokenizer.encode(s)[0] for s in target_text]

        # print(numeric_src, numeric_targ)
        # print(type(numeric_src), type(numeric_targ))
        return torch.Tensor(numeric_src), torch.Tensor(numeric_targ)


class TestDataset(Dataset):

    def __init__(self, df, src_col, use_tokenizer: PreTrainedTokenizer=None):
        """
        Args:
            df (DataFrame): dataframe containing the X and y columns
            src_col (str): source column name
            use_tokenizer (PreTrainedTokenizer): tokenizer to use for numericalizing
        """
        self.df = df
        self.src_col = src_col
        self._len = df.shape[0]

        # Tokenizer used in training
        self.use_tokenizer = use_tokenizer

    def get_pad_index(self) -> int:
        """
        Returns index of the [PAD] token
        """
        if self.use_tokenizer is None:
            return None

        else:
            return self.use_tokenizer(PAD_TOKEN).input_ids[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> torch.Tensor:
        src_text = self.df.iloc[index][self.src_col]

        numeric_src = [self.use_tokenizer.encode(s)[0] for s in src_text]

        return torch.Tensor(numeric_src)

class MyCollate:
    """
    Custom collation callable class
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    
    def __call__(self, batch) -> tuple:
        """
        __call__: a default method
        First the obj is created using MyCollate(pad_idx) in data loader
        Then if obj(batch) is called -> __call__ runs by default
        """

        # Get all source indexed sentences of the batch
        source = [item[0] for item in batch] 
        target = [item[1] for item in batch] 
        # Pad source sentences to max source length in the batch
        
        # print('Size', source[0].shape[0])

        max_source_len = max(source, key = lambda x: x.shape[0]).shape[0]
        max_target_len = max(target, key = lambda x: x.shape[0]).shape[0]

        # print(max_source_len, max_target_len)
        max_len = max(max_source_len, max_target_len)

        source[0] = torch.nn.ConstantPad1d((0, max_len - source[0].shape[0]), self.pad_idx)(source[0])
        target[0] = torch.nn.ConstantPad1d((0, max_len - target[0].shape[0]), self.pad_idx)(target[0])

        pad_source = pad_sequence(source, batch_first=False, padding_value=self.pad_idx) 
        
        # Get all target indexed sentences of the batch
        
        # Pad target sentences to max target length in the batch
        pad_target = pad_sequence(target, batch_first=False, padding_value=self.pad_idx)

        # TODO Consider adding pack_pad_sequence to save computations

        return pad_source, pad_target



def get_train_loader(
    dataset: TrainDataset, 
    batch_size: int, 
    num_workers: int=0, 
    shuffle: bool=True, 
    pin_memory: bool=True
) -> DataLoader: #increase num_workers according to CPU
    """
    Create a dataloader for training
    Args:
        dataset (TrainDataset): dataset containing source and target columns
        batch_size (int): batch size of batch training
        num_workers (int): 
    """
    
    # Get pad_idx for collate fn
    pad_idx = dataset.get_pad_index()

    # Define loader
    loader = DataLoader(dataset, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                shuffle=shuffle,
                pin_memory=pin_memory, 
                collate_fn = MyCollate(pad_idx=pad_idx)
            ) 
            
    # MyCollate class runs __call__ method by default
    return loader


def get_test_loader(
    dataset: TestDataset, 
    batch_size: int=1, 
    num_workers: int=0, 
    shuffle: bool=False, 
    pin_memory: bool=True
): #increase num_workers according to CPU
    #get pad_idx for collate fn
    pad_idx = dataset.get_pad_index()
    #define loader
    loader = DataLoader(dataset, 
                batch_size=batch_size, 
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=pin_memory,
                collate_fn = MyCollate(pad_idx=pad_idx)
            ) 
    #MyCollate class runs __call__ method by default
    return loader


