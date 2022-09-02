import torch
from torch import nn
from mlp_tagger import MLPTagger
import pandas as pd
import numpy as np
import importlib
import preprocess
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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


## Read spoc tokenized input
cols = {0: 'pseudo', 1: 'code'}

train_df = pd.read_csv('../../data/input-tok-train-shuf.tsv', header=None, delimiter='\t')
train_df.rename(columns=cols, inplace=True)

preprocess.tokenize_column(train_df, col_to_tokenize='pseudo', tokenized_col_name='pseudo_tokens', inplace=True)
preprocess.tokenize_column(train_df, col_to_tokenize='code', tokenized_col_name='code_tokens', inplace=True)

# Create binary seq
code_binary_seq = train_df.apply(preprocess.create_binary_seq_from_row, args=('code_tokens', 'pseudo_tokens'), axis=1)
train_df['code_binary_seq'] = code_binary_seq
train_df

pseudo_voc = Vocabulary('pseudo_voc')
pseudo_voc.build_vocabulary(train_df, 'pseudo_tokens')



## Training hyperparams
embedding_size = 100
window_size = 2
hidden_size = 100
epochs = 50
dropout_p = 0.5

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
mlp = MLPTagger(len(pseudo_voc), embedding_size, window_size, hidden_size, dropout_p)
mlp.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(mlp.parameters())

writer = SummaryWriter("runs/mlp_tagger")
step = 0
running_loss = 0

# Training loop
for epoch in range(epochs):
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0], unit='row'):
        numer = pseudo_voc.numericalize(row['pseudo_tokens'])

        train = torch.zeros(len(numer), 4, dtype=torch.int64, device=device)
        # print(train.size())

        for i, word in enumerate(numer):
            train[i][0] = numer[i-2] if i-2 >=0 else pseudo_voc.stoi['[PAD]']
            train[i][1] = numer[i-1] if i-1 >=0 else pseudo_voc.stoi['[PAD]']
            train[i][2] = numer[i+1] if i+1 < len(numer) else pseudo_voc.stoi['[PAD]']
            train[i][3] = numer[i+2] if i+2 < len(numer) else pseudo_voc.stoi['[PAD]']

        labels = torch.Tensor(row['code_binary_seq'], device=device)
        labels = labels.unsqueeze(-1)
        
        out = mlp(train)
        # print(out)
        # print(labels)

        loss = criterion(out, labels)

        writer.add_scalar('loss', loss.item(), global_step=step)
        running_loss += loss.item()
        # print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        # print("=====")

    writer.add_scalar('Epoch loss', running_loss, global_step=epoch)
    running_loss = 0