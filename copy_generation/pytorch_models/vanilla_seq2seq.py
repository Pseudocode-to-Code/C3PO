import torch 
from torch import nn 
import random 

class S2SEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_p):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_p)

        # input_size is size of vocabulary (pseudocode)
        self.embedding = nn.Embedding(input_size, embedding_size)

        # num_layers is Stacked LSTM
        # self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)


    def forward(self, x):
        # x is (max_seq_len, BATCH_SIZE/no. of sentenses,)

        embedding = self.dropout(self.embedding(x))
        # embedding = self.embedding(x)
        # shape is (max_seq_len, BATCH, emb_size)

        outputs, (hidden, cell) = self.lstm(embedding)

        return hidden, cell 


class S2SDecoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_p):
        super().__init__() 

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_p)

        # input_size is size of vocabulary (pseudocode)
        self.embedding = nn.Embedding(input_size, embedding_size)

        # num_layers is Stacked LSTM
        # self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

        #TODO Should we add a softmax

    def forward(self, x, hidden, cell):
        # As we are passing 1 word at a time, No seq_len dimension here
        # We need to add that dimension as 1 
        x = x.unsqueeze(0)

        
        embedding = self.dropout(self.embedding(x))
        # Embedding shape: (1, Batchsize, embsize)

        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        preds = self.fc(outputs)

        preds = preds.squeeze(0)

        return preds, hidden, cell 


class VanillaSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, target_vocab_size, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        # target_vocab_size = len(english.vocab)

        # (Seq len, B, VocabSize for softmax)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        # print('Seq2Seq outputs:', outputs.size())

        hidden, cell = self.encoder(source)

        # First input to decoder
        x = target[0]

        for t in range(1, target_len):
            # Pass first input and context vector/thought vector
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # Picking next input to decoder (either predicted or actual word)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs