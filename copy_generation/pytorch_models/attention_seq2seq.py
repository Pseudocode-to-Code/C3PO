import torch 
from torch import nn 
import random 

class AttnEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_p):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_p)

        # input_size is size of vocabulary (pseudocode)
        self.embedding = nn.Embedding(input_size, embedding_size)

        # num_layers is Stacked LSTM
        # self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=dropout_p)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        # As bidirectional is true, hidden and cell will be 2 hidden and cell states

        # Reduce biredirectional LSTM states to single LSTM state using FC layer
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)


    def forward(self, x):
        # x is (max_seq_len, BATCH_SIZE/no. of sentenses,)

        embedding = self.dropout(self.embedding(x))
        # shape is (max_seq_len, BATCH, emb_size)

        encoder_states, (hidden, cell) = self.lstm(embedding)

        # print('Hidden size', hidden.size())
        # print('Encoder states', encoder_states.size())

        # Hidden size will be (2, BATCH, hidden_size). We concatenate along the hidden_size dim to make it ()
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        # print('Hidden size after', hidden.size())

        # encoder_states has states for all previous LSTM timesteps not only the latest ones like in hidden
        return encoder_states, hidden, cell 


class AttnDecoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_p):
        super().__init__() 

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_p)

        # input_size is size of vocabulary (pseudocode)
        self.embedding = nn.Embedding(input_size, embedding_size)
        # num_layers is Stacked LSTM
        # To the LSTM we will send the hidden states as well as 1 word emb at a time. In vanilla Seq2Seq it was only 1 word emb at a time
        # self.lstm = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers, dropout=dropout_p)
        self.lstm = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers)

        # hidden_size*2 is from the encoder hidden states (bidirectional) 
        # The other hidden_size is previous hidden state of Decoder (hj)
        # (eij = * hj)
        self.energy = nn.Linear(hidden_size*3 , 1)
        self.softmax = nn.Softmax(dim=0) # Softmax for attention weight calc
        self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell): #hidden is prev state of decoder, encoder_states is all states of encoder
        # As we are passing 1 word at a time, No seq_len dimension here
        # We need to add that dimension as 1 
        x = x.unsqueeze(0)
        
        embedding = self.dropout(self.embedding(x))
        # Embedding shape: (1, Batchsize, embsize)

        seq_len = encoder_states.shape[0]

        # print('Hidden shape', hidden.size())
        h_reshape = hidden.repeat(seq_len, 1, 1)
        # print('Hidden after reshape', h_reshape.size())

        # print('New size', torch.cat((h_reshape, encoder_states), dim = 2).size())

        energy = self.relu(self.energy(torch.cat((h_reshape, encoder_states), dim = 2)))
        # print('Enery size', energy.size())

        attention = self.softmax(energy) # Shape will be (max_seq_length, N, 1)
        # print('Attention', attention.size())

        # Attention shape will be (Seq, N, 1)
        # Encoderstates shape will be (Seq, N, 2*hidden_size)

        # Matrix Multiplication
        attention = attention.permute(1, 2, 0) # Switch BATCH and seq_len dims
        encoder_states = encoder_states.permute(1, 0, 2)

        # After permute Attention shape will be (N, 1, Seq)
        # Encoderstates shape will be (N, Seq, 2*hidden_size)

        # Matmul: (N, 1, 2*hidden_size)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2) # Switch it back to (1, N, hidden_size*2)

        rnn_input = torch.cat((context_vector, embedding), dim = 2)
        # It will be (1, N, hidden_size*2 + emb_size)

        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        preds = self.fc(outputs)
        preds = preds.squeeze(0)

        return preds, hidden, cell 


class AttnSeq2Seq(nn.Module):
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

        encoder_states, hidden, cell = self.encoder(source)

        # First input to decoder
        x = target[0]

        for t in range(1, target_len):
            # Pass first input and context vector/thought vector
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # Picking next input to decoder (either predicted or actual word)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs