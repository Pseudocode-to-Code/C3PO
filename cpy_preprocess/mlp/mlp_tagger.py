import torch 
from torch import nn

class MLPTagger(nn.Module):
    def __init__(self, vocab_size, embedding_size, window_size, hidden_size, dropout_p=0.5):
        super(MLPTagger, self).__init__()

        self.embedding_size = embedding_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.hidden_size_2 = int(self.hidden_size/4)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.fc1 = nn.Linear((self.embedding_size * 2*self.window_size), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = dropout_p)
        self.relu = nn.ReLU()

    def forward(self, data):
        emb = self.embedding(data)
        # print('1', emb.size())

        batch_size = emb.size()[0]
        emb = emb.view(batch_size, -1) # Concatenating all 4 words embeddings into one vector using view
        # print('2', emb.size())

        hidden_1 = self.dropout(self.fc1(emb))
        hidden_1 = self.relu(hidden_1)
        # print('2', hidden_1.size())

        hidden_2 = self.dropout(self.fc2(hidden_1))
        hidden_2 = self.relu(hidden_2)
        # print('3', hidden_2.size())

        output = self.sigmoid(self.fc3(hidden_2))
        # print('4', output.size())
        return output 
