import torch
import torch.nn as nn


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size,padding_idx=0)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.gru = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu=nn.ReLU()
        self.fc_2=nn.Linear(hidden_size, 1)
        self.hidden_output=0
        self.out1=0
        self.out2=0


    def forward(self, x):
        x_bool = (x == 0).int()
        x_lens = torch.argmax(x_bool, dim=1)
        x_lens[x_lens == 0] = x.shape[1]
        x_lens = x_lens.to(dtype=torch.int64).to('cpu')
        x = self.embedding(x)
        packed_output = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        _, out = self.gru(packed_output)
        out = out.transpose(0, 1)
        out = out.reshape(out.size(0), -1)
        self.hidden_output=out
        out = self.fc_1(out)
        self.out1=out
        out=self.relu(out)
        self.out2=out
        out=self.fc_2(out)
        return out
    def get_hidden_layer_output(self,x):
        self.forward(x)
        return self.hidden_output
    def get_fc_layer_output1(self,x):
        self.forward(x)
        return self.out1
    def get_fc_layer_output2(self,x):
        self.forward(x)
        return self.out2


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size,padding_idx=0)
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu=nn.ReLU()
        self.fc_2=nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_bool = (x == 0).int()
        x_lens = torch.argmax(x_bool, dim=1)
        x_lens[x_lens == 0] = x.shape[1]
        x_lens = x_lens.to(dtype=torch.int64).to('cpu')
        x = self.embedding(x)
        packed_output = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        _, out = self.LSTM(packed_output)
        out=out[0]
        out = out.transpose(0, 1)
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out=self.relu(out)
        out=self.fc_2(out)
        return out