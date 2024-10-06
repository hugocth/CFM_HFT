import torch
from torch import nn

from utils import init_rnn

class biLSTMClassifier(nn.Module):

    def __init__(self, LSTM_params, MLP_params, embeddings):
        super().__init__()

        self.LSTM_params = LSTM_params
        self.MLP_params = MLP_params
        self.norm = nn.LayerNorm((100, self.LSTM_params["input_size"]))
        self.LSTM = nn.LSTM(**LSTM_params)
        lstm_out_size = LSTM_params["hidden_size"]*4 if LSTM_params["bidirectional"] else LSTM_params["hidden_size"]*2

        self.MLP = nn.Sequential()
        for i in range(1, MLP_params["n_layers"]+1):

            if i == MLP_params["n_layers"]:
                self.MLP.add_module(f"dense_{i}", nn.Linear(lstm_out_size//i, MLP_params["out_features"]))
                break

            self.MLP.add_module(f"dense_{i}", nn.Linear(lstm_out_size//i, lstm_out_size//(i+1)))
            self.MLP.add_module(f"act_{i}", nn.SELU())
        
        init_rnn(self.LSTM, 'xavier')

        self.embeddings = embeddings
        if self.embeddings:
            self.embed_venue = nn.Embedding(6,8)    # 6 possible venues
            self.embed_action = nn.Embedding(3,8)   # A,D,U actions
            # self.embed_side = nn.Embedding(2,8)     # A or B side
            self.embed_trade = nn.Embedding(2,8)    # True or False trade

    def forward(self, x):
    
        if self.embeddings:
            embeddings_venue = self.embed_venue(x[:,:,-3].long())
            embeddings_action = self.embed_venue(x[:,:,-2].long())
            embeddings_trade = self.embed_venue(x[:,:,-1].long())
            x = torch.concat((x[:,:,:-3], embeddings_venue, embeddings_action, embeddings_trade), axis=-1)

        lstm_out, (hn, cn) = self.LSTM(self.norm(x))
        tokens = torch.concat((lstm_out.max(axis=1)[0], lstm_out.mean(axis=1)), axis=-1)
        out = self.MLP(tokens)

        return out


class biGRUClassifier(nn.Module):

    def __init__(self, GRU_params, MLP_params, embeddings):
        super().__init__()

        self.GRU_params = GRU_params
        self.MLP_params = MLP_params
        self.norm = nn.LayerNorm((100, self.GRU_params["input_size"]))
        self.GRU = nn.GRU(**GRU_params)
        GRU_out_size = GRU_params["hidden_size"]*4 if GRU_params["bidirectional"] else GRU_params["hidden_size"]*2

        self.MLP = nn.Sequential()
        for i in range(1, MLP_params["n_layers"]+1):

            if i == MLP_params["n_layers"]:
                self.MLP.add_module(f"dense_{i}", nn.Linear(GRU_out_size//i, MLP_params["out_features"]))
                break

            self.MLP.add_module(f"dense_{i}", nn.Linear(GRU_out_size//i, GRU_out_size//(i+1)))
            self.MLP.add_module(f"act_{i}", nn.SELU())
        
        init_rnn(self.GRU, 'xavier')

        self.embeddings = embeddings
        if self.embeddings:
            self.embed_venue = nn.Embedding(6,8)    # 6 possible venues
            self.embed_action = nn.Embedding(3,8)   # A,D,U actions
            # self.embed_side = nn.Embedding(2,8)     # A or B side
            self.embed_trade = nn.Embedding(2,8)    # True or False trade

    def forward(self, x):

        if self.embeddings:
            embeddings_venue = self.embed_venue(x[:,:,-3])
            embeddings_action = self.embed_venue(x[:,:,-2])
            embeddings_trade = self.embed_venue(x[:,:,-1])
            x = torch.concat((x[:,:,-3:], embeddings_venue, embeddings_action, embeddings_trade), axis=-1)

        GRU_out, hn = self.GRU(self.norm(x))
        tokens = torch.concat((GRU_out.max(axis=1)[0], GRU_out.mean(axis=1)), axis=-1)
        out = self.MLP(tokens)

        return out

if __name__ == "__main__":
    pass

