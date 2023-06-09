import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.autograd import Variable

class MCTS_PolicyNet(nn.Module):
    def __init__(self, args):
        super(MCTS_PolicyNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.input_size = self.args.rnn_input_size
        self.policy_layers = nn.ModuleList()

        # Input Layer
        self.policy_layers.append(nn.Linear(self.input_size, self.args.mlp_h_size))
        self.policy_layers.append(nn.ELU())

        # Hidden Layers
        for i in range (self.args.mlp_n_layers - 1):
            self.policy_layers.append(nn.Linear(self.args.mlp_h_size, self.args.mlp_h_size))
            self.policy_layers.append(nn.ELU())
        
        # Output layer
        self.policy_layers.append(nn.Linear(self.args.mlp_h_size, self.args.rnn_n_actions))
        self.policy_layers.append(nn.Softmax(dim=-1))

        self.policy = nn.Sequential(*self.policy_layers)

    def forward(self, zs):
        return self.policy(zs)
    
class MCTS_ValueNet(nn.Module):
    def __init__(self, args):
        super(MCTS_ValueNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        # RNN setup
        self.input_size = self.args.rnn_input_size
        self.h_size = self.args.rnn_h_size
        self.n_layers = self.args.rnn_n_layers
        self.seq_len = self.args.rnn_sequence_length
        self.rnn = nn.GRU(self.input_size, self.h_size, self.n_layers, batch_first=True)

        # Output heads
        self.value_ext = nn.Linear(self.h_size, 1)
        self.value_int = nn.Linear(self.h_size, 1)

    def forward(self, zs):
        h0 = Variable(torch.zeros(self.n_layers, zs.shape[0], self.h_size)).to(self.device)
        _, h = self.rnn(zs, h0)
        h = h[-1]
        v_ext = self.value_ext(h)
        v_int = self.value_int(h)

        return v_ext, v_int

    



    
    
