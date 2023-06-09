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
        self.input_size = self.args.rnn_input_size
        self.value_layers = nn.ModuleList()

        # Input Layer
        self.value_layers.append(nn.Linear(self.input_size, self.args.mlp_h_size))
        self.value_layers.append(nn.ELU())

        # Hidden Layers
        for i in range (self.args.mlp_n_layers - 1):
            self.value_layers.append(nn.Linear(self.args.mlp_h_size, self.args.mlp_h_size))
            self.value_layers.append(nn.ELU())
        
        # Output layer
        self.value_layers.append(nn.Linear(self.args.mlp_h_size, 1))

        self.value_net = nn.Sequential(*self.value_layers)

    def forward(self, zs):
        return self.value_net(zs)
    



    
    
