import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.training_data import TrainingData

class RNN_net(nn.Module):
    def __init__(self, args):
        super(RNN_net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.input_size = self.args.rnn_input_size # z-size + action (32 + 1)
        self.seq_len = self.args.rnn_sequence_length
        self.h_size = self.args.rnn_h_size
        self.n_layers = self.args.rnn_n_layers 
        self.output_size = self.args.rnn_output_size

        self.rnn = nn.GRU(self.input_size, self.h_size, self.n_layers, batch_first=True)
        #self.rnn = nn.LSTM(self.input_size, self.h_size, self.n_layers, batch_first=True)

        # Linear layer to predict death
        self.death = nn.Linear(self.h_size, 1)

        # Deterministic output with only the mean of a single Gaussian (i.e. vanilla, with no MDN output layer)
        if self.args.rnn_deterministic:
            print("RNN mode: Deterministic")
            self.output_layer = nn.Linear(self.h_size, self.output_size)

        # MDN output
        else:
            print("RNN mode: MDN")
            self.n_mixtures = self.args.rnn_n_mixtures
            self.pi = nn.Linear(self.h_size, self.output_size*self.n_mixtures)
            self.sigma = nn.Linear(self.h_size, self.output_size*self.n_mixtures)
            self.mu = nn.Linear(self.h_size, self.output_size*self.n_mixtures)
        

    def get_mdn_coef(self, y):
        pi = self.pi(y) # Pi shape: 32 x 160
        pi = pi.view(y.shape[0], self.n_mixtures, self.output_size) # Reshape: batch_size x output_size x n_mixture (32 x 32 x 5)
        pi = F.softmax(pi, 1) 
        sigma = self.sigma(y)
        sigma = sigma.view(y.shape[0], self.n_mixtures, self.output_size)
        sigma = torch.exp(sigma - torch.logsumexp(sigma, dim=1, keepdim=True)) # log-sum-exp trick to prevent under- and overflow
        mu = self.mu(y)
        mu = mu.view(y.shape[0], self.n_mixtures, self.output_size)
        return pi, mu, sigma

    def forward(self, x):
        # Initialize hidden cell and cell state
        h0 = Variable(torch.zeros(self.n_layers, x.shape[0], self.h_size)).to(self.device)
        #c0 = Variable(torch.zeros(self.n_layers, x.shape[0], self.h_size)).to(self.device) # LSTM

        # Forward Pass
        _, h = self.rnn(x, h0) # GRU
        #_, (h,_) = self.rnn(x, (h0,c0)) # LSTM

        h = h[-1] # Format for output layer
        
        # Deterministic output
        if self.args.rnn_deterministic:
            z_pred = self.output_layer(h)
            death = torch.sigmoid(self.death(h)).squeeze()
            return z_pred, death
        
        # MDN output
        else:
            # Predict sucessor z-state
            pi, mu, sigma = self.get_mdn_coef(h)
            death = torch.sigmoid(self.death(h)).squeeze()

            return pi, mu, sigma, death

class RNN():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnn = RNN_net(args).to(self.device)

        # gradient clipping prior to backward pass
        for p in self.rnn.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -self.args.rnn_clip_value, self.args.rnn_clip_value))

        self.optimizer = optim.Adam(self.rnn.parameters(), lr = self.args.rnn_learning_rate)

    # Save function
    def save(self, run_name, episode):

        save_path = os.path.join(f"checkpoints/{run_name}/rnn/{episode}.pth")

        torch.save({'model_state_dict': self.rnn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)
    
    # Load function
    def load(self, path, mode = "eval"):
        checkpoint = torch.load(path, map_location=self.device)
        self.rnn.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if mode == "eval":
            self.rnn.eval()
            return
        
        if mode == "train":
            self.rnn.train()
            return

    def mdn_loss(self, pi, mu, sigma, targets):
        eps = 1e-5 # Stability constant to ensure non-zero values, in order to avoid NaN-values when taking log of zero
        targets = targets.unsqueeze(1)
        sigma = sigma + eps # Avoids taking log of 0 in gaussian.log_prob() function
        gaussian = torch.distributions.Normal(loc=mu, scale=sigma)
        result = torch.exp(gaussian.log_prob(targets))
        result = torch.sum(result*pi, dim=1)
        result = F.relu(result) # Ensures non-negativity to avoid NaN-values when taking log
        result = -torch.log(result + eps)
        return torch.mean(result)
    
    def sample(self, pi, mu, sigma):
        z = torch.sum(pi * torch.normal(mu,sigma), dim=1)
        return z

    def mean(self, pi, mu):
        z = torch.sum(pi * mu, dim = 1)
        return z

    def train(self, z_states, targets):
        dataset = TrainingData(z_states, targets)
        dataLoader = DataLoader(dataset, shuffle=True, batch_size=self.args.rnn_batch_size)

        epoch_losses = []
        for epoch in range (self.args.rnn_n_epochs):

            batch_losses = []
            for batch, targets in dataLoader:
                self.optimizer.zero_grad()

                # Deterministic forward pass and loss computation
                if self.args.rnn_deterministic:
                    z_preds, death = self.rnn(batch.to(self.device))
                    rnn_loss = F.mse_loss(z_preds, targets[ : , : -1].to(self.device))
                    bce_loss = F.binary_cross_entropy(death, targets[ : ,-1].to(self.device))
                    loss = rnn_loss + bce_loss

                # MDN forward pass and loss computation
                else:
                    pi, mu, sigma, death = self.rnn(batch.to(self.device))
                    mdn_loss = self.mdn_loss(pi, mu, sigma, targets[ : , : -1].to(self.device))
                    bce_loss = F.binary_cross_entropy(death, targets[ : ,-1].to(self.device))
                    loss = mdn_loss + bce_loss

                # Backward pass and optimise
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())
            
            epoch_losses.append(np.mean(batch_losses))
        
        return np.mean(epoch_losses)