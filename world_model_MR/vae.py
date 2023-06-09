import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from utils.training_data import TrainingData

class VAE_net(nn.Module):
    def __init__(self, img_size = 96, img_ch=3, z_size=32, h_size = 1024):
        super(VAE_net, self).__init__()
        self.img_ch = img_ch
        self.z_size = z_size
        self.img_size = img_size

        if img_size == 64:
            self.encoder = nn.Sequential(
                nn.Conv2d(img_ch, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten()
            )
            self.decoder = nn.Sequential(
                nn.Unflatten(1,(h_size,1,1)),
                nn.ConvTranspose2d(h_size, 128, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, img_ch, kernel_size=6, stride=2),
                nn.Sigmoid(),
            )
        
        if img_size == 84:
            self.encoder = nn.Sequential(
            nn.Conv2d(img_ch, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Flatten()
            )
            self.decoder = nn.Sequential(
                nn.Unflatten(1,(h_size,1,1)),
                nn.ConvTranspose2d(h_size, 128, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, kernel_size=10, stride=2),
                nn.Sigmoid(),
            )

        if img_size == 96:
            self.encoder = nn.Sequential(
                nn.Conv2d(img_ch, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=8, stride=2),
                nn.ReLU(),
                nn.Flatten()
            )
            self.decoder = nn.Sequential(
                nn.Unflatten(1,(h_size,1,1)),
                nn.ConvTranspose2d(h_size, 128, kernel_size=7, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, img_ch, kernel_size=10, stride=2),
                nn.Sigmoid(),
            )

        self.encoder_fc1 = nn.Linear(h_size, z_size)
        self.encoder_fc2 = nn.Linear(h_size, z_size)
        self.decoder_fc = nn.Linear(z_size, h_size)

    # Function to sample from distribution
    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar/2)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.encoder_fc1(h)
        logvar = self.encoder_fc2(h)
        return mu, logvar
    
    def decode(self,z):
        y = self.decoder_fc(z)
        y = self.decoder(y)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return y, mu, logvar

class VAE():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = VAE_net(img_size=self.args.img_size).to(self.device)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.args.vae_learning_rate)
    
    def encode(self, x):
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z = self.vae.reparameterize(mu, logvar)
        return z
    
    def decode(self, z):
        with torch.no_grad():
            r_states = self.vae.decode(z)
        return r_states

    # Save function
    def save(self, run_name, episode):
        save_path = os.path.join(f"checkpoints/{run_name}/{episode}.pth")

        torch.save({'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)
    
    # Load Function
    def load(self, path, mode = "eval"):
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if mode == "eval":
            self.vae.eval()
            return
        
        if mode == "train":
            self.vae.train()
            return
    
    # Train Function
    def train(self, states):
        dataset = TrainingData(states, states)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.args.vae_batch_size)

        epoch_losses = []
        for epoch in range(self.args.vae_n_epochs):

            batch_losses = []
            for batch, targets in dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                r_states, mus, logvars = self.vae(batch.to(self.device))

                # Compute loss, backward pass and optimize
                r_loss = F.binary_cross_entropy(r_states, targets.to(self.device), reduction='sum')
                kl_loss = - 0.5 * torch.mean(1 + logvars - mus.pow(2) - logvars.exp())
                loss = r_loss + kl_loss
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())
            
            epoch_losses.append(np.mean(batch_losses))

        return np.mean(epoch_losses)


