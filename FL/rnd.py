import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from utils.training_data import TrainingData


class RND_Net(nn.Module):
    def __init__(self):
        super(RND_Net, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(32, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
        )

        self.target = nn.Sequential(
            nn.Linear(32, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
        )

        """
        self.predictor = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=8, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,512),
        )

        self.target = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512)
        )
        """
        # Initialize Weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(4))
                m.bias.data.zero_()

        # Set target network parameters as untrainable
        for p in self.target.parameters():
            p.requires_grad = False            

    def forward(self, x):
        return self.predictor(x)

    def get_targets(self, x):
        return self.target(x)

class RND(nn.Module):
    def __init__(self, args):
        super(RND, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnd = RND_Net().to(self.device)
        self.optimizer = optim.Adam(self.rnd.parameters(), lr = self.args.rnd_learning_rate)

    # Save Function
    def save(self, run_name, episode):
        save_path = os.path.join(f"checkpoints/{run_name}/rnd/{episode}.pth")

        torch.save({'rnd_state_dict': self.rnd.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)
    
    # Load function
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.rnd.load_state_dict(checkpoint['rnd_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, x):
        
        targets = self.rnd.get_targets(x)
        
        dataset = TrainingData(x, targets)
        dataLoader = DataLoader(dataset, shuffle=True, batch_size=self.args.rnd_batch_size)

        epoch_losses = []
        for epoch in range (self.args.rnd_n_epochs):

            batch_losses = []
            for batch, targets in dataLoader:
                self.optimizer.zero_grad()
                
                # Send to GPU
                batch, targets = batch.to(self.device), targets.to(self.device)

                # Forward pass and loss computation
                preds = self.rnd(batch)
                loss = F.mse_loss(preds, targets)

                # Backward pass and optimise
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())
            
            epoch_losses.append(np.mean(batch_losses))
        
        return np.mean(epoch_losses)
        

    def reward(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.rnd(x)
            target = self.rnd.get_targets(x)
        loss = F.mse_loss(pred, target)
        return loss.item()





