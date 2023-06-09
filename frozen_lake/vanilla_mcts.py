import os
import numpy as np
from math import *
from copy import deepcopy
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Custom classes
from vae import VAE
from utils.training_data import TrainingData
from utils.get_pixel_state import get_pixel_state
from vanilla_mcts_net import MCTS_PolicyNet, MCTS_ValueNet
import torch.optim as optim

# ---------- Classes ----------- #

class Node:
    def __init__(self, sim_env, state, reward_ext=0, done=False, parent = None, action_index = None, prior = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.children = []
        self.total_value = 0 # W(s,a)
        self.mean_value = 0 # Q(s,a)
        self.visit_count = 0 # N(s,a)
        self.sim_env = sim_env
        self.state = state
        self.reward_ext = reward_ext
        self.done = done
        self.parent = parent
        self.action_index = action_index

        self.prior = prior
        self.puct_const = 1.0

    # Score for choosing the next node doing selection phase
    def puct_score(self):
        c1 = 1.25
        c2 = 19.625
        #return self.mean_value + self.puct_const * self.prior * (sqrt(self.parent.visit_count) / (1 + self.visit_count))
        return self.mean_value + self.prior * (sqrt(self.parent.visit_count) / (1 + self.visit_count)) \
                * (c1 + log( (self.parent.visit_count + c2 + 1) / c2 ))
    
    def detach_parent(self):
        self.parent = None

    def expand(self, priors):
              
        if self.done:
            return
        
        for action in range (self.sim_env.action_space.n):
            sim_env = deepcopy(self.sim_env)
            _, reward_ext, terminated, truncated, _ = sim_env.step(action)
            state = get_pixel_state(sim_env)
            done = np.any([terminated, truncated], axis=0)
            self.children.append(Node(  sim_env = sim_env, state = state, reward_ext = reward_ext, done = done, parent = self, 
                                        action_index = action, prior = priors[action].item()))

    # Tree policy for choosing the next action according to tree statistics.
    def tree_policy(self, temperature = 1):
        
        if temperature > 0:
            visit_counts = [child.visit_count ** (1 / temperature) for child in self.children]
            visit_count_sum = sum(visit_counts)
            visit_count_distribution = [visit_count / visit_count_sum for visit_count in visit_counts]
            new_root = np.random.choice(self.children, p=visit_count_distribution)        
            return new_root, new_root.action_index          

        else:
            max_visit_count = max(child.visit_count for child in self.children)
            max_children = [child for child in self.children if child.visit_count == max_visit_count]         
            max_child = random.choice(max_children)
            return max_child, max_child.action_index      

class MCTS():
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.temperature = self.args.mcts_temperature

        # MCTS net setup
        self.policy = MCTS_PolicyNet(args).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = self.args.rnn_learning_rate)

        self.value_net = MCTS_ValueNet(args).to(self.device)
        for p in self.value_net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -self.args.rnn_clip_value, self.args.rnn_clip_value))
        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr = self.args.rnn_learning_rate)

        # Vae setup
        self.vae = VAE(args)

    def dirichlet_noise(self, priors):
        eps = self.args.mcts_dirichlet_eps
        alpha = self.args.mcts_dirichlet_alpha
        dirichlet = torch.distributions.Dirichlet(torch.full(priors.shape, alpha))
        noisy_priors = (1- eps) * priors + eps * dirichlet.sample().to(self.device)

        return noisy_priors
        
    # MCTS algorithm
    def search(self, root):

        for i in range (self.args.mcts_n_simulations):
            node = root
            
            # SELECT: Traverse tree by selecting children with the highest puct score, until we reach a leaf node
            while node.children:
                max_score = max(child.puct_score() for child in node.children)
                best_children = [child for child in node.children if child.puct_score() == max_score]
                node = random.choice(best_children)

            # EVALUATE
            state = node.state.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get z-state
                z_state = self.vae.encode(state)

                # Ask mcts networks for action preferences, extrinsic and intrinsic expected return
                priors = self.policy(z_state)
                v_ext, = self.value_net(z_state)
                priors, v_ext = priors.squeeze(), v_ext.squeeze().item()
            
            # Compute total value
            value = node.reward_ext + v_ext*(1-int(node.done))

            # Add dirichlet noise, if root node (Additional exploration)
            if not node.parent:
                priors = self.dirichlet_noise(priors)
            
            # EXPAND
            node.expand(priors)

            # BACKUP
            while node:
                node.visit_count += 1
                #node.total_value += value
                #node.mean_value = node.total_value / node.visit_count
                node.mean_value = (node.visit_count * node.mean_value + value) / (node.visit_count + 1)
                node = node.parent
        
        # Find best action according statistics and make subtree
        new_tree, next_action = root.tree_policy(self.temperature)
        new_tree.detach_parent()
        return new_tree, next_action


    def train(self, states, targets, args):
        
        # ------------ Train Policy Net ------------ #
        z_states = self.vae.encode(states)    
        policy_targets = targets[ : , : -1] # Everything except the last item in the array
        
        policy_dataset = TrainingData(z_states, policy_targets)
        policy_dataLoader = DataLoader(policy_dataset, shuffle=True, batch_size=args.rnn_batch_size)

        epoch_loss_lst = []
        for epoch in range (args.rnn_n_epochs):
            
            batch_loss_lst = []
            for batch, policy_targets in policy_dataLoader:
                self.policy_optimizer.zero_grad()

                # Send to GPU
                batch, policy_targets = batch.to(self.device), policy_targets.to(self.device)

                # Forward pass
                priors = self.policy(batch)

                # Loss computation, Backward pass and optimization
                loss = F.cross_entropy(priors, policy_targets)
                loss.backward()
                self.policy_optimizer.step()
            
                batch_loss_lst.append(loss.item())
            
            epoch_loss_lst.append(np.mean(batch_loss_lst))
        policy_loss = np.mean(epoch_loss_lst)

        # ------------ Train Value Net ------------ #
        value_targets = targets[ : , -1] # Last item in the array

        value_dataset = TrainingData(z_states, value_targets)
        value_dataLoader = DataLoader(value_dataset, shuffle=True, batch_size=args.rnn_batch_size)        

        epoch_loss_lst = []
        for epoch in range (args.rnn_n_epochs):
            
            batch_loss_lst = []
            for batch, targets in value_dataLoader:
                self.value_net_optimizer.zero_grad()

                # Send to GPU
                batch, targets = batch.to(self.device), targets.to(self.device)

                # Forward pass
                v_exts = self.value_net(batch)

                # Loss computation, Backward pass and optimization
                loss = F.mse_loss(v_exts.squeeze(), targets)
                loss.backward()
                self.value_net_optimizer.step()
            
                batch_loss_lst.append(loss.item())
            
            epoch_loss_lst.append(np.mean(batch_loss_lst))
        
        value_loss = np.mean(epoch_loss_lst)
        
        return policy_loss, value_loss

    def save(self, run_name, update):
        save_file = os.path.join(f"checkpoints/{run_name}/mcts/{update}.pth")
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                    'value_net_state_dict': self.value_net.state_dict(),
                    'value_net_optimizer_state_dict': self.value_net_optimizer.state_dict(),
                },
                save_file)  
                    
    def load(self, path, mode = "eval"):
        checkpoint = torch.load(path,  map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.value_net_optimizer.load_state_dict(checkpoint['value_net_optimizer_state_dict'])

        if mode == "eval":
            self.policy.eval()
            self.value_net.eval()
            return
        
        if mode == "train":
            self.policy.train()
            self.value_net.train()
            return
