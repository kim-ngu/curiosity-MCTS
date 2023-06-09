import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from copy import deepcopy
import torchvision.transforms as T

class Framestack:
    def __init__(self, env):        
        self.states = [self.get_pixel_state(env) for _ in range (4)]

    def append(self,x):
        self.states.pop(0)
        self.states.append(x)
    
    def get_pixel_state(self, env):
        render_env = deepcopy(env)
        state = render_env.render()
        state = torch.from_numpy(state).float()
        state = state.permute(2,0,1)
        state = T.functional.rgb_to_grayscale(state)
        state = torch.div(state, 255.0)
        transform = T.Resize((84,84))
        state = transform(state)
        return state.squeeze(0)
    
    def get(self, env):
        state = self.get_pixel_state(env)
        self.append(state)
        return torch.stack(self.states).unsqueeze(0)

class ReplayBuffer:
    def __init__(self, device, num_steps, obs_shape):
        self.states = torch.zeros((num_steps,) + obs_shape).to(device)
        self.actions = torch.zeros(num_steps).to(device)
        self.logprobs = torch.zeros(num_steps).to(device)
        self.rewards = torch.zeros(num_steps).to(device)
        self.dones = torch.zeros(num_steps).to(device)
        self.values = torch.zeros(num_steps).to(device)

# Initialise layers as in PPO-paper
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AtariAgent(nn.Module):
    def __init__(self, env):
        super(AtariAgent, self).__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, env.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)    

    def get_value(self, x):
        return self.critic(self.cnn(x))

    def get_action_and_value(self, x, action=None):
        h = self.cnn(x)
        logits = self.actor(h) # Unnormalized action probabilities
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), self.critic(h)
    