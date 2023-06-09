import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

def one_hot_encode(x, space_size):
    return [1. if i == x else 0. for i in range(space_size)]

class Memory():
    def __init__(self, env, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.args = args

        self.states = []
        self.rewards = []
        self.deaths = []
        self.actions = []

    def clear(self):
        self.states.clear()
        self.rewards.clear()
        self.deaths.clear()
        self.actions.clear()
    
    def append_state(self, state):
        state = torch.from_numpy(state).float()
        state = state.permute(2,0,1) # H x W x C -> C x H x W
        state = torch.div(state, 255.0) # Normalize to range [0-1]
        self.states.append(state)
    
    def append_action(self, action):
        action = torch.tensor([action], dtype=torch.float32)
        action = torch.div(action, self.env.action_space.n) # Normalize to range [0-1]
        self.actions.append(action)

    def append_reward(self, reward):
        reward = torch.tensor([reward], dtype=torch.float32)
        self.rewards.append(reward)

    def append_death(self, death):
        death = torch.tensor([death], dtype=torch.float32)
        self.deaths.append(death)
    
    def get_states(self):
        return torch.stack(self.states).to(self.device)
    
    def make_rnn_training_data(self, z_states):

        # Prepare data for concatenation
        z_states = [z.squeeze() for z in z_states]
        actions = torch.stack(self.actions).to(self.device)
        rewards = torch.stack(self.rewards).to(self.device)
        deaths = torch.stack(self.deaths).to(self.device)

        # Make (z-state, action) pairs
        za_pairs = [torch.cat( (z_states[i], actions[i]) ) for i in range (self.args.n_steps)]
        za_pairs = torch.stack(za_pairs)

        # Make sequences and targets
        n_sequences = self.args.n_steps - self.args.rnn_sequence_length - 1
        target_step =  self.args.rnn_sequence_length + 1

        za_sequences = [za_pairs[i : i + self.args.rnn_sequence_length] for i in range (n_sequences)]
        targets = [torch.cat( (z_states[i + target_step], rewards[i + target_step], deaths[i + target_step])) for i in range (n_sequences)]

        # Append death examples n extra times
        if self.args.death_augment:
            for i in range (n_sequences):
                if deaths[i + target_step] == 1:
                    for _ in range (self.args.death_reps):
                        za_sequences.append(za_sequences[i])
                        targets.append(targets[i])
                
        return torch.stack(za_sequences), torch.stack(targets)   
    

    