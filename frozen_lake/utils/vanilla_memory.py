import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

def one_hot_encode(x, space_size):
    return [1 if i == x else 0 for i in range(space_size)]

class Memory():
    def __init__(self, env, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.args = args
        self.states = []
        self.rewards_ext = []
        self.actions = []
        self.actions_one_hot = []

    def clear(self):
        self.states.clear()
        self.rewards_ext.clear()
        self.actions.clear()
        self.actions_one_hot.clear()

    def append(self, state, action, reward_ext):
        self.states.append(state)

        # One-hot encoding for targets
        action_one_hot = torch.tensor(one_hot_encode(action, self.env.action_space.n), dtype=torch.float32)
        self.actions_one_hot.append(action_one_hot)

        action = torch.tensor([action], dtype=torch.float32)
        action /= self.env.action_space.n
        self.actions.append(action)

        reward_ext = torch.tensor([self.args.rnd_coef_ext*reward_ext], dtype=torch.float32)
        self.rewards_ext.append(reward_ext)
    
    def append_action(self, action):
        action = torch.tensor([action], dtype=torch.float32)
        action /= self.env.action_space.n
        self.actions.append(action)

    def get_expected_returns(self, rewards, gamma):
        expected_returns = []
        
        for i in range(len(rewards)):
            expected_return = 0

            # No expected return at last entry
            if i == len(rewards) - 1:
                expected_return = torch.tensor([0], dtype=torch.float32)
            else:
                discount = gamma
                for j in range (i, len(rewards)-1):
                    expected_return += discount*rewards[j+1]
                    discount *= gamma
            
            expected_returns.append(expected_return)
        
        return torch.stack(expected_returns)

    def make_mcts_training_data(self):

        # Prepare data
        states = torch.stack(self.states)
        actions_one_hot = torch.stack(self.actions_one_hot)
        expected_returns_ext = self.get_expected_returns(self.rewards_ext, self.args.rnd_gamma_ext)

        targets = [ torch.cat( (actions_one_hot[i], expected_returns_ext[i]) ) for i in range (len(actions_one_hot))]


        return states, targets
    

    

    