from torch.autograd import Variable
from torch.utils.data import Dataset

class TrainingData(Dataset):

    def __init__(self, x, y):
        self.x = Variable(x)
        self.y = Variable(y)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return len(self.x)

class RNNTrainingData(Dataset):

    def __init__(self, x, history, y):
        self.x = Variable(x)
        self.history = Variable(history)
        self.y = Variable(y)

    def __getitem__(self, index):
        return (self.x[index], self.history[index], self.y[index])
    
    def __len__(self):
        return len(self.x)