import torch
import torch.nn as nn
import torch.nn.functional as F

class U_t_train(nn.Module):
    """
    u(t) = 1 非発話
    　　    0 発話
    """
    def __init__(self, num_layers = 1, input_size=64, hidden_size = 64):
        super(U_t_train, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()
        

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        assert len(x.shape) == 2 , print('data shape is incorrect.')
        
        x = self.dr1(self.relu1(self.fc1(x)))
        y = self.fc2(x)
        return y