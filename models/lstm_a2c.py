import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class LSTMA2C(nn.Module):
    def __init__(self, observation_space, action_space, h_size=128):
        super(LSTMA2C, self).__init__()
        s_size = observation_space.shape[1]
        a_size = action_space.n
        self.fc_shared = nn.Linear(s_size, h_size)
        self.lstm_h_dim = 128
        self.lstm = nn.LSTM(
            input_size=h_size, hidden_size=self.lstm_h_dim, num_layers=1
        )
        self.fc_policy = nn.Linear(self.lstm_h_dim, a_size)
        self.fc_value_function = nn.Linear(self.lstm_h_dim, 1)

    def forward(self, x, hidden):
        x = F.relu(self.fc_shared(x))
        x = x.unsqueeze(0).unsqueeze(0)
        x, new_hidden = self.lstm(x, hidden)
        x = x.squeeze(0).squeeze(0)
        logits = self.fc_policy(x)
        value = self.fc_value_function(x)
        dist = Categorical(logits=logits)
        return dist, value, new_hidden
