import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class ActorCriticLSTM(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_size=128, lstm_hidden_dim=128
    ):
        super(ActorCriticLSTM, self).__init__()
        state_size = observation_space.shape[1]
        action_size = action_space.n

        self.fc_shared = nn.Linear(state_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=lstm_hidden_dim, num_layers=1
        )
        self.fc_policy = nn.Linear(lstm_hidden_dim, action_size)
        self.fc_value_function = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x, hidden):
        x = F.relu(self.fc_shared(x))
        x = x.unsqueeze(0).unsqueeze(0)
        x, new_hidden = self.lstm(x, hidden)
        x = x.squeeze(0).squeeze(0)
        logits = self.fc_policy(x)
        value = self.fc_value_function(x)
        dist = Categorical(logits=logits)
        return dist, value, new_hidden
