import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers.transformer_wrapper import Transformer
from configs.transformer_config import transformer_config


class ActorCriticTransformer(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCriticTransformer, self).__init__()
        s_size = observation_space.shape[1]
        a_size = action_space.n

        self.fc_shared = nn.Linear(s_size, transformer_config["d_model"])
        self.transformer = Transformer(**transformer_config)
        self.fc_policy = nn.Linear(transformer_config["output_dim"], a_size)
        self.fc_value_function = nn.Linear(transformer_config["output_dim"], 1)

    def reset_mem(self):
        self.transformer.reset_mem()

    def forward(self, x):
        x = F.relu(self.fc_shared(x))
        if len(x.shape) == 2:
            # Stacked Observations
            x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)
            logits = self.fc_policy(x)[-1]
            value = self.fc_value_function(x)[-1]
            dist = Categorical(logits=logits)
        else:
            # Single observations
            x = x.unsqueeze(0).unsqueeze(0)
            x = self.transformer(x)
            x = x.squeeze(0).squeeze(0)
            logits = self.fc_policy(x)
            value = self.fc_value_function(x)
            dist = Categorical(logits=logits)

        return dist, value
