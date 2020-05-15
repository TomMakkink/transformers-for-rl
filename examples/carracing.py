        # if len(obs.shape) == 1:
        #      # Process a single observation 
        #     obs = obs.reshape(4, 1, 4)      # [seq_len, batch_size, features]
        #     obs = self.transformer(obs)
        #     obs = obs.view(-1)              # [seq_len * features]
        # elif len(obs.shape) == 2:
        #     # Process a batch of observations 
        #     batch_size, obs_dim = obs.shape
        #     obs = obs.reshape(4, batch_size, 4)
        #     obs = self.transformer(obs)
        #     obs = obs.view(batch_size, -1)