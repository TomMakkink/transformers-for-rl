class Agent:
    def __init__(self, model, env, device):
        super(Agent, self).__init__()
        self.device = device

    def optimise_network(self, *args):
        pass

    def act(self, state):
        pass

    def collect_experience(self, *args):
        pass
