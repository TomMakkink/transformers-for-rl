from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, state_size, action_size, memory):
        super(Agent, self).__init__()

    @abstractmethod
    def optimize_network(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def collect_experience(self, state, action, reward, next_state, done):
        pass
