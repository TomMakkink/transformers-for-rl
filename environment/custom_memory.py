from bsuite.environments.memory_chain import MemoryChain
import dm_env
from dm_env import specs
import numpy as np
import random


class CustomMemoryChain(MemoryChain):
    def __init__(self, memory_length: int, num_bits: int = 1, seed: int = None):
        super(CustomMemoryChain, self).__init__(memory_length, num_bits, seed)
        assert (
            memory_length > num_bits
        ), "Memory length must be greater than number of bits for custom Memory Chain."
        self._context_timesteps = random.sample(range(memory_length - 1), num_bits)
        self._context_timesteps.sort()
        self._context_index = 0

    def _get_observation(self):
        """Observation of form [time, query, num_bits of context]."""
        obs = np.zeros(shape=(1, self._num_bits + 2), dtype=np.float32)
        # Show the time, on every step.
        obs[0, 0] = 1 - self._timestep / self._memory_length

        # Show the query, on the last step
        if self._timestep == self._memory_length - 1:
            obs[0, 1] = self._query

        # Show part of the context, on varied steps
        if self._timestep == 0:
            obs[0, 2:] = 2 * self._context - 1
            # print(f"Full Context: {obs}")

        # Show part of the context, on varied steps
        if self._timestep in self._context_timesteps:
            obs[0, 2 + self._context_index] = 2 * self._context[self._context_index] - 1
            self._context_index += 1

        return obs

    # def _step(self, action: int) -> dm_env.TimeStep:
    #     observation = self._get_observation()
    #     self._timestep += 1

    #     if self._timestep - 1 < self._memory_length:
    #     # On all but the last step provide a reward of 0.
    #     return dm_env.transition(reward=0., observation=observation)
    #     if self._timestep - 1 > self._memory_length:
    #     raise RuntimeError('Invalid state.')  # We shouldn't get here.

    #     if action == self._context[self._query]:
    #     reward = 1.
    #     self._total_perfect += 1
    #     else:
    #     reward = -1.
    #     self._total_regret += 2.
    #     return dm_env.termination(reward=reward, observation=observation)

    def _reset(self):
        self._context_timesteps = random.sample(
            range(self._memory_length - 1), self._num_bits
        )
        self._context_timesteps.sort()
        self._context_index = 0

        self._timestep = 0
        self._episode_mistakes = 0
        self._context = self._rng.binomial(1, 0.5, self._num_bits)
        self._query = self._rng.randint(self._num_bits)
        observation = self._get_observation()

        self._timestep += 1

        return dm_env.restart(observation)
