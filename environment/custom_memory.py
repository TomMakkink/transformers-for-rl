from bsuite.environments.memory_chain import MemoryChain
import dm_env
from dm_env import specs
import numpy as np
import random


class CustomMemoryChain(MemoryChain):
    def __init__(self, memory_length: int, num_bits: int, seed: int = 0):
        super().__init__(memory_length, num_bits, seed)
        assert (
            memory_length >= num_bits
        ), "Memory length must be greater than number of bits for custom Memory Chain."
        assert num_bits % 2 != 0, "Num bits must be an odd number"

        random.seed(seed)

        self._context_timesteps = random.sample(range(memory_length), num_bits)
        self._context_timesteps.sort()
        self._context_index = 0

        # Ignore query
        self._query = 0.0

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
            obs_copy = obs.copy()
            obs_copy[0, 2:] = 2 * self._context - 1
            # print(f"Raw Context: {self._context}")
            # print(f"Context Observed: {obs_copy}")

        # Show part of the context, on varied steps
        if self._timestep in self._context_timesteps:
            obs[0, 2 + self._context_index] = 2 * self._context[self._context_index] - 1
            self._context_index += 1

        return obs

    def _step(self, action: int) -> dm_env.TimeStep:
        observation = self._get_observation()
        self._timestep += 1

        if self._timestep - 1 < self._memory_length:
            # On all but the last step provide a reward of 0.
            return dm_env.transition(reward=0.0, observation=observation)
        if self._timestep - 1 > self._memory_length:
            raise RuntimeError("Invalid state.")  # We shouldn't get here.

        # Convert context from [0, 1] to [-1, 1]
        context = 2 * self._context - 1

        # If sum(context) > 0, action 1
        # If sum(context) < 0, action 0
        if (sum(context) > 0 and action == 1) or (sum(context) < 0 and action == 0):
            reward = 1.0
            self._total_perfect += 1
        else:
            reward = -1.0
            self._total_regret += 2.0
        return dm_env.termination(reward=reward, observation=observation)

    def _reset(self):
        self._context_timesteps = random.sample(
            range(self._memory_length), self._num_bits
        )
        self._context_timesteps.sort()
        self._context_index = 0

        self._timestep = 0
        self._episode_mistakes = 0
        self._context = self._rng.binomial(1, 0.5, self._num_bits)
        # Ignore query
        self._query = 0.0
        observation = self._get_observation()

        self._timestep += 1

        return dm_env.restart(observation)


# Length: 3
# Bits: 3
# Length: 10
# Bits: 3, 5, 7, 9
# Length: 30
# Bits: 3, 5, 7, 9, 17, 25
# Length: 100
# Bits: 3, 5, 7, 9, 17, 25, 29, 35, 39

custom_memory_sweep = (
    "memory_custom/0",
    "memory_custom/1",
    "memory_custom/2",
    "memory_custom/3",
    "memory_custom/4",
    "memory_custom/5",
    "memory_custom/6",
    "memory_custom/7",
    "memory_custom/8",
    "memory_custom/9",
    "memory_custom/10",
    "memory_custom/11",
    "memory_custom/12",
)


def load_custom_memory_env(experiment: str):
    memory_length, num_bits = {
        "memory_custom/0": (3, 3),
        "memory_custom/1": (5, 3),
        "memory_custom/2": (5, 5),
        "memory_custom/3": (10, 3),
        "memory_custom/4": (10, 5),
        "memory_custom/5": (10, 7),
        "memory_custom/6": (10, 9),
        "memory_custom/7": (30, 3),
        "memory_custom/8": (30, 5),
        "memory_custom/9": (30, 7),
        "memory_custom/10": (30, 9),
        "memory_custom/11": (30, 17),
        "memory_custom/12": (30, 25),
    }.get(experiment)

    return CustomMemoryChain(memory_length=memory_length, num_bits=num_bits)
