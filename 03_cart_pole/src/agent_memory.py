from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class AgentMemory:

    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)

        # stop()

        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)