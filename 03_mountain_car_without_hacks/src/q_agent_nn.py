from typing import List
from collections import namedtuple, deque
from pdb import set_trace as stop

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class QAgentNN:

    def __init__(self, env, learning_rate, discount_factor):

        self.env = env

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.preprocessor = self._init_preprocessor()
        self.nn = self._init_nn()

    def _init_preprocessor(self):

        # normalize position and velocity to have zero mean and unit variance
        scaler = StandardScaler()
        observation_examples = [self.env.observation_space.sample() for x in range(100)]

        return scaler.fit(observation_examples)

    def _init_nn(self):

        model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        return model

    def _preprocess(self, state) -> torch.Tensor:
        return torch.from_numpy(self.preprocessor.transform(state)).float()

    def get_action(self, state):

        state_ = self._preprocess([state])
        q_values = self.nn(state_)
        return np.argmax(q_values)


    # def update_params(self, state, action, reward, next_state):
    def update_params(self, transitions: List[Transition]):



        pass


from tqdm import tqdm
from typing import Union, Callable, Tuple

def train(
    agent,
    env,
    replay_buffer,
    n_episodes: int,
    epsilon: Union[float, Callable]
) -> Tuple[List, List]:

    # For plotting metrics
    reward_per_episode = []
    max_position_per_episode = []

    pbar = tqdm(range(0, n_episodes))
    for i in pbar:

        state = env.reset()

        rewards = 0
        max_position = -99

        # handle case when epsilon is either
        # - a float
        # - or a function that returns a float given the episode nubmer
        epsilon_ = epsilon if isinstance(epsilon, float) else epsilon(i)

        pbar.set_description(f'Epsilon: {epsilon_:.2f}')

        done = False
        while not done:

            if random.uniform(0, 1) < epsilon_:
                # Explore action space
                action = env.action_space.sample()
            else:
                # Exploit learned values
                action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)

            agent.update_parameters(state, action, reward, next_state)

            rewards += reward
            if next_state[0] > max_position:
                max_position = next_state[0]

            state = next_state

        reward_per_episode.append(rewards)
        max_position_per_episode.append(max_position)

    return reward_per_episode, max_position_per_episode




