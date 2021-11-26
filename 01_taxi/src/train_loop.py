from typing import Tuple, List
import random

import numpy as np
from tqdm import tqdm


def train(
    agent,
    env,
    n_episodes: int,
    epsilon: float
) -> Tuple[List, List]:

    # For plotting metrics
    timesteps_per_episode = []
    penalties_per_episode = []

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:

            if random.uniform(0, 1) < epsilon:
                # Explore action space
                action = env.action_space.sample()
            else:
                # Exploit learned values
                action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)

            agent.update_parameters(state, action, reward, next_state)

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        timesteps_per_episode.append(epochs)
        penalties_per_episode.append(penalties)

    return timesteps_per_episode, penalties_per_episode

def train_many_runs(
    agent,
    env,
    n_episodes: int,
    epsilon: float,
    n_runs: int,
) -> Tuple[List, List]:
    """
    Calls 'train' many times, stores results and averages them out.
    """
    timesteps = np.zeros(shape=(n_runs, n_episodes))
    penalties = np.zeros(shape=(n_runs, n_episodes))

    for i in range(0, n_runs):

        agent.reset()

        timesteps[i, :], penalties[i, :] = train(
            agent, env, n_episodes, epsilon
        )
    timesteps = np.mean(timesteps, axis=0).tolist()
    penalties = np.mean(penalties, axis=0).tolist()

    return timesteps, penalties