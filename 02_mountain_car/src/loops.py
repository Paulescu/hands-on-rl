from typing import Tuple, List, Callable, Union, Optional
import random

from tqdm import tqdm

def train(
    agent,
    env,
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

            action = agent.get_action(state, epsilon_)

            next_state, reward, done, info = env.step(action)

            agent.update_parameters(state, action, reward, next_state, epsilon_)

            rewards += reward
            if next_state[0] > max_position:
                max_position = next_state[0]

            state = next_state

        reward_per_episode.append(rewards)
        max_position_per_episode.append(max_position)

    return reward_per_episode, max_position_per_episode


def evaluate(
    agent,
    env,
    n_episodes: int,
    epsilon: Optional[Union[float, Callable]] = None
) -> Tuple[List, List]:

    # For plotting metrics
    reward_per_episode = []
    max_position_per_episode = []

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()

        rewards = 0
        max_position = -99

        done = False
        while not done:

            epsilon_ = None
            if epsilon is not None:
                epsilon_ = epsilon if isinstance(epsilon, float) else epsilon(i)
            action = agent.get_action(state, epsilon_)

            next_state, reward, done, info = env.step(action)

            agent.update_parameters(state, action, reward, next_state, epsilon_)

            rewards += reward
            if next_state[0] > max_position:
                max_position = next_state[0]

            state = next_state

        reward_per_episode.append(rewards)
        max_position_per_episode.append(max_position)

    return reward_per_episode, max_position_per_episode

if __name__ == '__main__':

    # environment
    import gym
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000

    # agent
    from src.sarsa_agent import SarsaAgent
    alpha = 0.1
    gamma = 0.6
    agent = SarsaAgent(env, alpha, gamma)

    rewards, max_positions = train(agent, env, n_episodes=100, epsilon=0.1)