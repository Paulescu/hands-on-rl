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


def evaluate(
    agent,
    env,
    n_episodes: int,
    epsilon: Optional[float] = None
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

            if (epsilon is not None) and (random.uniform(0, 1) < epsilon):
                action = env.action_space.sample()
            else:
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