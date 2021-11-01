"""
Q-learing in tabular world

"""

import random
from argparse import ArgumentParser
from pdb import set_trace as stop

from tqdm import tqdm
import numpy as np
import gym
import gym_tictac4



def get_action(q_table, state, allowed_actions, epsilon) -> int:
    """
    Given the state, the current q-function and the exploration probability
    epsilon, choose and return the next agent's action.
    """
    action = None
    if random.uniform(0, 1) < epsilon:
        # random action
        action = random.choice(allowed_actions)
    else:
        # greedy. Select the action with the highest Q-value
        sorted_actions = q_table[state].argsort()[::1]
        for a in sorted_actions:
            if a in allowed_actions:
                action = a
                break

    if action is None:
        raise Exception('This should never happen')

    return action


def get_epsilon(episode):
    """
    """
    if episode < 500000:
        # first 100k steps --> 10%
        return 0.10
    else:
        return 0.01


def train_agent(
    n_episodes: int,
):
    # load environment
    env = gym.make('tictac4-v0')

    # The Q function is a matrix where states are rows, and actions are columns.
    # Each entry in this matrix represents the total payoff the agent
    # could get if at state s took action a, and after that
    # sticked to an optimal policy that maximizes overall reward.
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # stop()

    # hyper-parameters
    epsilon = 0.10
    alpha = 0.1 # <--- TODO: what is this?
    gamma = 0.6 # <--- TODO: what is this?

    # For plotting metrics
    # rewards_first_player = []
    # rewards_second_player = []
    wins = []
    draws = []
    losses = []
    pbar = tqdm(range(n_episodes))
    for episode in pbar:

        state = env.reset()
        done = False
        epoch = 0
        epsilon = get_epsilon(episode)

        while not done:

            # here is where the agent decides WHAT to do.
            # We are using a epsilon-greedy policy
            # with epsilon probability we choose a random action
            # with (1 - epsilon) probability we choose the action with largest Q-value
            # if (epoch % 2) == 0:
            #     # first player's turn
            #     action = get_action(q_table, state, env.allowed_actions, epsilon)
            # else:
            #     # second player's turn
            #     action = get_action((-1) * q_table, state, env.allowed_actions, epsilon)
            action = get_action(q_table, state, env.allowed_actions, epsilon)

            # update the state of the world and collect any rewards.
            next_state, reward, done, info = env.step(action)

            # Bellman equation to refine our approximation
            # to the optimal Q(s, a) value function
            q_table[state, action] = \
                (1 - alpha) * q_table[state, action] + \
                alpha * (abs(reward) + gamma * np.max(q_table[next_state]))

            # updates before we start new iteration
            state = next_state
            epoch += 1

            if done:
                wins.append(1 if reward == 1 else 0)  # first player wins
                draws.append(1 if reward == 0 else 0)
                losses.append(1 if reward == -1 else 0) # first player loses

        if ((episode + 1) % 1000) == 0:

            win_perc = np.array(wins[-1000:]).mean()
            draw_perc = np.array(draws[-1000:]).mean()
            pbar.set_description(f"Win rate: {win_perc:.0%} Draw rate: {draw_perc:.0%} Epsilon: {epsilon:.2f}")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--n_episodes', dest='n_episodes', type=int, required=True)

    args = parser.parse_args()

    train_agent(
        n_episodes=args.n_episodes,
    )