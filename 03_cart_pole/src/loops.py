from typing import Tuple, List, Callable, Union, Optional
import random
from pathlib import Path
from collections import deque
from pdb import set_trace as stop

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter



def train(
    agent,
    env,
    n_episodes: int,
    log_dir: Optional[Path] = None,
    max_steps: Optional[int] = float("inf"),
    n_episodes_evaluate_agent: Optional[int] = 100,
    freq_episodes_evaluate_agent: int = 200,
) -> None:

    # Tensorborad log writer
    logging = False
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
        logging = True

    reward_per_episode = []
    steps_per_episode = []
    global_step_counter = 0

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()

        rewards = 0
        steps = 0
        done = False
        while not done:

            action = agent.act(state)

            # agents takes a step and the environment throws out a new state and
            # a reward
            next_state, reward, done, info = env.step(action)

            # agent observes transition and stores it for later use
            agent.observe(state, action, reward, next_state, done)

            # learning happens here, through experience replay
            agent.replay()

            global_step_counter += 1
            steps += 1
            rewards += reward
            state = next_state

        # log to Tensorboard
        if logging:
            writer.add_scalar('train/rewards', rewards, i)
            writer.add_scalar('train/steps', steps, i)
            writer.add_scalar('train/epsilon', agent.epsilon, i)
            writer.add_scalar('train/replay_memory_size', len(agent.memory), i)

        reward_per_episode.append(rewards)
        steps_per_episode.append(steps)

        # if (i > 0) and (i % freq_episodes_evaluate_agent) == 0:
        if (i + 1) % freq_episodes_evaluate_agent == 0:
            # evaluate agent
            eval_rewards, eval_steps = evaluate(agent, env,
                                                n_episodes=n_episodes_evaluate_agent,
                                                epsilon=0.01)

            # from src.utils import get_success_rate_from_n_steps
            # success_rate = get_success_rate_from_n_steps(env, eval_steps)
            print(f'Reward mean: {np.mean(eval_rewards):.2f}, std: {np.std(eval_rewards):.2f}')
            print(f'Num steps mean: {np.mean(eval_steps):.2f}, std: {np.std(eval_steps):.2f}')
            # print(f'Success rate: {success_rate:.2%}')
            if logging:
                writer.add_scalar('eval/avg_reward', np.mean(eval_rewards), i)
                writer.add_scalar('eval/avg_steps', np.mean(eval_steps), i)
            # writer.add_scalar('eval/success_rate', success_rate, i)

        if global_step_counter > max_steps:
            break


def evaluate(
    agent,
    env,
    n_episodes: int,
    epsilon: Optional[float] = None,
    seed: Optional[int] = 0,
) -> Tuple[List, List]:

    from src.utils import set_seed
    set_seed(env, seed)

    # output metrics
    reward_per_episode = []
    steps_per_episode = []

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()
        rewards = 0
        steps = 0
        done = False
        while not done:

            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)

            rewards += reward
            steps += 1
            state = next_state

        reward_per_episode.append(rewards)
        steps_per_episode.append(steps)

    return reward_per_episode, steps_per_episode