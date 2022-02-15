from typing import Tuple, List, Callable, Union, Optional
from pathlib import Path
from pdb import set_trace as stop

import numpy as np
from tqdm import tqdm


def train(
    agent,
    env,
    n_episodes: int,
    freq_episodes_evaluate_agent: Optional[int] = None,
    n_episodes_evaluate_agent: Optional[int] = 100,
    run = None,
) -> None:

    # are we going to log stuff to Neptune?
    logging = True if run is not None else False

    reward_per_episode = []
    steps_per_episode = []
    best_avg_reward = -1

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

            steps += 1
            rewards += reward
            state = next_state

        # log to Tensorboard
        if logging:
            run['train/reward'].log(rewards)
            run['train/steps'].log(steps)
            run['train/epsilon'].log(agent.epsilon)
            run['train/replay_memory_size'].log(len(agent.memory))

        reward_per_episode.append(rewards)
        steps_per_episode.append(steps)

        if (freq_episodes_evaluate_agent is not None) and ((i + 1) % freq_episodes_evaluate_agent == 0):
            # evaluate agent
            eval_rewards, eval_steps = evaluate(agent, env,
                                                n_episodes=n_episodes_evaluate_agent,
                                                epsilon=0.0)

            print(f'Reward mean: {np.mean(eval_rewards):.2f}, std: {np.std(eval_rewards):.2f}')
            print(f'Num steps mean: {np.mean(eval_steps):.2f}, std: {np.std(eval_steps):.2f}')

            if logging:
                run['eval/avg_reward'].log(np.mean(eval_rewards))
                run['eval/avg_steps'].log(np.mean(eval_steps))

                if np.mean(eval_rewards) > best_avg_reward:
                    # save model checkpoint
                    agent.save_checkpoint()


def evaluate(
    agent,
    env,
    n_episodes: int,
    epsilon: Optional[float] = None,
    # seed: Optional[int] = 0,
) -> Tuple[List, List]:

    # from src.utils import set_seed
    # set_seed(env, seed)

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