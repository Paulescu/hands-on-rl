from time import sleep
from argparse import ArgumentParser
from pdb import set_trace as stop
from typing import Optional

import pandas as pd
import gym

from src.config import SAVED_AGENTS_DIR

import numpy as np


def show_video(agent, env, sleep_sec: float = 0.1, seed: Optional[int] = 0, mode: str = "rgb_array"):

    env.seed(seed)
    state = env.reset()

    # LAPADULA
    if mode == "rgb_array":
        from matplotlib import pyplot as plt
        from IPython.display import display, clear_output
        steps = 0
        fig, ax = plt.subplots(figsize=(8, 6))

    done = False
    while not done:

        action = agent.act(state, epsilon=0.001)
        state, reward, done, info = env.step(action)

        # LAPADULA
        if mode == "rgb_array":
            steps += 1
            frame = env.render(mode=mode)
            ax.cla()
            ax.axes.yaxis.set_visible(False)
            ax.imshow(frame)
            ax.set_title(f'Steps: {steps}')
            display(fig)
            clear_output(wait=True)
            plt.pause(sleep_sec)
        else:
            env.render()
            sleep(sleep_sec)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--agent_id', type=str, required=True)
    parser.add_argument('--sleep_sec', type=float, required=False, default=0.1)
    args = parser.parse_args()

    from src.base_agent import BaseAgent
    agent_path = SAVED_AGENTS_DIR / args.agent_file
    agent = BaseAgent.load_from_disk(agent_path)

    from src.q_agent import QAgent


    env = gym.make('CartPole-v1')
    # env._max_episode_steps = 1000

    show_video(agent, env, sleep_sec=args.sleep_sec)








