from time import sleep
from argparse import ArgumentParser
from pdb import set_trace as stop

import pandas as pd
import gym

from src.config import SAVED_AGENTS_DIR

import numpy as np


def plot_policy(agent, positions: np.arange, velocities: np.arange, figsize = None):
    """"""
    data = []
    int2str = {
        0: 'Accelerate Left',
        1: 'Do nothing',
        2: 'Accelerate Right'
    }
    for position in positions:
        for velocity in velocities:

            state = np.array([position, velocity])
            action = int2str[agent.get_action(state)]

            data.append({
                'position': position,
                'velocity': velocity,
                'action': action,
            })

    data = pd.DataFrame(data)

    import seaborn as sns
    import matplotlib.pyplot as plt

    if figsize:
        plt.figure(figsize=figsize)

    colors = {
        'Accelerate Left': 'blue',
        'Do nothing': 'grey',
        'Accelerate Right': 'orange'
    }
    sns.scatterplot(x="position", y="velocity", hue="action", data=data,
                    palette=colors)

    plt.show()
    return data

def show_video(agent, env, sleep_sec: float = 0.1, mode: str = "rgb_array"):

    state = env.reset()
    done = False

    # LAPADULA
    if mode == "rgb_array":
        from matplotlib import pyplot as plt
        from IPython.display import display, clear_output
        steps = 0
        fig, ax = plt.subplots(figsize=(8, 6))

    while not done:

        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        # LAPADULA
        if mode == "rgb_array":
            steps += 1
            frame = env.render(mode=mode)
            ax.cla()
            ax.axes.yaxis.set_visible(False)
            ax.imshow(frame, extent=[env.min_position, env.max_position, 0, 1])
            ax.set_title(f'Steps: {steps}')
            display(fig)
            clear_output(wait=True)
            plt.pause(sleep_sec)
        else:
            env.render()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--agent_file', type=str, required=True)
    parser.add_argument('--sleep_sec', type=float, required=False, default=0.1)
    args = parser.parse_args()

    from src.base_agent import BaseAgent
    agent_path = SAVED_AGENTS_DIR / args.agent_file
    agent = BaseAgent.load_from_disk(agent_path)

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000

    show_video(agent, env, sleep_sec=args.sleep_sec)








