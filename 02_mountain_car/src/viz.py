from time import sleep
from argparse import ArgumentParser

import gym

from src.config import SAVED_AGENTS_DIR


def viz(agent, env, sleep_sec: float = 0.1):

    state = env.reset()
    done = False
    while not done:

        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()
        sleep(sleep_sec)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--agent_file', type=str, required=True)
    parser.add_argument('--initial_position', type=float, required=False)
    parser.add_argument('--initial_speed', type=float, required=False)
    args = parser.parse_args()

    from src.base_agent import BaseAgent
    agent_path = SAVED_AGENTS_DIR / args.agent_file
    agent = BaseAgent.load_from_disk(agent_path)

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000

    viz(agent, env)








